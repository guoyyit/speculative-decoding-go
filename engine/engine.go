// Package engine drives speculative generation for a single request.
package engine

import (
	"context"
	"errors"
	"time"

	"github.com/guoyyit/speculative-decoding-go/cache"
	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
	"github.com/guoyyit/speculative-decoding-go/stats"
)

// Fallback reasons recorded in diagnostics (re-exported for callers).
const (
	ReasonVerifyTimeout = stats.ReasonVerifyTimeout
	ReasonVerifyError   = stats.ReasonVerifyError
)

// Config configures the generation engine.
type Config struct {
	MaxDraftTokens  int
	MaxTokens       int
	VerifyTimeoutMS int
}

// Engine owns one request's generation loop.
type Engine struct {
	draft     model.Proposer
	target    model.Verifier
	draftRNG  sampling.Source
	targetRNG sampling.Source
	cache     *cache.Cache
	stats     *stats.Counters
	cfg       Config
}

// New constructs an empty request engine skeleton.
func New(draft model.Proposer, target model.Verifier, draftRNG, targetRNG sampling.Source,
	seqCache *cache.Cache, counters *stats.Counters, cfg Config) *Engine {
	return &Engine{
		draft: draft, target: target,
		draftRNG: draftRNG, targetRNG: targetRNG,
		cache: seqCache, stats: counters, cfg: cfg,
	}
}

// Generate runs speculative decoding until MaxTokens are produced, the
// request context is cancelled, or an unrecoverable error occurs. On
// verification timeout or transient verification error the round is fully
// rolled back and replaced by one non-speculative target token; no partial
// candidate is ever emitted.
func (e *Engine) Generate(ctx context.Context) ([]model.TokenID, error) {
	for e.cache.Len() < e.cfg.MaxTokens {
		if err := ctx.Err(); err != nil {
			e.stats.RecordFailure()
			return e.cache.Tokens(), err
		}

		snapshot := e.cache.Snapshot()
		outcome, err := e.runStep(ctx)
		if err != nil {
			// Discard every tentative state from the failed speculative round.
			e.cache.Rollback(snapshot)
			e.stats.RecordRollback()

			// Request-level cancellation aborts the whole request instead of
			// silently falling back.
			if ctx.Err() != nil {
				e.stats.RecordFailure()
				return e.cache.Tokens(), ctx.Err()
			}

			// Per-step timeout or transient error: plain single-token step.
			reason := ReasonVerifyError
			if errors.Is(err, context.DeadlineExceeded) {
				reason = ReasonVerifyTimeout
			}
			tok, fallbackErr := e.fallback(ctx)
			if fallbackErr != nil {
				e.stats.RecordFailure()
				return e.cache.Tokens(), fallbackErr
			}
			e.cache.Append(tok)
			e.cache.Commit()
			e.stats.RecordFallback(reason)
			continue
		}

		e.commitOutcome(outcome)
	}
	return e.cache.Tokens(), nil
}

func (e *Engine) runStep(ctx context.Context) (*sampling.StepOutcome, error) {
	prefix := e.cache.Tokens()
	if e.cfg.VerifyTimeoutMS > 0 {
		stepCtx, cancel := context.WithTimeout(ctx, time.Duration(e.cfg.VerifyTimeoutMS)*time.Millisecond)
		defer cancel()
		return sampling.Step(stepCtx, e.draft, e.target, prefix, e.cfg.MaxDraftTokens, e.draftRNG, e.targetRNG)
	}
	return sampling.Step(ctx, e.draft, e.target, prefix, e.cfg.MaxDraftTokens, e.draftRNG, e.targetRNG)
}

func (e *Engine) commitOutcome(outcome *sampling.StepOutcome) {
	proposed := len(outcome.Proposed)
	remaining := e.cfg.MaxTokens - e.cache.Len()
	if outcome.RejectedAt >= 0 {
		// Accepted prefix plus the resampled correction token. Rejected
		// draft tokens are never appended to the cache.
		committed := append(outcome.Accepted, outcome.Correction)
		if len(committed) > remaining {
			committed = committed[:remaining]
		}
		e.cache.Append(committed...)
		e.cache.Commit()
		e.stats.RecordRollback()
		e.stats.RecordStep(proposed, len(outcome.Accepted), len(committed))
		return
	}

	output := len(outcome.Accepted)
	accepted := outcome.Accepted
	if len(accepted) > remaining {
		accepted = accepted[:remaining]
		output = len(accepted)
	}
	e.cache.Append(accepted...)
	if outcome.HasBonus && remaining-len(accepted) >= 1 {
		e.cache.Append(outcome.Bonus)
		output++
	}
	e.cache.Commit()
	e.stats.RecordStep(proposed, len(outcome.Accepted), output)
}

func (e *Engine) fallback(ctx context.Context) (model.TokenID, error) {
	tok, _, err := sampling.FallbackStep(ctx, e.target, e.cache.Tokens(), e.targetRNG)
	return tok, err
}
