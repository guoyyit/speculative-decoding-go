package spec

import (
	"context"
	"time"

	"github.com/guoyyit/speculative-decoding-go/kv"
	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
)

// StepOutcome classifies a single speculative step.
type StepOutcome int

const (
	OutcomeSuccess StepOutcome = iota
	OutcomeFallback
	OutcomeFailure
)

func (o StepOutcome) String() string {
	switch o {
	case OutcomeSuccess:
		return "success"
	case OutcomeFallback:
		return "fallback"
	default:
		return "failure"
	}
}

// Fallback causes recorded in StepResult.
const (
	CauseVerifyTimeout = "verify_timeout"
)

// StepResult is the diagnostic record of one generation step.
type StepResult struct {
	Outcome       StepOutcome
	Emitted       int
	Proposed      int
	Accepted      int
	RolledBack    bool
	FallbackCause string
}

// Config configures the speculative decoder.
type Config struct {
	// DraftK is the maximum number of tokens proposed per step.
	DraftK int
	// VerifyTimeoutMS bounds one parallel verification; on timeout the step
	// falls back to a single non-speculative target token.
	VerifyTimeoutMS int
}

// Verifier is the target verification surface used by a Decoder. The
// default implementation is *model.Target; the service supplies a batching
// verifier that groups concurrent calls. Both return identical rows for
// identical inputs, so scheduling cannot change output distributions.
type Verifier interface {
	Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error)
}

// Decoder runs speculative decoding for one independent generation request.
// Each request owns its Decoder (and its own seeded sampler and cache), so
// concurrent requests never share state.
type Decoder struct {
	target   *model.Target
	draft    *model.Draft
	cache    *kv.Cache
	sampler  *sampling.Sampler
	cfg      Config
	stats    Stats
	verifier Verifier
}

// New builds a Decoder. seed fully determines the random stream.
func New(target *model.Target, draft *model.Draft, seed uint64, cacheCapacity int, cfg Config) *Decoder {
	if cfg.DraftK <= 0 {
		cfg.DraftK = 4
	}
	d := &Decoder{
		target:   target,
		draft:    draft,
		cache:    kv.New(cacheCapacity),
		sampler:  sampling.New(seed),
		cfg:      cfg,
		verifier: target,
	}
	return d
}

// WithVerifier overrides the verification path (e.g. with the service's
// batching verifier). It must be called before Generate.
func (d *Decoder) WithVerifier(v Verifier) *Decoder {
	if v != nil {
		d.verifier = v
	}
	return d
}

// Generate runs until maxNewTokens are emitted or EOS is sampled. The token
// slice contains only generated (non-prompt) tokens.
func (d *Decoder) Generate(ctx context.Context, prompt []int, maxNewTokens int) ([]int, []StepResult, error) {
	d.cache.Append(prompt...)
	generated := make([]int, 0, maxNewTokens)
	results := make([]StepResult, 0)
	for len(generated) < maxNewTokens {
		newTokens, res, err := d.step(ctx, maxNewTokens-len(generated))
		results = append(results, res)
		if err != nil {
			return generated, results, err
		}
		// The committed token stream is returned by the step and is
		// independent of the bounded KV cache (which may have evicted old
		// prefix tokens). Honor the remaining generation budget.
		for _, tok := range newTokens {
			if len(generated) >= maxNewTokens {
				break
			}
			generated = append(generated, tok)
		}
		if len(newTokens) == 0 || containsEOS(generated) {
			break
		}
	}
	return generated, results, nil
}

func containsEOS(tokens []int) bool {
	for _, t := range tokens {
		if t == 0 {
			return true
		}
	}
	return false
}

// step performs one speculative iteration (or one vanilla fallback token).
func (d *Decoder) step(ctx context.Context, budget int) ([]int, StepResult, error) {
	d.stats.Steps++
	history := d.cache.Tokens()

	proposal, err := d.draft.Propose(ctx, history, d.cfg.DraftK, d.sampler)
	if err != nil {
		d.stats.FailedSteps++
		return nil, StepResult{Outcome: OutcomeFailure, FallbackCause: "propose_canceled"}, err
	}

	// Speculatively stage the candidates in the working cache. They are only
	// retained if the target accepts them; otherwise the cache is truncated
	// back to this accept point.
	d.cache.Append(proposal.Tokens...)

	verifyCtx := ctx
	var cancel context.CancelFunc
	if d.cfg.VerifyTimeoutMS > 0 {
		verifyCtx, cancel = context.WithTimeout(ctx, time.Duration(d.cfg.VerifyTimeoutMS)*time.Millisecond)
	} else {
		verifyCtx, cancel = context.WithCancel(ctx)
	}
	pRows, verr := d.verifier.Verify(verifyCtx, history, proposal.Tokens)
	cancel()

	if verr != nil {
		// Parent-request cancellation aborts the request outright; it must
		// not produce a token or leave staged candidates behind.
		if ctx.Err() != nil {
			d.cache.Reset(history)
			d.stats.FailedSteps++
			return nil, StepResult{Outcome: OutcomeFailure, FallbackCause: "request_canceled"}, ctx.Err()
		}
		// Verification timeout (or other verify failure): discard every
		// staged candidate and emit exactly one vanilla target token.
		d.cache.Reset(history)
		d.stats.Rollbacks++
		d.stats.TimeoutFallbacks++
		d.stats.FallbackSteps++
		tok := d.vanillaToken(history)
		d.cache.Append(tok)
		d.recordEmitted(1)
		return []int{tok}, StepResult{
			Outcome:       OutcomeFallback,
			Emitted:       1,
			Proposed:      len(proposal.Tokens),
			RolledBack:    true,
			FallbackCause: CauseVerifyTimeout,
		}, nil
	}

	res := StepResult{Outcome: OutcomeSuccess, Proposed: len(proposal.Tokens)}

	// Accept the longest accepted prefix using target probabilities only.
	rejected := false
	accepted := 0
	for accepted < len(proposal.Tokens) {
		tok := proposal.Tokens[accepted]
		p := pRows[accepted][tok]
		q := proposal.Q[accepted][tok]
		u := d.sampler.Float64()
		if u > acceptanceProb(p, q) {
			// Reject at position accepted; resample from the normalized
			// residual distribution norm(max(p - q, 0)).
			adj := residualDistribution(pRows[accepted], proposal.Q[accepted])
			fixed := d.sampler.Sample(adj)
			// Rebuild from the pre-step history copy plus the accepted
			// prefix and the correction token. Truncating the staged cache
			// in place is unsafe when bounded eviction has shifted entries.
			restored := make([]int, 0, len(history)+accepted+1)
			restored = append(restored, history...)
			restored = append(restored, proposal.Tokens[:accepted]...)
			restored = append(restored, fixed)
			d.cache.Reset(restored)
			res.Accepted = accepted
			res.Emitted = accepted + 1
			res.RolledBack = true
			d.stats.Rollbacks++
			d.stats.AcceptedTokens += int64(accepted)
			rejected = true
			break
		}
		accepted++
		// Cache already holds this staged token. An accepted EOS ends the
		// sequence immediately: discard anything beyond it, no bonus draw.
		if tok == 0 {
			eosTokens := append([]int(nil), proposal.Tokens[:accepted]...)
			restored := make([]int, 0, len(history)+accepted)
			restored = append(restored, history...)
			restored = append(restored, eosTokens...)
			d.cache.Reset(restored)
			res.Accepted = accepted
			res.Emitted = accepted
			d.stats.AcceptedTokens += int64(accepted)
			d.stats.ProposedTokens += int64(len(proposal.Tokens))
			d.stats.SuccessSteps++
			d.recordEmitted(res.Emitted)
			return eosTokens, res, nil
		}
	}

	var committed []int
	if !rejected {
		// Every proposed token was accepted (accepted == len(tokens)).
		// Sample one bonus token from the target distribution at the final
		// position, giving up to k+1 new tokens in a single step.
		bonus := d.sampler.Sample(pRows[accepted])
		d.cache.Append(bonus)
		committed = append(append([]int(nil), proposal.Tokens...), bonus)
		res.Accepted = accepted
		res.Emitted = accepted + 1
		d.stats.AcceptedTokens += int64(accepted)
	} else {
		// Rejected at the last accepted boundary: committed prefix plus the
		// resampled correction token. Nothing after the boundary survives.
		fixedCount := accepted + 1
		if fixedCount > budget {
			fixedCount = budget
		}
		committed = make([]int, 0, fixedCount)
		for i := 0; i < accepted && i < budget; i++ {
			committed = append(committed, proposal.Tokens[i])
		}
		if len(committed) < budget {
			// Append the correction token; it was stored as the last element.
			all := d.cache.Tokens()
			committed = append(committed, all[len(all)-1])
		}
	}

	d.stats.ProposedTokens += int64(len(proposal.Tokens))
	d.stats.SuccessSteps++
	d.recordEmitted(res.Emitted)
	return committed, res, nil
}

// vanillaToken produces exactly one token from target.Dist, i.e. the same
// operation as non-speculative generation.
func (d *Decoder) vanillaToken(history []int) int {
	return d.sampler.Sample(d.target.Dist(history))
}

func (d *Decoder) recordEmitted(n int) {
	d.stats.EmittedTokens += int64(n)
}

// acceptanceProb is min(1, p/q), the speculative-decoding acceptance rule.
// It is a probability derived solely from target p and draft q; there is no
// fixed threshold or greedy cutoff.
func acceptanceProb(p, q float64) float64 {
	if p <= 0 {
		return 0
	}
	if q <= 0 {
		return 1
	}
	if r := p / q; r < 1 {
		return r
	}
	return 1
}

// residualDistribution returns norm(max(p - q, 0)), the distribution used to
// resample the correction token at the first rejected position.
func residualDistribution(p, q []float64) []float64 {
	out := make([]float64, len(p))
	sum := 0.0
	for i := range p {
		if d := p[i] - q[i]; d > 0 {
			out[i] = d
			sum += d
		}
	}
	if sum <= 0 {
		// Degenerate guard: p and q agree within precision; fall back to p.
		copy(out, p)
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Stats returns a copy of the per-decoder diagnostic counters.
func (d *Decoder) Stats() Stats { return d.stats }

// Stats is the per-decoder diagnostic counter set.
type Stats struct {
	Steps            int64
	SuccessSteps     int64
	FallbackSteps    int64
	FailedSteps      int64
	ProposedTokens   int64
	AcceptedTokens   int64
	EmittedTokens    int64
	Rollbacks        int64
	TimeoutFallbacks int64
}

// AcceptRate is AcceptedTokens / ProposedTokens.
func (s Stats) AcceptRate() float64 {
	if s.ProposedTokens == 0 {
		return 0
	}
	return float64(s.AcceptedTokens) / float64(s.ProposedTokens)
}

// TokensPerStep is EmittedTokens / Steps.
func (s Stats) TokensPerStep() float64 {
	if s.Steps == 0 {
		return 0
	}
	return float64(s.EmittedTokens) / float64(s.Steps)
}
