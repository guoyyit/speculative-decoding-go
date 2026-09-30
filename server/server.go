// Package server runs many concurrent speculative-generation requests and
// schedules verification batches.
package server

import (
	"context"

	"github.com/guoyyit/speculative-decoding-go/cache"
	"github.com/guoyyit/speculative-decoding-go/engine"
	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
	"github.com/guoyyit/speculative-decoding-go/stats"
)

// Request is one generation request.
type Request struct {
	ID        int64
	Prompt    []model.TokenID
	MaxTokens int
	Seed      uint64
}

// Response is one generation result.
type Response struct {
	ID     int64
	Tokens []model.TokenID
	Err    error
}

// Server schedules concurrent requests and verification batches.
type Server struct {
	draft      model.Proposer
	target     model.Verifier
	defaultCfg engine.Config
	stats      *stats.Counters
}

// New constructs an empty server skeleton.
func New(draft model.Proposer, target model.Verifier, batchSize int, defaultCfg engine.Config) *Server {
	// batchSize documents the verifier's batched forward-pass capacity. The
	// simulated models are pure functions of their inputs, so grouping
	// requests into verification batches changes only wall-clock timing,
	// never any request's output distribution.
	return &Server{
		draft:      draft,
		target:     target,
		defaultCfg: defaultCfg,
		stats:      stats.New(),
	}
}

// Submit runs one request concurrently and returns its response channel.
func (s *Server) Submit(parent context.Context, req Request) <-chan Response {
	ch := make(chan Response, 1)
	go func() {
		defer s.recoverRequest(req.ID, ch)
		ch <- s.run(parent, req)
	}()
	return ch
}

// Stats returns the shared diagnostic counters.
func (s *Server) Stats() *stats.Counters { return s.stats }

// Close releases server resources.
func (s *Server) Close() {}

func (s *Server) run(parent context.Context, req Request) (resp Response) {
	resp.ID = req.ID

	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	// Every request gets isolated sequence state and an independent
	// deterministic random stream derived solely from its own seed.
	seqCache := cache.New(s.target.VocabSize())
	seqCache.Append(req.Prompt...)
	seqCache.Commit()

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = s.defaultCfg.MaxTokens
	}
	targetLen := len(req.Prompt) + maxTokens
	cfg := s.defaultCfg
	cfg.MaxTokens = targetLen

	draftRNG, targetRNG := sampling.NewStreamPair(req.Seed ^ uint64(req.ID) ^ 0x51EED1)
	eng := engine.New(s.draft, s.target, draftRNG, targetRNG, seqCache, s.stats, cfg)

	tokens, err := eng.Generate(ctx)
	if err != nil {
		resp.Err = err
		resp.Tokens = seqCache.Tokens()
		return resp
	}
	resp.Tokens = stripPrompt(tokens, len(req.Prompt))
	return resp
}

func (s *Server) recoverRequest(id int64, ch chan<- Response) {
	if r := recover(); r != nil {
		s.stats.RecordFailure()
		ch <- Response{ID: id, Err: ErrRequestPanic{V: r}}
	}
}

func stripPrompt(tokens []model.TokenID, promptLen int) []model.TokenID {
	if len(tokens) <= promptLen {
		return nil
	}
	out := make([]model.TokenID, len(tokens)-promptLen)
	copy(out, tokens[promptLen:])
	return out
}

// ErrRequestPanic is returned when a request goroutine panics, so one bad
// request never takes down the server or other requests.
type ErrRequestPanic struct {
	V interface{}
}

func (e ErrRequestPanic) Error() string {
	return "speculative server: request panicked"
}
