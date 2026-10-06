package service

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/spec"
)

// Request is one independent generation request.
type Request struct {
	ID           string
	Prompt       []int
	MaxNewTokens int
	Seed         uint64
}

// Result is the terminal outcome of a request.
type Result struct {
	ID     string
	Tokens []int
	Err    error
	// Stats is this request's exact per-decoder diagnostic snapshot. The
	// service-wide aggregate equals the sum of these over every completed
	// request (plus InFlight).
	Stats spec.Stats
}

// Stats are service-wide diagnostics. The three step paths are counted
// separately: speculative success, non-speculative fallback and failure.
type Stats struct {
	InFlight         int64
	RequestsOK       int64
	RequestsFailed   int64
	SuccessSteps     int64
	FallbackSteps    int64
	FailedSteps      int64
	ProposedTokens   int64
	AcceptedTokens   int64
	EmittedTokens    int64
	Rollbacks        int64
	TimeoutFallbacks int64
	ErrorFallbacks   int64
}

// AcceptRate is AcceptedTokens / ProposedTokens across all requests.
func (s Stats) AcceptRate() float64 {
	if s.ProposedTokens == 0 {
		return 0
	}
	return float64(s.AcceptedTokens) / float64(s.ProposedTokens)
}

// TokensPerStep is EmittedTokens / total steps.
func (s Stats) TokensPerStep() float64 {
	total := s.SuccessSteps + s.FallbackSteps + s.FailedSteps
	if total == 0 {
		return 0
	}
	return float64(s.EmittedTokens) / float64(total)
}

// Config configures the service.
type Config struct {
	Model         model.Config
	Decoder       spec.Config
	CacheCapacity int
	BatchSize     int
	BatchWindowMS int
	Workers       int
}

// Service schedules concurrent requests and shared verification batches.
// Each request gets its own Decoder (own seeded RNG and KV cache); only the
// stateless model calls are grouped, so requests cannot observe each other.
type Service struct {
	cfg       Config
	target    *model.Target
	draft     *model.Draft
	verifier  verifyTarget
	batchers  []*batcher
	workers   int
	closed    atomic.Bool
	closeOnce sync.Once
	routeSeq  atomic.Uint64

	inFlight         atomic.Int64
	requestsOK       atomic.Int64
	requestsFailed   atomic.Int64
	successSteps     atomic.Int64
	fallbackSteps    atomic.Int64
	failedSteps      atomic.Int64
	proposedTokens   atomic.Int64
	acceptedTokens   atomic.Int64
	emittedTokens    atomic.Int64
	rollbacks        atomic.Int64
	timeoutFallbacks atomic.Int64
	errorFallbacks   atomic.Int64
}

// New creates a service and starts its worker/batcher goroutines.
func New(cfg Config) *Service {
	if cfg.Workers <= 0 {
		cfg.Workers = 4
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 8
	}
	if cfg.BatchWindowMS <= 0 {
		cfg.BatchWindowMS = 2
	}
	if cfg.CacheCapacity < model.ContextWindow {
		cfg.CacheCapacity = model.ContextWindow
	}
	s := &Service{
		cfg:      cfg,
		target:   model.NewTarget(cfg.Model),
		draft:    model.NewDraft(cfg.Model),
		verifier: model.NewTarget(cfg.Model),
		batchers: make([]*batcher, cfg.Workers),
		workers:  cfg.Workers,
	}
	for i := range s.batchers {
		b := newBatcher(s.verifier, cfg.BatchSize, time.Duration(cfg.BatchWindowMS)*time.Millisecond)
		s.batchers[i] = b
	}
	return s
}

// newServiceWithVerifier builds a service whose verification batches are
// served by v instead of the simulated target. It exists solely for
// deterministic fault injection in tests (e.g. non-timeout verify errors).
func newServiceWithVerifier(cfg Config, v verifyTarget) *Service {
	s := New(cfg)
	for _, b := range s.batchers {
		b.target = v
	}
	return s
}

// Submit runs one request concurrently. It returns a Result; a request
// failure or cancellation is reported on its own Result and never affects
// other requests.
func (s *Service) Submit(ctx context.Context, req Request) *Result {
	if s.closed.Load() {
		return &Result{ID: req.ID, Err: errServiceClosed}
	}
	s.inFlight.Add(1)
	// Pin the request to one worker/batcher by seed-independent routing.
	idx := int(s.routeSeq.Add(1)-1) % s.workers
	done := make(chan *Result, 1)
	go func() {
		defer s.inFlight.Add(-1)
		res := s.runOne(ctx, req, s.batchers[idx])
		done <- res
	}()
	select {
	case res := <-done:
		return res
	case <-ctx.Done():
		// The goroutine observes ctx cancellation through the decoder; it
		// still completes and releases all resources.
		go func() { <-done }()
		return &Result{ID: req.ID, Err: ctx.Err()}
	}
}

// runOne owns the full lifecycle of one request.
func (s *Service) runOne(parent context.Context, req Request, b *batcher) (res *Result) {
	defer func() {
		if r := recover(); r != nil {
			s.requestsFailed.Add(1)
			res = &Result{ID: req.ID, Err: panicError(r)}
		}
	}()
	decoder := spec.New(s.target, s.draft, req.Seed, s.cfg.CacheCapacity, s.cfg.Decoder)
	decoder.WithVerifier(b)
	tokens, stepResults, err := decoder.Generate(parent, req.Prompt, req.MaxNewTokens)

	st := decoder.Stats()
	s.successSteps.Add(st.SuccessSteps)
	s.fallbackSteps.Add(st.FallbackSteps)
	s.failedSteps.Add(st.FailedSteps)
	s.proposedTokens.Add(st.ProposedTokens)
	s.acceptedTokens.Add(st.AcceptedTokens)
	s.emittedTokens.Add(st.EmittedTokens)
	s.rollbacks.Add(st.Rollbacks)
	s.timeoutFallbacks.Add(st.TimeoutFallbacks)
	s.errorFallbacks.Add(st.ErrorFallbacks)

	if err != nil {
		s.requestsFailed.Add(1)
		return &Result{ID: req.ID, Tokens: tokens, Err: err, Stats: st}
	}
	_ = stepResults
	s.requestsOK.Add(1)
	return &Result{ID: req.ID, Tokens: tokens, Stats: st}
}

// Stats returns an aggregate snapshot.
func (s *Service) Stats() Stats {
	return Stats{
		InFlight:         s.inFlight.Load(),
		RequestsOK:       s.requestsOK.Load(),
		RequestsFailed:   s.requestsFailed.Load(),
		SuccessSteps:     s.successSteps.Load(),
		FallbackSteps:    s.fallbackSteps.Load(),
		FailedSteps:      s.failedSteps.Load(),
		ProposedTokens:   s.proposedTokens.Load(),
		AcceptedTokens:   s.acceptedTokens.Load(),
		EmittedTokens:    s.emittedTokens.Load(),
		Rollbacks:        s.rollbacks.Load(),
		TimeoutFallbacks: s.timeoutFallbacks.Load(),
		ErrorFallbacks:   s.errorFallbacks.Load(),
	}
}

// Close stops background batcher goroutines. It is safe to call once.
func (s *Service) Close() {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		for _, b := range s.batchers {
			b.close()
		}
	})
}
