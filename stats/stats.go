// Package stats records speculative-decoding diagnostics.
package stats

import "sync"

// Kind classifies a request step path.
type Kind int

const (
	KindSpeculativeSuccess Kind = iota
	KindFallback
	KindFailure
)

// Fallback reasons recorded via RecordFallback.
const (
	ReasonVerifyTimeout = "verify_timeout"
	ReasonVerifyError   = "verify_error"
)

// Counters is a thread-safe set of diagnostic counters.
type Counters struct {
	mu              sync.Mutex
	steps           int64
	proposedTokens  int64
	acceptedTokens  int64
	outputTokens    int64
	rollbacks       int64
	fallbackCount   int64
	failureCount    int64
	fallbackReasons map[string]int64
}

// New returns an empty counters skeleton.
func New() *Counters { return &Counters{fallbackReasons: map[string]int64{}} }

// RecordStep records one speculative step. output is the number of tokens the
// step emitted to the request (accepted prefix plus correction or bonus).
func (c *Counters) RecordStep(proposed, accepted, output int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.steps++
	c.proposedTokens += int64(proposed)
	c.acceptedTokens += int64(accepted)
	c.outputTokens += int64(output)
}

// RecordRollback records one cache rollback.
func (c *Counters) RecordRollback() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.rollbacks++
}

// RecordFallback records a non-speculative fallback with a reason.
func (c *Counters) RecordFallback(reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.fallbackCount++
	c.fallbackReasons[reason]++
	c.outputTokens++
}

// RecordFailure records a failed/cancelled request.
func (c *Counters) RecordFailure() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failureCount++
}

// Snapshot is an immutable view of the counters.
type Snapshot struct {
	Steps            int64
	ProposedTokens   int64
	AcceptedTokens   int64
	OutputTokens     int64
	Rollbacks        int64
	FallbackCount    int64
	FailureCount     int64
	FallbackReasons  map[string]int64
	AcceptanceRate   float64
	AvgTokensPerStep float64
}

// Snapshot returns the current counters.
func (c *Counters) Snapshot() Snapshot {
	c.mu.Lock()
	defer c.mu.Unlock()
	s := Snapshot{
		Steps:           c.steps,
		ProposedTokens:  c.proposedTokens,
		AcceptedTokens:  c.acceptedTokens,
		OutputTokens:    c.outputTokens,
		Rollbacks:       c.rollbacks,
		FallbackCount:   c.fallbackCount,
		FailureCount:    c.failureCount,
		FallbackReasons: make(map[string]int64, len(c.fallbackReasons)),
	}
	for reason, n := range c.fallbackReasons {
		s.FallbackReasons[reason] = n
	}
	if c.proposedTokens > 0 {
		s.AcceptanceRate = float64(c.acceptedTokens) / float64(c.proposedTokens)
	}
	totalSteps := c.steps + c.fallbackCount
	if totalSteps > 0 {
		s.AvgTokensPerStep = float64(c.outputTokens) / float64(totalSteps)
	}
	return s
}
