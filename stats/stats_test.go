package stats

import (
	"sync"
	"testing"
)

func TestCountersSnapshot(t *testing.T) {
	c := New()

	// One speculative step: proposed 4, accepted 2, emitted 3 (2 + correction).
	c.RecordStep(4, 2, 3)
	// Second step: full acceptance of 3 plus bonus = 4 emitted.
	c.RecordStep(3, 3, 4)
	c.RecordRollback()
	c.RecordFallback(ReasonVerifyTimeout)
	c.RecordFallback(ReasonVerifyTimeout)
	c.RecordFallback(ReasonVerifyError)
	c.RecordFailure()

	s := c.Snapshot()
	if s.Steps != 2 || s.ProposedTokens != 7 || s.AcceptedTokens != 5 {
		t.Fatalf("unexpected step counters: %+v", s)
	}
	if s.OutputTokens != 3+4+3 { // 3 speculative outputs + 3 fallback tokens
		t.Fatalf("output tokens = %d", s.OutputTokens)
	}
	if s.Rollbacks != 1 {
		t.Fatalf("rollbacks = %d", s.Rollbacks)
	}
	if s.FallbackCount != 3 || s.FallbackReasons[ReasonVerifyTimeout] != 2 ||
		s.FallbackReasons[ReasonVerifyError] != 1 {
		t.Fatalf("fallback counters wrong: %+v", s.FallbackReasons)
	}
	if s.FailureCount != 1 {
		t.Fatalf("failures = %d", s.FailureCount)
	}
	if s.AcceptanceRate != float64(5)/float64(7) {
		t.Fatalf("acceptance rate = %v", s.AcceptanceRate)
	}
	// avg = total output / (spec steps + fallback steps) = 10 / 5
	if s.AvgTokensPerStep != 2.0 {
		t.Fatalf("avg tokens/step = %v", s.AvgTokensPerStep)
	}
}

func TestCountersConcurrentSafe(t *testing.T) {
	c := New()
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				c.RecordStep(3, 2, 3)
				c.RecordRollback()
				c.RecordFallback(ReasonVerifyTimeout)
				c.Snapshot()
			}
		}()
	}
	wg.Wait()
	s := c.Snapshot()
	if s.Steps != 4000 {
		t.Fatalf("lost updates under concurrency: steps=%d", s.Steps)
	}
}
