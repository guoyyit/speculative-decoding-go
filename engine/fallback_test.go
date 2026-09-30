package engine

import (
	"context"
	"testing"
	"time"

	"github.com/guoyyit/speculative-decoding-go/cache"
	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
	"github.com/guoyyit/speculative-decoding-go/stats"
)

// alwaysSlowVerifier always exceeds the verify deadline on batched Verify,
// while the single-token Next (used by fallback) returns immediately.
type alwaysSlowVerifier struct {
	model.Verifier
}

func (alwaysSlowVerifier) Verify(ctx context.Context, _ []model.TokenID, cands []model.TokenID) ([]model.Distribution, error) {
	select {
	case <-time.After(50 * time.Millisecond):
		return nil, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func TestEveryStepTimesOutFallsBackToTarget(t *testing.T) {
	target := model.NewTarget(model.Config{VocabSize: 8, Temperature: 1.1})
	draft := model.NewDraft(model.Config{VocabSize: 8, Temperature: 1.1, Noise: 0.2})
	slow := alwaysSlowVerifier{Verifier: target}

	ctx := context.Background()
	prompt := []model.TokenID{2, 6}
	const n = 20
	const seed = uint64(31337)

	c := cache.New(8)
	c.Append(prompt...)
	c.Commit()
	counters := stats.New()
	draftRNG, targetRNG := sampling.NewStreamPair(seed)
	eng := New(draft, slow, draftRNG, targetRNG, c, counters, Config{
		MaxDraftTokens:  4,
		MaxTokens:       len(prompt) + n,
		VerifyTimeoutMS: 5,
	})
	out, err := eng.Generate(ctx)
	if err != nil {
		t.Fatal(err)
	}
	got := out[len(prompt):]
	if len(got) != n {
		t.Fatalf("expected %d fallback tokens, got %d", n, len(got))
	}

	// With every verification timing out, no draft candidate is ever accepted;
	// each token is sampled directly from the target, consuming exactly one
	// draw per token from the start of the stream.
	want, err := plainGenerate(ctx, target, prompt, n, seed)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			t.Fatalf("fallback token %d = %d, want plain target token %d; got=%v want=%v",
				i, got[i], want[i], got, want)
		}
	}

	s := counters.Snapshot()
	if s.Steps != 0 {
		t.Fatalf("no speculative step should commit, got %d", s.Steps)
	}
	if s.FallbackCount != n {
		t.Fatalf("expected %d timeout fallbacks, got %d", n, s.FallbackCount)
	}
	if s.FallbackReasons[stats.ReasonVerifyTimeout] != n {
		t.Fatalf("timeout reason not recorded: %v", s.FallbackReasons)
	}
	if s.Rollbacks != n {
		t.Fatalf("expected %d rollbacks (one per timed-out round), got %d", n, s.Rollbacks)
	}
	if s.FailureCount != 0 {
		t.Fatalf("timeout fallback must not be a failure, got %d", s.FailureCount)
	}
}
