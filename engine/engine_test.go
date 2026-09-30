package engine

import (
	"context"
	"testing"

	"github.com/guoyyit/speculative-decoding-go/cache"
	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
	"github.com/guoyyit/speculative-decoding-go/stats"
)

func testModels(t *testing.T, noise float64) (*model.DraftModel, *model.TargetModel) {
	t.Helper()
	cfg := model.Config{VocabSize: 8, Temperature: 1.2, Noise: noise}
	return model.NewDraft(cfg), model.NewTarget(cfg)
}

// plainGenerate performs unbatched, non-speculative target-only generation.
// It consumes the same per-request RNG stream contract: one draw per token.
func plainGenerate(ctx context.Context, target *model.TargetModel, prompt []model.TokenID, n int, seed uint64) ([]model.TokenID, error) {
	_, src := sampling.NewStreamPair(seed)
	seq := append([]model.TokenID(nil), prompt...)
	for i := 0; i < n; i++ {
		dist, err := target.Next(ctx, seq)
		if err != nil {
			return nil, err
		}
		seq = append(seq, sampling.Sample(dist, src))
	}
	return seq[len(prompt):], nil
}

func runSpeculative(t *testing.T, ctx context.Context, draft *model.DraftModel, target *model.TargetModel,
	prompt []model.TokenID, n int, seed uint64, counters *stats.Counters) ([]model.TokenID, error) {
	t.Helper()
	c := cache.New(target.VocabSize())
	c.Append(prompt...)
	c.Commit()
	draftRNG, targetRNG := sampling.NewStreamPair(seed)
	eng := New(draft, target, draftRNG, targetRNG, c, counters, Config{
		MaxDraftTokens:  3,
		MaxTokens:       len(prompt) + n,
		VerifyTimeoutMS: 0,
	})
	tokens, err := eng.Generate(ctx)
	if err != nil {
		return nil, err
	}
	return tokens[len(prompt):], nil
}

func seedFor(i int) uint64 {
	return 1000 + uint64(i)*2_654_435_761
}

// TestDistributionEquivalenceEndToEnd checks that speculative decoding has the
// same per-position token distribution as target-only generation by comparing
// empirical histograms over hundreds of independent seeds.
func TestDistributionEquivalenceEndToEnd(t *testing.T) {
	ctx := context.Background()
	draft, target := testModels(t, 0.35)
	prompt := []model.TokenID{2, 5}
	const n = 6
	const trials = 500

	plainHist := make([][]int, n)
	specHist := make([][]int, n)
	for i := range plainHist {
		plainHist[i] = make([]int, target.VocabSize())
		specHist[i] = make([]int, target.VocabSize())
	}

	for i := 0; i < trials; i++ {
		// Independent seed streams for the two sampling algorithms: the
		// correctness contract is equality of marginal distributions, not a
		// shared RNG tape (speculative and plain steps consume draws
		// differently).
		plain, err := plainGenerate(ctx, target, prompt, n, 100_000+seedFor(i))
		if err != nil {
			t.Fatal(err)
		}
		spec, err := runSpeculative(t, ctx, draft, target, prompt, n, 200_000+seedFor(i), stats.New())
		if err != nil {
			t.Fatal(err)
		}
		for pos := 0; pos < n; pos++ {
			plainHist[pos][plain[pos]]++
			specHist[pos][spec[pos]]++
		}
	}

	// Compare the two empirical histograms (200 vs 200-cell style chi-square
	// with pooled expected counts) per position. Threshold is permissive to
	// avoid flaky failures but still catches a broken acceptance rule.
	for pos := 0; pos < n; pos++ {
		chi := 0.0
		for v := 0; v < target.VocabSize(); v++ {
			a := float64(plainHist[pos][v])
			b := float64(specHist[pos][v])
			total := a + b
			if total == 0 {
				continue
			}
			expected := total / 2
			chi += (a-expected)*(a-expected)/expected + (b-expected)*(b-expected)/expected
		}
		if chi > 25 {
			t.Fatalf("position %d distribution diverges: chi2=%.2f plain=%v spec=%v",
				pos, chi, plainHist[pos], specHist[pos])
		}
	}
}

func TestFixedSeedReproducibility(t *testing.T) {
	ctx := context.Background()
	draft, target := testModels(t, 0.2)
	prompt := []model.TokenID{1, 3}

	first, err := runSpeculative(t, ctx, draft, target, prompt, 20, 777, stats.New())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		again, err := runSpeculative(t, ctx, draft, target, prompt, 20, 777, stats.New())
		if err != nil {
			t.Fatal(err)
		}
		if len(again) != len(first) {
			t.Fatalf("length mismatch: %d vs %d", len(again), len(first))
		}
		for j := range first {
			if again[j] != first[j] {
				t.Fatalf("seeded run not reproducible at token %d: %v vs %v", j, again, first)
			}
		}
	}
}

func TestTokensPerStepGreaterThanOneWhenClose(t *testing.T) {
	ctx := context.Background()
	// Small relative noise => draft nearly equals target => high acceptance.
	draft, target := testModels(t, 0.01)
	counters := stats.New()
	out, err := runSpeculative(t, ctx, draft, target, []model.TokenID{0}, 200, 42, counters)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 200 {
		t.Fatalf("expected 200 tokens, got %d", len(out))
	}
	snap := counters.Snapshot()
	if snap.AvgTokensPerStep <= 1.0 {
		t.Fatalf("expected avg tokens/step > 1 with close distributions, got %.3f (acceptance %.3f)",
			snap.AvgTokensPerStep, snap.AcceptanceRate)
	}
	if snap.AcceptanceRate < 0.9 {
		t.Fatalf("acceptance rate too low for near-identical models: %.3f", snap.AcceptanceRate)
	}
}

func TestRollbackBoundedAndStatsRecorded(t *testing.T) {
	ctx := context.Background()
	// Large noise forces frequent rejects and rollbacks.
	draft, target := testModels(t, 1.5)
	counters := stats.New()
	out, err := runSpeculative(t, ctx, draft, target, []model.TokenID{3}, 60, 9, counters)
	if err != nil {
		t.Fatal(err)
	}
	snap := counters.Snapshot()
	if len(out) != 60 {
		t.Fatalf("expected 60 tokens, got %d", len(out))
	}
	if snap.Rollbacks == 0 {
		t.Fatal("expected rollbacks with diverging models")
	}
	if snap.AvgTokensPerStep < 1.0 {
		t.Fatalf("avg tokens/step must be >= 1, got %.3f", snap.AvgTokensPerStep)
	}
	// Every emitted token must be a valid vocabulary id.
	for _, tok := range out {
		if tok < 0 || int(tok) >= target.VocabSize() {
			t.Fatalf("invalid token emitted: %d", tok)
		}
	}
}
