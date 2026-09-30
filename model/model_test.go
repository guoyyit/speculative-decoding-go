package model

import (
	"context"
	"math"
	"testing"

	"github.com/guoyyit/speculative-decoding-go/sampling"
)

func testConfig() Config {
	return Config{VocabSize: 16, Seed: 7, DraftNoise: 0.3, EOSBias: -2}
}

func approxEqual(a, b []float64, eps float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}

func sumsTo1(t *testing.T, p []float64) {
	t.Helper()
	sum := 0.0
	for _, v := range p {
		if v < 0 {
			t.Fatalf("negative probability %v", v)
		}
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("probabilities sum to %.6f", sum)
	}
}

func TestTargetDistDeterministic(t *testing.T) {
	tgt := NewTarget(testConfig())
	p1 := tgt.Dist([]int{1, 2, 3})
	p2 := tgt.Dist([]int{1, 2, 3})
	if !approxEqual(p1, p2, 0) {
		t.Fatal("same history must yield identical distributions")
	}
	sumsTo1(t, p1)

	p3 := tgt.Dist([]int{1, 2, 4})
	if approxEqual(p1, p3, 1e-12) {
		t.Fatal("different histories should produce different distributions")
	}
}

func TestDistOnlyDependsOnContextWindow(t *testing.T) {
	tgt := NewTarget(testConfig())
	a := tgt.Dist([]int{1, 7, 7, 7, 7, 9, 9, 5, 6})
	b := tgt.Dist([]int{2, 7, 7, 7, 7, 9, 9, 5, 6})
	c := tgt.Dist([]int{2, 7, 7, 7, 7, 9, 9, 5, 7})
	if !approxEqual(a, b, 1e-12) {
		t.Fatal("tokens outside context window must not matter")
	}
	if approxEqual(a, c, 1e-9) {
		t.Fatal("tokens inside context window must matter")
	}
}

func TestDraftEqualsTargetWhenNoiseZero(t *testing.T) {
	cfg := testConfig()
	cfg.DraftNoise = 0
	tgt := NewTarget(cfg)
	dr := NewDraft(cfg)
	history := []int{3, 4, 5}
	if !approxEqual(tgt.Dist(history), dr.dist(history), 1e-12) {
		t.Fatal("draft must equal target when DraftNoise == 0")
	}
}

func TestVerifyRowsMatchDist(t *testing.T) {
	tgt := NewTarget(testConfig())
	history := []int{1, 2}
	candidates := []int{3, 4, 5}
	rows, err := tgt.Verify(context.Background(), history, candidates)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if len(rows) != len(candidates)+1 {
		t.Fatalf("Verify rows = %d, want %d", len(rows), len(candidates)+1)
	}
	cur := append([]int(nil), history...)
	for i := 0; i <= len(candidates); i++ {
		if !approxEqual(rows[i], tgt.Dist(cur), 1e-12) {
			t.Fatalf("row %d does not match pointwise Dist", i)
		}
		sumsTo1(t, rows[i])
		if i < len(candidates) {
			cur = append(cur, candidates[i])
		}
	}
}

func TestProposalDeterministicWithSeed(t *testing.T) {
	cfg := testConfig()
	d1 := NewDraft(cfg)
	d2 := NewDraft(cfg)
	r1 := sampling.New(123)
	r2 := sampling.New(123)
	p1, err := d1.Propose(context.Background(), []int{1}, 4, r1)
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	p2, err := d2.Propose(context.Background(), []int{1}, 4, r2)
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	if len(p1.Tokens) != 4 || len(p1.Q) != 4 {
		t.Fatalf("proposal shape = %d/%d", len(p1.Tokens), len(p1.Q))
	}
	for i := range p1.Tokens {
		if p1.Tokens[i] != p2.Tokens[i] {
			t.Fatal("same seed must produce same proposal")
		}
	}
}

func TestVerifyCanceled(t *testing.T) {
	cfg := testConfig()
	cfg.VerifyLatencyMS = 200
	tgt := NewTarget(cfg)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tgt.Verify(ctx, nil, []int{1, 2}); err == nil {
		t.Fatal("canceled context must abort Verify")
	}
}
