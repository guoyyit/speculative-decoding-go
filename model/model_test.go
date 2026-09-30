package model

import (
	"context"
	"math"
	"testing"
)

func TestDistributionsDeterministic(t *testing.T) {
	cfg := Config{VocabSize: 12, Temperature: 1.0}
	target := NewTarget(cfg)
	ctx := context.Background()
	prefix := []TokenID{1, 4, 7}

	d1, err := target.Next(ctx, prefix)
	if err != nil {
		t.Fatal(err)
	}
	d2, err := target.Next(ctx, prefix)
	if err != nil {
		t.Fatal(err)
	}
	for v := range d1 {
		if d1[v] != d2[v] {
			t.Fatalf("target distribution nondeterministic at %d", v)
		}
	}
	sum := 0.0
	for _, p := range d1 {
		if p < 0 || math.IsNaN(p) {
			t.Fatal("invalid probability")
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("distribution not normalized, sum=%v", sum)
	}
}

func TestDraftEqualsTargetAtZeroNoise(t *testing.T) {
	ctx := context.Background()
	cfg := Config{VocabSize: 12, Temperature: 1.3, Noise: 0}
	draft := NewDraft(cfg)
	target := NewTarget(cfg)

	for _, prefix := range [][]TokenID{{}, {2}, {3, 8, 1, 9}} {
		d, err := target.Next(ctx, prefix)
		if err != nil {
			t.Fatal(err)
		}
		_, qs, err := draft.Propose(ctx, prefix, 1, func() float64 { return 0.5 })
		if err != nil {
			t.Fatal(err)
		}
		for v := range d {
			if math.Abs(d[v]-qs[0][v]) > 1e-15 {
				t.Fatalf("draft != target at zero noise (prefix %v, v=%d): %v vs %v",
					prefix, v, d[v], qs[0][v])
			}
		}
	}
}

func TestDraftDivergesWithNoise(t *testing.T) {
	ctx := context.Background()
	base := Config{VocabSize: 12, Temperature: 1.0}
	target := NewTarget(base)
	draft := NewDraft(Config{VocabSize: 12, Temperature: 1.0, Noise: 0.35})

	p, err := target.Next(ctx, []TokenID{5})
	if err != nil {
		t.Fatal(err)
	}
	_, qs, err := draft.Propose(ctx, []TokenID{5}, 1, func() float64 { return 0.5 })
	if err != nil {
		t.Fatal(err)
	}
	diff := 0.0
	for v := range p {
		diff += math.Abs(p[v] - qs[0][v])
	}
	if diff < 1e-6 {
		t.Fatalf("expected draft/target divergence with noise, L1=%v", diff)
	}
}

func TestVerifyReturnsKPlusOneDistributions(t *testing.T) {
	ctx := context.Background()
	target := NewTarget(Config{VocabSize: 6})
	cands := []TokenID{1, 2, 3}
	dists, err := target.Verify(ctx, []TokenID{0}, cands)
	if err != nil {
		t.Fatal(err)
	}
	if len(dists) != len(cands)+1 {
		t.Fatalf("expected %d dists, got %d", len(cands)+1, len(dists))
	}

	// dists[i] must equal Next(prefix+cands[:i]).
	seq := []TokenID{0}
	for i := 0; i <= len(cands); i++ {
		want, err := target.Next(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		for v := range want {
			if math.Abs(want[v]-dists[i][v]) > 1e-15 {
				t.Fatalf("verify dist %d mismatches Next", i)
			}
		}
		if i < len(cands) {
			seq = append(seq, cands[i])
		}
	}
}

func TestProposeReproducibleAcrossCalls(t *testing.T) {
	ctx := context.Background()
	draft := NewDraft(Config{VocabSize: 8, Noise: 0.1})
	draw := func() float64 { return 0.3 }

	t1, _, err := draft.Propose(ctx, []TokenID{1}, 3, draw)
	if err != nil {
		t.Fatal(err)
	}
	t2, _, err := draft.Propose(ctx, []TokenID{1}, 3, draw)
	if err != nil {
		t.Fatal(err)
	}
	for i := range t1 {
		if t1[i] != t2[i] {
			t.Fatal("proposal not reproducible for identical stream and prefix")
		}
	}
}
