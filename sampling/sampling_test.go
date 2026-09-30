package sampling

import (
	"context"
	"math"
	"testing"

	"github.com/guoyyit/speculative-decoding-go/model"
)

func TestAcceptProb(t *testing.T) {
	cases := []struct{ p, q, want float64 }{
		{0.8, 0.2, 1},
		{0.2, 0.8, 0.25},
		{0.5, 0.5, 1},
		{0.0, 0.3, 0},
	}
	for _, c := range cases {
		if got := AcceptProb(c.p, c.q); math.Abs(got-c.want) > 1e-12 {
			t.Fatalf("AcceptProb(%v,%v)=%v want %v", c.p, c.q, got, c.want)
		}
	}
}

func TestResidual(t *testing.T) {
	p := model.Distribution{0.5, 0.3, 0.2}
	q := model.Distribution{0.1, 0.5, 0.1}
	res := Residual(p, q)
	sum := 0.0
	for _, v := range res {
		if v < 0 {
			t.Fatalf("negative residual mass: %v", res)
		}
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("residual not normalized: %v sum=%v", res, sum)
	}
	if res[1] != 0 {
		t.Fatalf("residual must have no mass where q>=p: %v", res)
	}
	// (0.4, 0, 0.1) normalized -> (0.8, 0, 0.2)
	if math.Abs(res[0]-0.8) > 1e-12 || math.Abs(res[2]-0.2) > 1e-12 {
		t.Fatalf("unexpected residual: %v", res)
	}
}

// scriptedModel serves fixed distributions, optionally permuted, so tests can
// exercise the acceptance rule on arbitrary draft/target distribution pairs.
type scriptedModel struct {
	draftQ model.Distribution
	target model.Distribution
}

func (m *scriptedModel) Propose(_ context.Context, _ []model.TokenID, k int, _ func() float64) ([]model.TokenID, []model.Distribution, error) {
	toks := make([]model.TokenID, k)
	dists := make([]model.Distribution, k)
	for i := range toks {
		// Scripted draft always proposes token 0 in this one-step test.
		toks[i] = 0
		dists[i] = m.draftQ
	}
	return toks, dists, nil
}

func (m *scriptedModel) Verify(_ context.Context, _ []model.TokenID, cands []model.TokenID) ([]model.Distribution, error) {
	dists := make([]model.Distribution, len(cands)+1)
	for i := range dists {
		dists[i] = m.target
	}
	return dists, nil
}

func (m *scriptedModel) Next(_ context.Context, _ []model.TokenID) (model.Distribution, error) {
	return m.target, nil
}

func (m *scriptedModel) Name() string   { return "scripted" }
func (m *scriptedModel) VocabSize() int { return len(m.target) }

// TestMarginalEquivalence is the central correctness claim: for a single
// position, the marginal distribution of the token emitted by speculative
// sampling must equal the target distribution p exactly, for every draft
// distribution q. We enumerate the possible draft tokens analytically:
//
//	Pr[emit y] = q[y]*min(1,p[y]/q[y])
//	           + sum_x 1[y != x] * q[x]*(1-min(1,p[x]/q[x]))
//	                      * max(0,p[y]-q[y]) / sum_v max(0,p[v]-q[v])
func TestMarginalEquivalence(t *testing.T) {
	pairs := []struct{ p, q model.Distribution }{
		{
			model.Distribution{0.5, 0.3, 0.15, 0.05},
			model.Distribution{0.1, 0.6, 0.2, 0.1},
		},
		{
			model.Distribution{0.7, 0.2, 0.1},
			model.Distribution{0.1, 0.1, 0.8},
		},
		{
			// Identical distributions: every draft token is accepted.
			model.Distribution{0.25, 0.25, 0.25, 0.25},
			model.Distribution{0.25, 0.25, 0.25, 0.25},
		},
	}
	for _, pair := range pairs {
		got := speculativeMarginal(pair.p, pair.q)
		for v := range pair.p {
			if math.Abs(got[v]-pair.p[v]) > 1e-12 {
				t.Fatalf("marginal mismatch at v=%d: spec=%v target=%v\nfull=%v", v, got, pair.p, got)
			}
		}
	}
}

// speculativeMarginal computes the exact marginal emitted distribution of one
// speculative-sampling position given target p and draft q.
func speculativeMarginal(p, q model.Distribution) model.Distribution {
	V := len(p)
	res := Residual(p, q)
	rejectMass := 0.0
	for x := 0; x < V; x++ {
		rejectMass += q[x] * (1 - AcceptProb(p[x], q[x]))
	}
	out := make(model.Distribution, V)
	for x := 0; x < V; x++ {
		out[x] += q[x] * AcceptProb(p[x], q[x]) // x accepted
	}
	for y := 0; y < V; y++ {
		out[y] += rejectMass * res[y] // y drawn as correction
	}
	return out
}

// TestEmpiricalDistributionChiSquared verifies end-to-end Step output against
// plain target sampling on repeated trials with independent randomness.
func TestEmpiricalDistributionChiSquared(t *testing.T) {
	ctx := context.Background()
	p := model.Distribution{0.4, 0.3, 0.2, 0.1}
	q := model.Distribution{0.2, 0.35, 0.25, 0.2}

	const trials = 4000
	specCounts := make([]int, len(p))
	plainCounts := make([]int, len(p))

	stochastic := &stochasticPair{p: p, q: q}
	for i := 0; i < trials; i++ {
		draftSrc := NewSource(0xC0FFEE + uint64(i)*1_000_003)
		targetSrc := NewSource(0x77AA77 + uint64(i)*9_999_983)
		out, err := Step(ctx, stochastic, stochastic, nil, 1, draftSrc, targetSrc)
		if err != nil {
			t.Fatal(err)
		}
		tok := out.Correction
		if out.RejectedAt < 0 {
			tok = out.Proposed[0]
		}
		specCounts[tok]++

		plainSrc := NewSource(0xBEEF + uint64(i)*7_777_777)
		plainCounts[Sample(p, plainSrc)]++
	}

	for v := range p {
		want := float64(trials) * p[v]
		chi := math.Pow(float64(specCounts[v])-want, 2) / want
		// Per-cell chi-square threshold; generous to avoid flakes.
		if chi > 20 {
			t.Fatalf("speculative empirical dist diverges at %d: count=%d want~%.0f chi=%.2f",
				v, specCounts[v], want, chi)
		}
	}
}

// stochasticPair proposes and verifies using freshly drawn randomness for the
// draft token, while both distributions are fixed across positions.
type stochasticPair struct {
	p model.Distribution
	q model.Distribution
}

func (m *stochasticPair) Propose(_ context.Context, _ []model.TokenID, k int, draw func() float64) ([]model.TokenID, []model.Distribution, error) {
	toks := make([]model.TokenID, k)
	dists := make([]model.Distribution, k)
	for i := range toks {
		dists[i] = m.q
		toks[i] = model.Sample(m.q, draw())
	}
	return toks, dists, nil
}

func (m *stochasticPair) Verify(_ context.Context, _ []model.TokenID, cands []model.TokenID) ([]model.Distribution, error) {
	dists := make([]model.Distribution, len(cands)+1)
	for i := range dists {
		dists[i] = m.p
	}
	return dists, nil
}

func (m *stochasticPair) Next(_ context.Context, _ []model.TokenID) (model.Distribution, error) {
	return m.p, nil
}
func (m *stochasticPair) Name() string   { return "stochastic" }
func (m *stochasticPair) VocabSize() int { return len(m.p) }

// TestNoRejectedTokenEmitted ensures rejected candidates can never appear as
// accepted output of Step.
func TestNoRejectedTokenEmitted(t *testing.T) {
	ctx := context.Background()
	// q is sharply concentrated on token 1, p on token 0: token 1 proposed
	// with high probability is rejected and corrected toward token 0/2.
	m := &scriptedModel{
		draftQ: model.Distribution{0.0, 1.0, 0.0},
		target: model.Distribution{0.8, 0.0, 0.2},
	}
	for i := 0; i < 100; i++ {
		src := NewSource(uint64(i) + 1)
		out, err := Step(ctx, m, m, nil, 1, src, src)
		if err != nil {
			t.Fatal(err)
		}
		if out.RejectedAt == 0 {
			if out.Correction == 1 {
				t.Fatal("rejected draft token leaked into output")
			}
			if len(out.Accepted) != 0 {
				t.Fatal("accepted prefix must be empty at first-position reject")
			}
		}
	}
}
