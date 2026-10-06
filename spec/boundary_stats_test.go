package spec

import (
	"context"
	"errors"
	"testing"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
)

// TestEmittedEqualsDeliveredAllAccepted covers the boundary where the
// remaining generation budget is smaller than (and equal to) the number of
// tokens one step can hand back. With identical draft/target distributions a
// single step emits DraftK+1 tokens; EmittedTokens must still equal the exact
// token count returned to the caller at every budget.
func TestEmittedEqualsDeliveredAllAccepted(t *testing.T) {
	mcfg := specCfg()
	mcfg.DraftNoise = 0
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)

	// Reference run with a large budget; the per-budget outputs must be its
	// exact prefixes (token sequence never changes with the budget).
	ref := New(tgt, dr, 42, 64, decCfg())
	refTokens, _, err := ref.Generate(context.Background(), []int{1, 2}, 12)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}

	for _, budget := range []int{1, 2, 3, 4, 5} {
		d := New(tgt, dr, 42, 64, decCfg())
		tokens, results, err := d.Generate(context.Background(), []int{1, 2}, budget)
		if err != nil {
			t.Fatalf("budget=%d Generate: %v", budget, err)
		}
		if len(tokens) != budget {
			t.Fatalf("budget=%d: got %d tokens", budget, len(tokens))
		}
		if !equalInts(tokens, refTokens[:budget]) {
			t.Fatalf("budget=%d output %v is not the reference prefix %v", budget, tokens, refTokens[:budget])
		}
		st := d.Stats()
		if st.EmittedTokens != int64(budget) {
			t.Fatalf("budget=%d: EmittedTokens=%d, want %d (must equal delivered)", budget, st.EmittedTokens, budget)
		}
		var sumEmitted int
		for i, r := range results {
			if r.Emitted < 0 {
				t.Fatalf("negative emitted on step %d", i)
			}
			sumEmitted += r.Emitted
		}
		if sumEmitted != budget {
			t.Fatalf("budget=%d: sum(step.Emitted)=%d, want %d", budget, sumEmitted, budget)
		}
		if st.Steps != int64(len(results)) {
			t.Fatalf("budget=%d: Steps=%d, want %d", budget, st.Steps, len(results))
		}
		if st.SuccessSteps != int64(len(results)) {
			t.Fatalf("budget=%d: SuccessSteps=%d, want %d", budget, st.SuccessSteps, len(results))
		}
		if st.FallbackSteps != 0 || st.FailedSteps != 0 {
			t.Fatalf("budget=%d: unexpected fallback/failed steps", budget)
		}
		if st.EmittedTokens > int64(len(tokens)) {
			t.Fatalf("budget=%d: statistic exceeds reality: emitted=%d tokens=%d", budget, st.EmittedTokens, len(tokens))
		}
	}
}

// TestRejectCorrectionExactlyExhaustsBudget: two candidates accepted, the
// third rejected and replaced by a correction token, with exactly three
// tokens of budget left. The correction must be delivered (not dropped), the
// cache must hold precisely history + accepted prefix + correction, and the
// emitted counters must equal exactly three.
func TestRejectCorrectionExactlyExhaustsBudget(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	d := New(tgt, dr, 11, 64, decCfg())

	history := []int{1, 2}
	prop, err := dr.Propose(context.Background(), history, 4, sampling.New(11))
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	if len(prop.Tokens) != 4 {
		t.Fatalf("expected 4 proposals, got %d", len(prop.Tokens))
	}
	fixedToken := 9
	for _, pt := range prop.Tokens[:3] {
		if pt == fixedToken {
			t.Fatalf("fixed token collides with a draft token; pick another")
		}
	}
	rows := make([][]float64, 0, 5)
	// True target rows at positions 0 and 1: with identical-ish accept draws
	// the first two candidates go through acceptance; position 2 is a point
	// mass on fixedToken, forcing rejection there with probability 1.
	rows = append(rows, tgt.Dist(history))
	rows = append(rows, tgt.Dist(append(append([]int(nil), history...), prop.Tokens[0])))
	rowReject := make([]float64, tgt.Size())
	rowReject[fixedToken] = 1
	rows = append(rows, rowReject)
	rows = append(rows, rowReject) // unused after the rejection
	rows = append(rows, rowReject) // bonus row, unused
	d.WithVerifier(&forcedVerifier{rows: rows})

	const budget = 3
	tokens, results, err := d.Generate(context.Background(), history, budget)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	want := []int{prop.Tokens[0], prop.Tokens[1], fixedToken}
	if !equalInts(tokens, want) {
		t.Fatalf("tokens=%v, want %v", tokens, want)
	}
	r := results[0]
	if r.Outcome != OutcomeSuccess || !r.RolledBack || r.Accepted != 2 || r.Emitted != 3 || r.Proposed != 4 {
		t.Fatalf("step result = %+v, want accepted=2 emitted=3 proposed=4 rolledback", r)
	}
	// Cache must contain exactly the delivered prefix; nothing past it.
	wantCache := append(append([]int(nil), history...), want...)
	if !equalInts(d.cache.Tokens(), wantCache) {
		t.Fatalf("cache=%v, want %v", d.cache.Tokens(), wantCache)
	}
	st := d.Stats()
	if st.EmittedTokens != 3 || st.AcceptedTokens != 2 || st.ProposedTokens != 4 {
		t.Fatalf("stats emitted=%d accepted=%d proposed=%d, want 3/2/4", st.EmittedTokens, st.AcceptedTokens, st.ProposedTokens)
	}
	if st.Rollbacks != 1 || st.SuccessSteps != 1 || st.FallbackSteps != 0 {
		t.Fatalf("unexpected step counters: %+v", st)
	}
}

// pointMassBonusEOSVerifier accepts every proposed candidate with certainty
// (each verification row is a point mass on that candidate) and makes the
// bonus position deterministically EOS, so a fully-accepted step ends on EOS.
type pointMassBonusEOSVerifier struct {
	size int
}

func (v *pointMassBonusEOSVerifier) Verify(_ context.Context, _ []int, candidates []int) ([][]float64, error) {
	rows := make([][]float64, 0, len(candidates)+1)
	for _, c := range candidates {
		row := make([]float64, v.size)
		row[c] = 1
		rows = append(rows, row)
	}
	eosRow := make([]float64, v.size)
	eosRow[0] = 1
	rows = append(rows, eosRow)
	return rows, nil
}

// TestEOSClampedEmittedEqualsDelivered: when an EOS bonus would overshoot the
// remaining budget, the caller receives only the in-budget prefix and EOS is
// not part of it; EmittedTokens must still equal exactly that prefix length.
func TestEOSClampedEmittedEqualsDelivered(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)

	for _, budget := range []int{2, 4} {
		d := New(tgt, dr, 77, 64, decCfg())
		d.WithVerifier(&pointMassBonusEOSVerifier{size: tgt.Size()})
		tokens, results, err := d.Generate(context.Background(), []int{3, 4}, budget)
		if err != nil {
			t.Fatalf("budget=%d Generate: %v", budget, err)
		}
		if len(tokens) != budget {
			t.Fatalf("budget=%d: got %d tokens", budget, len(tokens))
		}
		if containsEOS(tokens) {
			t.Fatalf("budget=%d: EOS must be clipped out of the delivered prefix: %v", budget, tokens)
		}
		st := d.Stats()
		if st.EmittedTokens != int64(budget) {
			t.Fatalf("budget=%d: EmittedTokens=%d want %d", budget, st.EmittedTokens, budget)
		}
		var sumEmitted int
		for _, r := range results {
			sumEmitted += r.Emitted
		}
		if sumEmitted != budget {
			t.Fatalf("budget=%d: summed step Emitted=%d want %d", budget, sumEmitted, budget)
		}
		if st.SuccessSteps != int64(len(results)) {
			t.Fatalf("budget=%d: SuccessSteps mismatch", budget)
		}
	}
}

var errInjectedVerify = errors.New("injected verification failure")

// errorOnceVerifier fails the first verification with a non-timeout error,
// then delegates to the wrapped verifier.
type errorOnceVerifier struct {
	inner Verifier
	calls int
}

func (v *errorOnceVerifier) Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error) {
	v.calls++
	if v.calls == 1 {
		return nil, errInjectedVerify
	}
	return v.inner.Verify(ctx, history, candidates)
}

// alwaysErrorVerifier fails every verification with a non-timeout error.
type alwaysErrorVerifier struct{}

func (alwaysErrorVerifier) Verify(context.Context, []int, []int) ([][]float64, error) {
	return nil, errInjectedVerify
}

// TestNonTimeoutVerifyFailureFallback: a verification failure for a reason
// other than timeout must be recorded as verify_error, counted separately
// from timeouts, roll the proposal back and continue the request with one
// vanilla target token.
func TestNonTimeoutVerifyFailureFallback(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)

	// Always-failing verification: every step falls back to one vanilla
	// target token; the output must equal plain non-speculative generation.
	d := New(tgt, dr, 321, 64, decCfg())
	d.WithVerifier(alwaysErrorVerifier{})
	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 6)
	if err != nil {
		t.Fatalf("Generate must not fail on verify error: %v", err)
	}
	if len(tokens) != 6 {
		t.Fatalf("got %d tokens, want 6", len(tokens))
	}

	// Independently reproduce the exact sampler stream: each failed step the
	// draft proposal still consumes one uniform draw per proposed token
	// (categorical uses Float64), after which the vanilla fallback consumes
	// one draw from the target distribution.
	sm := sampling.New(321)
	cur := append([]int(nil), []int{1, 2}...)
	want := make([]int, 0, 6)
	for i := 0; i < 6; i++ {
		// Replay the draft proposal: it consumes one uniform draw per
		// proposed token; the drawn values are discarded with the proposal.
		if _, err := dr.Propose(context.Background(), cur, decCfg().DraftK, sm); err != nil {
			t.Fatalf("replay Propose: %v", err)
		}
		tok := sm.Sample(tgt.Dist(cur))
		want = append(want, tok)
		cur = append(cur, tok)
	}
	if !equalInts(tokens, want) {
		t.Fatalf("fallback output %v != pure target stream %v", tokens, want)
	}

	for i, r := range results {
		if r.Outcome != OutcomeFallback || r.FallbackCause != CauseVerifyError {
			t.Fatalf("step %d = %+v, want fallback/verify_error", i, r)
		}
		if !r.RolledBack || r.Emitted != 1 {
			t.Fatalf("step %d must roll back and emit exactly one token: %+v", i, r)
		}
	}
	st := d.Stats()
	if st.ErrorFallbacks != 6 || st.TimeoutFallbacks != 0 {
		t.Fatalf("ErrorFallbacks=%d TimeoutFallbacks=%d, want 6/0", st.ErrorFallbacks, st.TimeoutFallbacks)
	}
	if st.FallbackSteps != 6 || st.Rollbacks != 6 || st.SuccessSteps != 0 || st.FailedSteps != 0 {
		t.Fatalf("unexpected step counters: %+v", st)
	}
	if st.EmittedTokens != 6 {
		t.Fatalf("EmittedTokens=%d, want 6", st.EmittedTokens)
	}
	// No staged candidate may survive: cache tail equals the generated stream.
	wantCache := append(append([]int(nil), 1, 2), want...)
	if !equalInts(d.cache.Tokens(), wantCache) {
		t.Fatalf("cache=%v, want %v", d.cache.Tokens(), wantCache)
	}
}

// TestErrorFallbackThenRecoveries: one non-timeout failure followed by normal
// speculative steps; only the first step is a verify_error fallback and the
// request still completes with exact emitted accounting.
func TestErrorFallbackThenRecoveries(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)

	d := New(tgt, dr, 55, 64, decCfg())
	d.WithVerifier(&errorOnceVerifier{inner: tgt})
	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 10)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 10 {
		t.Fatalf("got %d tokens, want 10", len(tokens))
	}
	if results[0].Outcome != OutcomeFallback || results[0].FallbackCause != CauseVerifyError {
		t.Fatalf("first step = %+v, want verify_error fallback", results[0])
	}
	for i := 1; i < len(results); i++ {
		if results[i].Outcome != OutcomeSuccess {
			t.Fatalf("step %d = %+v, want recovery to success", i, results[i])
		}
		if results[i].FallbackCause != "" {
			t.Fatalf("step %d unexpectedly carries cause %q", i, results[i].FallbackCause)
		}
	}
	st := d.Stats()
	if st.ErrorFallbacks != 1 || st.TimeoutFallbacks != 0 {
		t.Fatalf("error/timeouts = %d/%d, want 1/0", st.ErrorFallbacks, st.TimeoutFallbacks)
	}
	if st.EmittedTokens != 10 {
		t.Fatalf("EmittedTokens=%d, want 10", st.EmittedTokens)
	}
	var sum int
	for _, r := range results {
		sum += r.Emitted
	}
	if sum != 10 {
		t.Fatalf("summed Emitted=%d, want 10", sum)
	}
	if st.SuccessSteps+st.FallbackSteps+st.FailedSteps != st.Steps {
		t.Fatalf("step classes do not partition total steps: %+v", st)
	}
}

// TestBudgetExactlyEqualsKAndKPlus1: when the remaining budget equals DraftK
// (no room for the bonus) or DraftK+1 (room for exactly one bonus), the
// delivered count and every counter must be exact under all-accept steps.
func TestBudgetExactlyEqualsKAndKPlus1(t *testing.T) {
	mcfg := specCfg()
	mcfg.DraftNoise = 0
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)

	k := decCfg().DraftK
	for _, budget := range []int{k, k + 1} {
		d := New(tgt, dr, 88, 64, decCfg())
		tokens, results, err := d.Generate(context.Background(), []int{5, 6}, budget)
		if err != nil {
			t.Fatalf("budget=%d Generate: %v", budget, err)
		}
		if len(tokens) != budget {
			t.Fatalf("budget=%d: got %d tokens", budget, len(tokens))
		}
		if len(results) != 1 {
			t.Fatalf("budget=%d: expected exactly one step, got %d", budget, len(results))
		}
		r := results[0]
		// budget == DraftK: bonus is sampled but must not be delivered or
		// counted as emitted. budget == DraftK+1: bonus is delivered.
		if r.Proposed != k || r.Accepted != k {
			t.Fatalf("budget=%d: proposed=%d accepted=%d want %d/%d", budget, r.Proposed, r.Accepted, k, k)
		}
		if r.Emitted != budget {
			t.Fatalf("budget=%d: step Emitted=%d want %d", budget, r.Emitted, budget)
		}
		st := d.Stats()
		if st.EmittedTokens != int64(budget) {
			t.Fatalf("budget=%d: EmittedTokens=%d want %d", budget, st.EmittedTokens, budget)
		}
		if st.ProposedTokens != int64(k) || st.AcceptedTokens != int64(k) {
			t.Fatalf("budget=%d: proposed=%d accepted=%d want %d/%d", budget, st.ProposedTokens, st.AcceptedTokens, k, k)
		}
		if st.AcceptRate() > 1+1e-12 {
			t.Fatalf("budget=%d: accept rate %f > 1", budget, st.AcceptRate())
		}
	}
}

// TestEarlyEOSProposalNotOvercounted: a proposal that ends early on EOS
// (fewer than DraftK candidates) delivers exactly through EOS and records no
// more emitted tokens than the caller receives. Uses strong EOS bias and
// scans seeds until the early-EOS case is observed.
func TestEarlyEOSProposalNotOvercounted(t *testing.T) {
	mcfg := specCfg()
	mcfg.VocabSize = 16
	mcfg.EOSBias = 3.0
	mcfg.DraftNoise = 0
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)

	found := false
	for seed := uint64(1); seed < 500 && !found; seed++ {
		d := New(tgt, dr, seed, 64, decCfg())
		tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 10)
		if err != nil {
			t.Fatalf("seed=%d Generate: %v", seed, err)
		}
		st := d.Stats()
		if st.EmittedTokens != int64(len(tokens)) {
			t.Fatalf("seed=%d EmittedTokens=%d != len(tokens)=%d", seed, st.EmittedTokens, len(tokens))
		}
		var sum int
		for _, r := range results {
			sum += r.Emitted
		}
		if sum != len(tokens) {
			t.Fatalf("seed=%d summed Emitted=%d != %d", seed, sum, len(tokens))
		}
		for _, r := range results {
			if r.Outcome == OutcomeSuccess && r.Proposed < decCfg().DraftK {
				// An early-ending proposal: the last delivered token is EOS,
				// delivery stops, and nothing is overcounted.
				if len(tokens) == 0 || tokens[len(tokens)-1] != 0 {
					t.Fatalf("seed=%d early proposal but no trailing EOS: %v", seed, tokens)
				}
				found = true
			}
		}
	}
	if !found {
		t.Skip("no early-EOS proposal observed in seed scan; adjust bias")
	}
}
