package spec

import (
	"context"
	"errors"
	"testing"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
)

// identicalDraftCfg returns a config where draft == target (every candidate
// is accepted), so each step commits exactly DraftK+1 tokens unless the
// remaining budget trims it.
func identicalDraftCfg() model.Config {
	return model.Config{VocabSize: 32, Seed: 5, DraftNoise: 0, EOSBias: -2.5}
}

func sumEmitted(results []StepResult) int {
	total := 0
	for _, r := range results {
		total += r.Emitted
	}
	return total
}

// Remaining budget smaller than one full step: the recorded emitted-token
// total must equal exactly what the caller received, not what the step
// could have committed.
func TestEmittedTokensEqualDeliveredWhenBudgetTrimsStep(t *testing.T) {
	mcfg := identicalDraftCfg()
	d := New(model.NewTarget(mcfg), model.NewDraft(mcfg), 42, 64, Config{DraftK: 4, VerifyTimeoutMS: 500})

	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 7)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Step 1 commits 5 tokens; step 2 could commit 5 but the budget is 2.
	if len(tokens) != 7 {
		t.Fatalf("delivered %d tokens, want 7", len(tokens))
	}
	if len(results) != 2 {
		t.Fatalf("steps = %d, want 2", len(results))
	}
	if results[0].Emitted != 5 || results[1].Emitted != 2 {
		t.Fatalf("step emissions = %d, %d; want 5, 2", results[0].Emitted, results[1].Emitted)
	}
	st := d.Stats()
	if st.EmittedTokens != 7 {
		t.Fatalf("EmittedTokens = %d, want exactly the 7 delivered tokens", st.EmittedTokens)
	}
	if st.EmittedTokens != int64(sumEmitted(results)) {
		t.Fatalf("EmittedTokens = %d, step results sum to %d", st.EmittedTokens, sumEmitted(results))
	}
	if st.SuccessSteps != 2 || st.Steps != 2 || st.FallbackSteps != 0 || st.FailedSteps != 0 {
		t.Fatalf("step counters = %+v, want 2 success steps only", st)
	}
	if got := st.TokensPerStep(); got != 3.5 {
		t.Fatalf("TokensPerStep = %v, want exactly 3.5 (7 delivered / 2 steps)", got)
	}
	if st.AcceptRate() != 1 {
		t.Fatalf("AcceptRate = %v, want 1 for identical draft", st.AcceptRate())
	}
	// Cache must hold exactly prompt + delivered tokens: nothing beyond what
	// the caller received may linger.
	want := append([]int{1, 2}, tokens...)
	if !equalInts(d.cache.Tokens(), want) {
		t.Fatalf("cache = %v, want prompt+delivered %v", d.cache.Tokens(), want)
	}
}

// Remaining budget exactly equal to one full step's output.
func TestEmittedTokensWhenBudgetEqualsStepOutput(t *testing.T) {
	mcfg := identicalDraftCfg()
	d := New(model.NewTarget(mcfg), model.NewDraft(mcfg), 42, 64, Config{DraftK: 4, VerifyTimeoutMS: 500})

	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 5)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 5 || len(results) != 1 || results[0].Emitted != 5 {
		t.Fatalf("tokens=%v results=%+v, want one 5-token step", tokens, results)
	}
	st := d.Stats()
	if st.EmittedTokens != 5 || st.Steps != 1 {
		t.Fatalf("EmittedTokens=%d Steps=%d, want 5 and 1", st.EmittedTokens, st.Steps)
	}
	if got := st.TokensPerStep(); got != 5 {
		t.Fatalf("TokensPerStep = %v, want exactly 5", got)
	}
}

// forcedRejectVerifier accepts the first acceptN candidates (its rows equal
// the draft distributions, so acceptance probability is exactly 1) and
// rejects at position acceptN with a one-hot row on fixedToken.
type forcedRejectVerifier struct {
	rows [][]float64
}

func (f *forcedRejectVerifier) Verify(_ context.Context, _ []int, candidates []int) ([][]float64, error) {
	if len(f.rows) != len(candidates)+1 {
		return nil, errors.New("forcedRejectVerifier: row mismatch")
	}
	return f.rows, nil
}

// buildRejectRows crafts verification rows that accept the first acceptN
// proposed tokens and deterministically reject the next one, replacing it
// with fixedToken via the residual resample.
func buildRejectRows(tgt *model.Target, prop *model.Proposal, acceptN int, fixedToken int) [][]float64 {
	rows := make([][]float64, 0, len(prop.Tokens)+1)
	for i := 0; i < acceptN; i++ {
		rows = append(rows, prop.Q[i]) // p == q: acceptance prob is exactly 1
	}
	oneHot := make([]float64, tgt.Size())
	oneHot[fixedToken] = 1
	rows = append(rows, oneHot)
	for len(rows) < len(prop.Tokens)+1 {
		rows = append(rows, oneHot) // unused past the first rejection
	}
	return rows
}

// Rejection whose correction token exactly exhausts the remaining budget:
// accepted prefix + correction == budget, nothing is trimmed.
func TestRejectedCorrectionExactlyExhaustsBudget(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	const seed = 11
	d := New(tgt, dr, seed, 64, decCfg())

	history := []int{1, 2}
	prop, err := dr.Propose(context.Background(), history, 4, sampling.New(seed))
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	if len(prop.Tokens) != 4 {
		t.Fatalf("proposal = %v, want 4 tokens for this deterministic setup", prop.Tokens)
	}
	rejectToken := prop.Tokens[2]
	fixedToken := (rejectToken + 1) % tgt.Size()
	if fixedToken == 0 {
		fixedToken++
	}
	d.WithVerifier(&forcedRejectVerifier{rows: buildRejectRows(tgt, prop, 2, fixedToken)})

	// Budget 3 == 2 accepted + 1 correction: exactly exhausted.
	tokens, results, err := d.Generate(context.Background(), history, 3)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	want := []int{prop.Tokens[0], prop.Tokens[1], fixedToken}
	if !equalInts(tokens, want) {
		t.Fatalf("tokens = %v, want %v", tokens, want)
	}
	if len(results) != 1 {
		t.Fatalf("steps = %d, want 1", len(results))
	}
	r := results[0]
	if r.Emitted != 3 || r.Accepted != 2 || !r.RolledBack || r.Outcome != OutcomeSuccess {
		t.Fatalf("step result = %+v, want emitted=3 accepted=2 rolled-back success", r)
	}
	st := d.Stats()
	if st.EmittedTokens != 3 || st.AcceptedTokens != 2 || st.ProposedTokens != 4 || st.Rollbacks != 1 {
		t.Fatalf("stats = %+v, want emitted=3 accepted=2 proposed=4 rollbacks=1", st)
	}
	if got := st.TokensPerStep(); got != 3 {
		t.Fatalf("TokensPerStep = %v, want exactly 3", got)
	}
	if got := st.AcceptRate(); got != 0.5 {
		t.Fatalf("AcceptRate = %v, want exactly 0.5", got)
	}
	wantCache := append(append([]int(nil), history...), want...)
	if !equalInts(d.cache.Tokens(), wantCache) {
		t.Fatalf("cache = %v, want %v (rejected candidates must not linger)", d.cache.Tokens(), wantCache)
	}
}

// Rejection where the budget trims even the correction token: only the
// delivered prefix may be counted and cached.
func TestRejectedCorrectionTrimmedByBudget(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	const seed = 11
	d := New(tgt, dr, seed, 64, decCfg())

	history := []int{1, 2}
	prop, err := dr.Propose(context.Background(), history, 4, sampling.New(seed))
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	if len(prop.Tokens) != 4 {
		t.Fatalf("proposal = %v, want 4 tokens for this deterministic setup", prop.Tokens)
	}
	rejectToken := prop.Tokens[3]
	fixedToken := (rejectToken + 1) % tgt.Size()
	if fixedToken == 0 {
		fixedToken++
	}
	d.WithVerifier(&forcedRejectVerifier{rows: buildRejectRows(tgt, prop, 3, fixedToken)})

	// Budget 2 < 3 accepted + 1 correction: only 2 tokens are delivered.
	tokens, results, err := d.Generate(context.Background(), history, 2)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	want := prop.Tokens[:2]
	if !equalInts(tokens, want) {
		t.Fatalf("tokens = %v, want trimmed prefix %v", tokens, want)
	}
	r := results[0]
	if r.Emitted != 2 || r.Accepted != 3 || !r.RolledBack {
		t.Fatalf("step result = %+v, want emitted=2 accepted=3 rolled back", r)
	}
	st := d.Stats()
	if st.EmittedTokens != 2 {
		t.Fatalf("EmittedTokens = %d, want exactly the 2 delivered tokens", st)
	}
	if st.AcceptedTokens != 3 || st.ProposedTokens != 4 {
		t.Fatalf("accepted/proposed = %d/%d, want 3/4", st.AcceptedTokens, st.ProposedTokens)
	}
	wantCache := append(append([]int(nil), history...), want...)
	if !equalInts(d.cache.Tokens(), wantCache) {
		t.Fatalf("cache = %v, want %v (trimmed tokens must not linger)", d.cache.Tokens(), wantCache)
	}
}

// EOS accepted mid-proposal ends the sequence early; the recorded emission
// must equal the delivered stream, and nothing past EOS may be cached.
func TestEOSEarlyStopEmittedAccounting(t *testing.T) {
	mcfg := model.Config{VocabSize: 16, Seed: 9, DraftNoise: 0, EOSBias: 1.5}

	// Seed 3: EOS is drafted inside step 3's proposal; the step commits the
	// 3 tokens up to and including EOS, then generation stops.
	d := New(model.NewTarget(mcfg), model.NewDraft(mcfg), 3, 64, Config{DraftK: 4, VerifyTimeoutMS: 500})
	tokens, results, err := d.Generate(context.Background(), []int{5, 6}, 20)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 13 || tokens[len(tokens)-1] != 0 {
		t.Fatalf("tokens = %v, want 13 tokens ending in EOS", tokens)
	}
	st := d.Stats()
	if st.EmittedTokens != int64(len(tokens)) {
		t.Fatalf("EmittedTokens = %d, want exactly %d delivered", st.EmittedTokens, len(tokens))
	}
	if st.EmittedTokens != int64(sumEmitted(results)) {
		t.Fatalf("EmittedTokens = %d, step results sum to %d", st.EmittedTokens, sumEmitted(results))
	}
	if st.Steps != 3 || st.SuccessSteps != 3 {
		t.Fatalf("steps = %d success = %d, want 3/3", st.Steps, st.SuccessSteps)
	}
	if results[2].Emitted != 3 || results[2].Proposed != 3 {
		t.Fatalf("EOS step = %+v, want emitted=3 proposed=3", results[2])
	}
	wantCache := append([]int{5, 6}, tokens...)
	if !equalInts(d.cache.Tokens(), wantCache) {
		t.Fatalf("cache = %v, want prompt+delivered %v (nothing past EOS)", d.cache.Tokens(), wantCache)
	}

	// Seed 555: the bonus token itself is EOS; the whole 5-token step is
	// delivered and counted, then generation stops.
	d2 := New(model.NewTarget(mcfg), model.NewDraft(mcfg), 555, 64, Config{DraftK: 4, VerifyTimeoutMS: 500})
	tokens2, results2, err := d2.Generate(context.Background(), []int{5, 6}, 20)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens2) != 5 || tokens2[len(tokens2)-1] != 0 {
		t.Fatalf("tokens = %v, want 5 tokens ending in EOS", tokens2)
	}
	st2 := d2.Stats()
	if st2.EmittedTokens != 5 || st2.Steps != 1 || st2.EmittedTokens != int64(sumEmitted(results2)) {
		t.Fatalf("stats = %+v, want emitted=5 in a single step", st2)
	}
}

// errVerifier always fails verification with a non-timeout error.
type errVerifier struct{ err error }

func (e *errVerifier) Verify(context.Context, []int, []int) ([][]float64, error) {
	return nil, e.err
}

// A non-timeout verification failure must fall back (one vanilla token per
// step), be classified as verify_error, and be counted separately from
// timeouts — while the request still completes.
func TestVerifyErrorFallbackCountedSeparately(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	d := New(tgt, dr, 21, 64, decCfg())
	d.WithVerifier(&errVerifier{err: errors.New("verify boom")})

	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 3)
	if err != nil {
		t.Fatalf("non-timeout verify error must fall back, not fail: %v", err)
	}
	if len(tokens) != 3 {
		t.Fatalf("delivered %d tokens, want 3 (one vanilla token per fallback step)", len(tokens))
	}
	for i, r := range results {
		if r.Outcome != OutcomeFallback || r.FallbackCause != CauseVerifyError {
			t.Fatalf("step %d = %+v, want fallback with cause %q", i, r, CauseVerifyError)
		}
		if r.Emitted != 1 || !r.RolledBack {
			t.Fatalf("step %d = %+v, want one emitted token and a rollback", i, r)
		}
	}
	st := d.Stats()
	if st.VerifyErrorFallbacks != 3 {
		t.Fatalf("VerifyErrorFallbacks = %d, want 3", st.VerifyErrorFallbacks)
	}
	if st.TimeoutFallbacks != 0 {
		t.Fatalf("TimeoutFallbacks = %d, want 0 (no timeout occurred)", st.TimeoutFallbacks)
	}
	if st.FallbackSteps != 3 || st.SuccessSteps != 0 || st.FailedSteps != 0 {
		t.Fatalf("step counters = %+v, want 3 fallback steps only", st)
	}
	if st.Rollbacks != 3 || st.EmittedTokens != 3 {
		t.Fatalf("rollbacks=%d emitted=%d, want 3/3", st.Rollbacks, st.EmittedTokens)
	}
	// The vanilla fallback tokens must equal non-speculative target output.
	// Reconstruct the exact sampler stream: each step first spends the draft
	// proposal draws, then one draw for the vanilla token.
	want := make([]int, 0, 3)
	vs := sampling.New(21)
	hist := []int{1, 2}
	for len(want) < 3 {
		if _, err := dr.Propose(context.Background(), hist, 4, vs); err != nil {
			t.Fatalf("Propose: %v", err)
		}
		tok := vs.Sample(tgt.Dist(hist))
		want = append(want, tok)
		hist = append(hist, tok)
	}
	if !equalInts(tokens, want) {
		t.Fatalf("fallback tokens = %v, want vanilla target stream %v", tokens, want)
	}
}

// A verify-deadline expiry must be classified as verify_timeout and counted
// separately from other verify errors.
func TestVerifyTimeoutFallbackCountedSeparately(t *testing.T) {
	mcfg := specCfg()
	mcfg.VerifyLatencyMS = 300
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	cfg := decCfg()
	cfg.VerifyTimeoutMS = 20
	d := New(tgt, dr, 21, 64, cfg)

	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 3)
	if err != nil {
		t.Fatalf("timeout must fall back, not fail: %v", err)
	}
	if len(tokens) != 3 {
		t.Fatalf("delivered %d tokens, want 3", len(tokens))
	}
	for i, r := range results {
		if r.Outcome != OutcomeFallback || r.FallbackCause != CauseVerifyTimeout {
			t.Fatalf("step %d = %+v, want fallback with cause %q", i, r, CauseVerifyTimeout)
		}
	}
	st := d.Stats()
	if st.TimeoutFallbacks != 3 {
		t.Fatalf("TimeoutFallbacks = %d, want 3", st.TimeoutFallbacks)
	}
	if st.VerifyErrorFallbacks != 0 {
		t.Fatalf("VerifyErrorFallbacks = %d, want 0 (no non-timeout error occurred)", st.VerifyErrorFallbacks)
	}
	if st.FallbackSteps != 3 || st.EmittedTokens != 3 || st.Rollbacks != 3 {
		t.Fatalf("stats = %+v, want 3 fallback steps, 3 emitted, 3 rollbacks", st)
	}
}

// failOnceVerifier fails the first verification with a non-timeout error and
// delegates the rest to the real target: the request must recover and keep
// generating speculatively.
type failOnceVerifier struct {
	inner *model.Target
	err   error
	calls int
}

func (f *failOnceVerifier) Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error) {
	f.calls++
	if f.calls == 1 {
		return nil, f.err
	}
	return f.inner.Verify(ctx, history, candidates)
}

func TestVerifyErrorDoesNotAbortRequest(t *testing.T) {
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	d := New(tgt, dr, 21, 64, decCfg())
	d.WithVerifier(&failOnceVerifier{inner: tgt, err: errors.New("boom")})

	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 10)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 10 {
		t.Fatalf("delivered %d tokens, want 10", len(tokens))
	}
	if len(results) != 3 {
		t.Fatalf("steps = %d, want 3", len(results))
	}
	if results[0].Outcome != OutcomeFallback || results[0].FallbackCause != CauseVerifyError || results[0].Emitted != 1 {
		t.Fatalf("step 0 = %+v, want 1-token verify_error fallback", results[0])
	}
	if results[1].Outcome != OutcomeSuccess || results[1].Emitted != 5 {
		t.Fatalf("step 1 = %+v, want 5-token success", results[1])
	}
	// Final step is budget-trimmed: 10 - (1+5) = 4 delivered.
	if results[2].Outcome != OutcomeSuccess || results[2].Emitted != 4 {
		t.Fatalf("step 2 = %+v, want 4-token budget-trimmed success", results[2])
	}
	st := d.Stats()
	if st.VerifyErrorFallbacks != 1 || st.TimeoutFallbacks != 0 {
		t.Fatalf("fallback counters = %d/%d, want 1 verify_error and 0 timeouts",
			st.VerifyErrorFallbacks, st.TimeoutFallbacks)
	}
	if st.FallbackSteps != 1 || st.SuccessSteps != 2 || st.Steps != 3 {
		t.Fatalf("step counters = %+v, want 1 fallback + 2 success", st)
	}
	if st.EmittedTokens != 10 || st.EmittedTokens != int64(sumEmitted(results)) {
		t.Fatalf("EmittedTokens = %d, want exactly the 10 delivered tokens", st.EmittedTokens)
	}
	if st.Rollbacks != 1 {
		t.Fatalf("Rollbacks = %d, want 1 (the failed verification)", st.Rollbacks)
	}
}
