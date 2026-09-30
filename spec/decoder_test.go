package spec

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/sampling"
)

func specCfg() model.Config {
	return model.Config{
		VocabSize: 32, Seed: 5, DraftNoise: 0.6, EOSBias: -2.5,
	}
}

func decCfg() Config {
	return Config{DraftK: 4, VerifyTimeoutMS: 500}
}

func TestIdenticalDraftAllAcceptedTokensPerStepGT1(t *testing.T) {
	mcfg := specCfg()
	mcfg.DraftNoise = 0
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	d := New(tgt, dr, 42, 64, decCfg())

	tokens, results, err := d.Generate(context.Background(), []int{1, 2}, 10)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 10 {
		t.Fatalf("generated %d tokens, want 10", len(tokens))
	}
	for _, r := range results {
		if r.Outcome != OutcomeSuccess {
			t.Fatalf("step outcome = %s, want success", r.Outcome)
		}
		if r.Emitted <= 1 {
			t.Fatalf("close distributions: emitted = %d, want > 1", r.Emitted)
		}
	}
	st := d.Stats()
	if st.TokensPerStep() <= 1 {
		t.Fatalf("tokens/step = %.2f, want > 1 when draft ~= target", st.TokensPerStep())
	}
	if st.AcceptRate() < 0.999 {
		t.Fatalf("accept rate = %.3f, want 1.0 for identical distributions", st.AcceptRate())
	}
}

func TestFixedSeedReproducible(t *testing.T) {
	mcfg := specCfg()
	prompt := []int{3, 7, 9}
	run := func() []int {
		d := New(model.NewTarget(mcfg), model.NewDraft(mcfg), 98765, 64, decCfg())
		tokens, _, err := d.Generate(context.Background(), prompt, 12)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		return tokens
	}
	first := run()
	second := run()
	if len(first) == 0 {
		t.Fatal("expected generated tokens")
	}
	if !equalInts(first, second) {
		t.Fatalf("same seed produced different outputs:\n%v\n%v", first, second)
	}

	// A different seed should (with overwhelming probability) differ.
	d2 := New(model.NewTarget(mcfg), model.NewDraft(mcfg), 1, 64, decCfg())
	other, _, _ := d2.Generate(context.Background(), prompt, 12)
	if equalInts(first, other) {
		t.Fatal("different seeds produced identical output (suspicious)")
	}
}

func TestRejectedCandidateNotCommittedAndRolledBack(t *testing.T) {
	// Force rejection of the first candidate with a verifier whose row makes
	// the proposed token impossible for the target.
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	d := New(tgt, dr, 11, 64, decCfg())

	// Propose directly to learn what the draft would emit, then craft a
	// target row that places zero mass on that token.
	history := []int{1, 2}
	prop, err := dr.Propose(context.Background(), history, 4, sampling.New(11))
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	v := &forcedVerifier{rows: make([][]float64, 0)}
	rejectToken := prop.Tokens[0]
	row0 := make([]float64, tgt.Size())
	// Put all mass on a different token, so acceptance prob is 0.
	fixedToken := (rejectToken + 1) % tgt.Size()
	if fixedToken == 0 {
		fixedToken++
	}
	row0[fixedToken] = 1
	v.rows = append(v.rows, row0)
	// Remaining rows unused after first rejection, but supply them anyway.
	for i := 1; i <= len(prop.Tokens); i++ {
		v.rows = append(v.rows, tgt.Dist(append(append([]int(nil), history...), prop.Tokens[:i]...)))
	}
	d.WithVerifier(v)

	tokens, results, err := d.Generate(context.Background(), history, 1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != fixedToken {
		t.Fatalf("got %v, want exactly the correction token %d", tokens, fixedToken)
	}
	if results[0].Accepted != 0 || !results[0].RolledBack || results[0].Emitted != 1 {
		t.Fatalf("unexpected step result: %+v", results[0])
	}
	// No rejected candidate may persist: cache must hold history + fixedToken.
	want := append(append([]int(nil), history...), fixedToken)
	if !equalInts(d.cache.Tokens(), want) {
		t.Fatalf("cache = %v, want %v (rejected candidate leaked)", d.cache.Tokens(), want)
	}
	if d.Stats().Rollbacks != 1 {
		t.Fatalf("rollbacks = %d, want 1", d.Stats().Rollbacks)
	}
}

func TestRollbackAfterBoundedEviction(t *testing.T) {
	// Cache exactly at the model context window; a long prompt fills it so
	// that staging candidates evicts older entries. A forced first-position
	// rejection must still restore the true accepted prefix (no leaked token).
	mcfg := specCfg()
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	d := New(tgt, dr, 11, model.ContextWindow, decCfg())

	prompt := []int{2, 3, 4, 5, 6, 7, 8, 9} // len == ContextWindow
	prop, err := dr.Propose(context.Background(), prompt, 4, sampling.New(11))
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	rejectToken := prop.Tokens[0]
	fixedToken := rejectToken + 1
	if fixedToken >= tgt.Size() {
		fixedToken = 1
	}
	if fixedToken == 0 {
		fixedToken = 2
	}
	row0 := make([]float64, tgt.Size())
	row0[fixedToken] = 1
	rows := [][]float64{row0}
	for i := 1; i <= len(prop.Tokens); i++ {
		rows = append(rows, tgt.Dist(append(append([]int(nil), prompt...), prop.Tokens[:i]...)))
	}
	d.WithVerifier(&forcedVerifier{rows: rows})

	tokens, _, err := d.Generate(context.Background(), prompt, 1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != fixedToken {
		t.Fatalf("output = %v, want correction token %d", tokens, fixedToken)
	}
	// Cache length stays bounded and its last token is the correction; none
	// of the rejected candidates may appear after the accept point.
	cached := d.cache.Tokens()
	if len(cached) > model.ContextWindow {
		t.Fatalf("cache grew beyond capacity: %d", len(cached))
	}
	if cached[len(cached)-1] != fixedToken {
		t.Fatalf("cache tail = %d, want %d; full cache=%v", cached[len(cached)-1], fixedToken, cached)
	}
	// The accepted prefix length is unchanged: no rejected draft tokens at
	// positions beyond prompt window boundary.
	for _, tk := range prop.Tokens {
		if tk == fixedToken {
			continue
		}
		// Candidate tokens must not survive in the tail beyond the prefix.
		for _, c := range cached[len(cached)-1:] {
			if c == tk {
				t.Fatalf("rejected candidate %d leaked into cache %v", tk, cached)
			}
		}
	}
}

func TestVerifyTimeoutFallback(t *testing.T) {
	mcfg := specCfg()
	mcfg.VerifyLatencyMS = 300
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	cfg := decCfg()
	cfg.VerifyTimeoutMS = 20
	d := New(tgt, dr, 21, 64, cfg)

	start := time.Now()
	tokens, results, err := d.Generate(context.Background(), []int{1}, 3)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(tokens) != 3 {
		t.Fatalf("generated %d tokens after fallback, want 3", len(tokens))
	}
	for _, r := range results {
		if r.Outcome != OutcomeFallback || r.FallbackCause != CauseVerifyTimeout {
			t.Fatalf("step = %+v, want timeout fallback", r)
		}
		if !r.RolledBack || r.Emitted != 1 || r.Proposed == 0 {
			t.Fatalf("fallback must roll back proposal and emit one token: %+v", r)
		}
	}
	st := d.Stats()
	if st.TimeoutFallbacks == 0 {
		t.Fatal("expected TimeoutFallbacks > 0")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("fallback took %v; timeout must not wait for full verify", elapsed)
	}
}

func TestCanceledRequestProducesNothing(t *testing.T) {
	mcfg := specCfg()
	mcfg.DraftLatencyMS = 200
	tgt := model.NewTarget(mcfg)
	dr := model.NewDraft(mcfg)
	d := New(tgt, dr, 31, 64, decCfg())

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tokens, _, err := d.Generate(ctx, []int{1, 2}, 5)
	if err == nil {
		t.Fatal("canceled request must return an error")
	}
	if len(tokens) != 0 {
		t.Fatalf("canceled request emitted %v; want no tokens", tokens)
	}
	// Cache must only contain the prompt: no intermediate candidate state.
	if !equalInts(d.cache.Tokens(), []int{1, 2}) {
		t.Fatalf("cache after cancel = %v, want prompt only", d.cache.Tokens())
	}
}

type forcedVerifier struct {
	rows [][]float64
}

var errForcedRows = errors.New("forced verifier: row mismatch")

func (f *forcedVerifier) Verify(_ context.Context, _ []int, candidates []int) ([][]float64, error) {
	if len(f.rows) != len(candidates)+1 {
		return nil, errForcedRows
	}
	return f.rows, nil
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
