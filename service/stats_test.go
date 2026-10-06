package service

import (
	"context"
	"sync"
	"testing"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/spec"
)

// runDirect executes the same requests one by one through standalone
// decoders (no service, no batching) and returns each request's tokens plus
// the exact sum of all per-request decoder stats. Because the models are
// stateless and every request owns its seeded sampler, the direct path must
// produce token-identical and stat-identical results to the service path.
func runDirect(t *testing.T, cfg Config, reqs []Request) ([][]int, spec.Stats) {
	t.Helper()
	target := model.NewTarget(cfg.Model)
	draft := model.NewDraft(cfg.Model)
	cacheCap := cfg.CacheCapacity
	if cacheCap < model.ContextWindow {
		cacheCap = model.ContextWindow
	}
	tokens := make([][]int, len(reqs))
	var sum spec.Stats
	for i, req := range reqs {
		d := spec.New(target, draft, req.Seed, cacheCap, cfg.Decoder)
		out, _, err := d.Generate(context.Background(), req.Prompt, req.MaxNewTokens)
		if err != nil {
			t.Fatalf("direct request %d: %v", i, err)
		}
		tokens[i] = out
		st := d.Stats()
		sum.SuccessSteps += st.SuccessSteps
		sum.FallbackSteps += st.FallbackSteps
		sum.FailedSteps += st.FailedSteps
		sum.ProposedTokens += st.ProposedTokens
		sum.AcceptedTokens += st.AcceptedTokens
		sum.EmittedTokens += st.EmittedTokens
		sum.Rollbacks += st.Rollbacks
		sum.TimeoutFallbacks += st.TimeoutFallbacks
		sum.VerifyErrorFallbacks += st.VerifyErrorFallbacks
	}
	return tokens, sum
}

func mixedRequests() []Request {
	// Mix requests shorter than, equal to, and longer than one full
	// speculative step (DraftK+1 = 5 tokens), plus multi-step ones.
	lengths := []int{1, 2, 3, 4, 5, 6, 7, 9, 12, 16}
	reqs := make([]Request, 0, len(lengths)*2)
	for i, n := range lengths {
		for j := 0; j < 2; j++ {
			reqs = append(reqs, Request{
				ID:           "m",
				Prompt:       []int{1, 2, (i + j) % 5},
				MaxNewTokens: n,
				Seed:         uint64(1000 + i*31 + j*777),
			})
		}
	}
	return reqs
}

func submitAll(svc *Service, reqs []Request) []*Result {
	results := make([]*Result, len(reqs))
	var wg sync.WaitGroup
	for i := range reqs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = svc.Submit(context.Background(), reqs[i])
		}(i)
	}
	wg.Wait()
	return results
}

// Concurrent long/short mixed requests: the service aggregate must equal,
// exactly, the sum of per-request decoder stats, and EmittedTokens must
// equal the exact number of tokens delivered to callers.
func TestAggregateStatsExactMixedLengths(t *testing.T) {
	cfg := testSvcCfg(0.4)
	cfg.CacheCapacity = 64
	reqs := mixedRequests()

	svc := New(cfg)
	defer svc.Close()
	results := submitAll(svc, reqs)

	directTokens, want := runDirect(t, cfg, reqs)

	var delivered int64
	for i, r := range results {
		if r.Err != nil {
			t.Fatalf("request %d failed: %v", i, r.Err)
		}
		if join(r.Tokens) != join(directTokens[i]) {
			t.Fatalf("request %d: service tokens %v != direct tokens %v", i, r.Tokens, directTokens[i])
		}
		delivered += int64(len(r.Tokens))
	}

	st := svc.Stats()
	if st.RequestsOK != int64(len(reqs)) || st.RequestsFailed != 0 {
		t.Fatalf("requests ok/failed = %d/%d, want %d/0", st.RequestsOK, st.RequestsFailed, len(reqs))
	}
	if st.EmittedTokens != delivered {
		t.Fatalf("EmittedTokens = %d, want exactly the %d delivered tokens", st.EmittedTokens, delivered)
	}
	if st.EmittedTokens != want.EmittedTokens {
		t.Fatalf("EmittedTokens = %d, direct sum = %d", st.EmittedTokens, want.EmittedTokens)
	}
	if st.SuccessSteps != want.SuccessSteps || st.FallbackSteps != want.FallbackSteps || st.FailedSteps != want.FailedSteps {
		t.Fatalf("step counters = %d/%d/%d, direct sum = %d/%d/%d",
			st.SuccessSteps, st.FallbackSteps, st.FailedSteps,
			want.SuccessSteps, want.FallbackSteps, want.FailedSteps)
	}
	if st.ProposedTokens != want.ProposedTokens || st.AcceptedTokens != want.AcceptedTokens {
		t.Fatalf("proposed/accepted = %d/%d, direct sum = %d/%d",
			st.ProposedTokens, st.AcceptedTokens, want.ProposedTokens, want.AcceptedTokens)
	}
	if st.Rollbacks != want.Rollbacks || st.TimeoutFallbacks != want.TimeoutFallbacks || st.VerifyErrorFallbacks != want.VerifyErrorFallbacks {
		t.Fatalf("rollback/timeout/verify-error = %d/%d/%d, direct sum = %d/%d/%d",
			st.Rollbacks, st.TimeoutFallbacks, st.VerifyErrorFallbacks,
			want.Rollbacks, want.TimeoutFallbacks, want.VerifyErrorFallbacks)
	}
	totalSteps := st.SuccessSteps + st.FallbackSteps + st.FailedSteps
	if totalSteps != want.SuccessSteps+want.FallbackSteps+want.FailedSteps {
		t.Fatalf("total steps = %d, want %d", totalSteps, want.SuccessSteps+want.FallbackSteps+want.FailedSteps)
	}
	if got := st.TokensPerStep(); got != float64(st.EmittedTokens)/float64(totalSteps) {
		t.Fatalf("TokensPerStep = %v, want emitted/steps = %v", got, float64(st.EmittedTokens)/float64(totalSteps))
	}
	if st.AcceptRate() < 0 || st.AcceptRate() > 1 {
		t.Fatalf("AcceptRate out of [0,1]: %v", st.AcceptRate())
	}
}

// The aggregate statistics must not depend on batch size, batching window,
// or worker count: scheduling only groups stateless model calls.
func TestAggregateStatsIndependentOfScheduling(t *testing.T) {
	base := testSvcCfg(0.4)
	base.CacheCapacity = 64
	reqs := mixedRequests()

	variants := []Config{
		func() Config { c := base; c.BatchSize = 1; c.BatchWindowMS = 1; c.Workers = 1; return c }(),
		func() Config { c := base; c.BatchSize = 8; c.BatchWindowMS = 2; c.Workers = 4; return c }(),
		func() Config { c := base; c.BatchSize = 64; c.BatchWindowMS = 10; c.Workers = 8; return c }(),
	}

	var reference *Stats
	for i, cfg := range variants {
		svc := New(cfg)
		results := submitAll(svc, reqs)
		st := svc.Stats()
		svc.Close()
		for j, r := range results {
			if r.Err != nil {
				t.Fatalf("variant %d request %d failed: %v", i, j, r.Err)
			}
		}
		if reference == nil {
			ref := st
			reference = &ref
			continue
		}
		if st != *reference {
			t.Fatalf("variant %d stats differ from reference:\n got %+v\nwant %+v", i, st, *reference)
		}
	}
}

// Service-level timeout fallback with exact counts: every verification
// exceeds the deadline, so every step is a one-token timeout fallback.
func TestServiceTimeoutFallbackExactCounts(t *testing.T) {
	cfg := testSvcCfg(0.3)
	cfg.Model.VerifyLatencyMS = 300
	cfg.Decoder.VerifyTimeoutMS = 15
	svc := New(cfg)
	defer svc.Close()

	r := svc.Submit(context.Background(), Request{ID: "t", Prompt: []int{1}, MaxNewTokens: 3, Seed: 9})
	if r.Err != nil {
		t.Fatalf("timeout must fall back, not fail: %v", r.Err)
	}
	if len(r.Tokens) != 3 {
		t.Fatalf("delivered %d tokens, want 3", len(r.Tokens))
	}
	st := svc.Stats()
	if st.TimeoutFallbacks != 3 {
		t.Fatalf("TimeoutFallbacks = %d, want exactly 3", st.TimeoutFallbacks)
	}
	if st.VerifyErrorFallbacks != 0 {
		t.Fatalf("VerifyErrorFallbacks = %d, want 0", st.VerifyErrorFallbacks)
	}
	if st.FallbackSteps != 3 || st.SuccessSteps != 0 || st.FailedSteps != 0 {
		t.Fatalf("step counters = %+v, want exactly 3 fallback steps", st)
	}
	if st.EmittedTokens != 3 || st.Rollbacks != 3 {
		t.Fatalf("emitted/rollbacks = %d/%d, want 3/3", st.EmittedTokens, st.Rollbacks)
	}
	if got := st.TokensPerStep(); got != 1 {
		t.Fatalf("TokensPerStep = %v, want exactly 1 (pure fallback)", got)
	}
}
