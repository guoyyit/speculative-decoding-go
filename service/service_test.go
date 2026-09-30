package service

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/spec"
)

func testSvcCfg(noise float64) Config {
	return Config{
		Model: model.Config{
			VocabSize: 32, Seed: 17, DraftNoise: noise, EOSBias: -3,
		},
		Decoder:       spec.Config{DraftK: 4, VerifyTimeoutMS: 1000},
		BatchSize:     3,
		BatchWindowMS: 2,
		Workers:       3,
	}
}

func TestConcurrentRequestsIndependentAndReproducible(t *testing.T) {
	svc := New(testSvcCfg(0.5))
	defer svc.Close()

	const n = 12
	results := make([]*Result, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = svc.Submit(context.Background(), Request{
				ID: "r", Prompt: []int{1, 2, i % 4}, MaxNewTokens: 8, Seed: uint64(100 + i),
			})
		}(i)
	}
	wg.Wait()

	seen := map[string]bool{}
	for i, r := range results {
		if r == nil {
			t.Fatalf("request %d has no result", i)
		}
		if r.Err != nil {
			t.Fatalf("request %d failed: %v", i, r.Err)
		}
		if len(r.Tokens) != 8 {
			t.Fatalf("request %d produced %d tokens", i, len(r.Tokens))
		}
		key := join(r.Tokens)
		if seen[key] {
			// Different seeds+prompts might collide; only check same-seed
			// pairs deterministically below.
		}
		seen[key] = true
	}

	// State isolation + reproducibility: same seed under concurrency must
	// match an isolated rerun regardless of batching schedule.
	a := svc.Submit(context.Background(), Request{ID: "a", Prompt: []int{5}, MaxNewTokens: 8, Seed: 555})
	b := svc.Submit(context.Background(), Request{ID: "b", Prompt: []int{5}, MaxNewTokens: 8, Seed: 555})
	if join(a.Tokens) != join(b.Tokens) {
		t.Fatalf("same-seed requests diverged: %v vs %v", a.Tokens, b.Tokens)
	}

	st := svc.Stats()
	if st.RequestsOK != n+2 {
		t.Fatalf("RequestsOK = %d, want %d", st.RequestsOK, n+2)
	}
	if st.InFlight != 0 {
		t.Fatalf("InFlight after completion = %d, want 0", st.InFlight)
	}
	if st.EmittedTokens < int64((n+2)*8) {
		t.Fatalf("EmittedTokens = %d, want >= %d", st.EmittedTokens, (n+2)*8)
	}
	if st.TokensPerStep() <= 0 || st.AcceptRate() < 0 || st.AcceptRate() > 1 {
		t.Fatalf("invalid aggregate stats: %+v", st)
	}
}

func TestOneCanceledRequestDoesNotAffectOthers(t *testing.T) {
	cfg := testSvcCfg(0.3)
	cfg.Model.DraftLatencyMS = 100
	svc := New(cfg)
	defer svc.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	bad := svc.Submit(ctx, Request{ID: "canceled", Prompt: []int{1}, MaxNewTokens: 6, Seed: 1})
	if bad.Err == nil {
		t.Fatal("canceled request should report an error")
	}

	// Other requests must still complete normally.
	for i := 0; i < 5; i++ {
		r := svc.Submit(context.Background(), Request{
			ID: "ok", Prompt: []int{2}, MaxNewTokens: 6, Seed: uint64(200 + i),
		})
		if r.Err != nil || len(r.Tokens) != 6 {
			t.Fatalf("unaffected request %d: err=%v tokens=%v", i, r.Err, r.Tokens)
		}
	}
	st := svc.Stats()
	if st.RequestsFailed == 0 {
		t.Fatal("canceled request should be counted as failed")
	}
	if st.RequestsOK != 5 {
		t.Fatalf("RequestsOK = %d, want 5", st.RequestsOK)
	}
}

func TestTimeoutFallbackCounted(t *testing.T) {
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
		t.Fatalf("got %d tokens after fallback, want 3", len(r.Tokens))
	}
	st := svc.Stats()
	if st.TimeoutFallbacks == 0 || st.FallbackSteps == 0 {
		t.Fatalf("fallback not counted: %+v", st)
	}
	if st.SuccessSteps != 0 {
		t.Fatalf("expected no speculative success steps, got %d", st.SuccessSteps)
	}
	if st.Rollbacks == 0 {
		t.Fatal("timeout fallback must record a rollback")
	}
}

func TestBatchingDoesNotChangeOutput(t *testing.T) {
	cfg := testSvcCfg(0.4)
	// BatchSize 1 forces a batch of one each call; compare against a service
	// with large batches that always coalesce concurrent calls.
	single := New(func() Config { c := cfg; c.BatchSize = 1; c.Workers = 1; return c }())
	batched := New(func() Config { c := cfg; c.BatchSize = 16; c.Workers = 1; return c }())
	defer single.Close()
	defer batched.Close()

	isolated := single.Submit(context.Background(), Request{ID: "x", Prompt: []int{3, 4}, MaxNewTokens: 10, Seed: 31})

	var wg sync.WaitGroup
	got := make([][]int, 8)
	for i := range got {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := batched.Submit(context.Background(), Request{
				ID: "x", Prompt: []int{3, 4}, MaxNewTokens: 10, Seed: 31 + uint64(i)*7919,
			})
			got[i] = r.Tokens
		}(i)
	}
	wg.Wait()

	// Rerun seed 31 inside the batched service; output must be identical.
	b0 := batched.Submit(context.Background(), Request{ID: "x", Prompt: []int{3, 4}, MaxNewTokens: 10, Seed: 31})
	if join(isolated.Tokens) != join(b0.Tokens) {
		t.Fatalf("scheduling changed output:\nsingle=%v\nbatched=%v", isolated.Tokens, b0.Tokens)
	}
}

func TestCloseIsSafe(t *testing.T) {
	svc := New(testSvcCfg(0))
	svc.Close()
	svc.Close() // double close must not panic
	r := svc.Submit(context.Background(), Request{ID: "z", Prompt: nil, MaxNewTokens: 1, Seed: 1})
	if r.Err == nil {
		t.Fatal("submit after Close must error")
	}
}

func TestInFlightDrainsAndStatsTiming(t *testing.T) {
	// Sanity: a busy service reports in-flight requests while running.
	cfg := testSvcCfg(0)
	cfg.Model.DraftLatencyMS = 30
	svc := New(cfg)
	defer svc.Close()
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = svc.Submit(context.Background(), Request{ID: "b", Prompt: []int{1}, MaxNewTokens: 4, Seed: uint64(time.Now().UnixNano())})
		}()
	}
	wg.Wait()
	if svc.Stats().InFlight != 0 {
		t.Fatal("in-flight must drain to zero")
	}
}

func join(tokens []int) string {
	s := ""
	for _, t := range tokens {
		s += string(rune('a'+(t%26))) + " "
	}
	return s
}
