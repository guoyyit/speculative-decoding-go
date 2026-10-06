package service

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/spec"
)

// TestAggregateStatsExactMixedRequests: the service-wide aggregate must equal
// the exact sum of the per-request stats, independent of batch size, wait
// window and worker count. Mixes short and long requests so steps include
// budget-clamped endings as well as full speculative steps.
func TestAggregateStatsExactMixedRequests(t *testing.T) {
	configs := []Config{}
	base := testSvcCfg(0.5)
	for _, bs := range []int{1, 3, 16} {
		for _, workers := range []int{1, 4} {
			c := base
			c.BatchSize = bs
			c.Workers = workers
			c.BatchWindowMS = 2
			configs = append(configs, c)
		}
	}

	for ci, cfg := range configs {
		svc := New(cfg)

		type item struct {
			budget int
			seed   uint64
		}
		items := []item{
			{1, 1001}, {2, 1002}, {3, 1003}, {4, 1004}, {5, 1005},
			{8, 1006}, {13, 1007}, {21, 1008}, {2, 1009}, {6, 1010},
			{7, 1011}, {1, 1012}, {12, 1013}, {3, 1014},
		}

		results := make([]*Result, len(items))
		var wg sync.WaitGroup
		for i, it := range items {
			wg.Add(1)
			go func(i int, it item) {
				defer wg.Done()
				results[i] = svc.Submit(context.Background(), Request{
					ID: "m", Prompt: []int{1, 2, (i % 4) + 1},
					MaxNewTokens: it.budget, Seed: it.seed,
				})
			}(i, it)
		}
		wg.Wait()

		var sum spec.Stats
		totalTokens := int64(0)
		for i, r := range results {
			if r == nil || r.Err != nil {
				t.Fatalf("cfg#%d request %d: %v", ci, i, r)
			}
			if len(r.Tokens) != items[i].budget {
				t.Fatalf("cfg#%d request %d: got %d tokens want %d", ci, i, len(r.Tokens), items[i].budget)
			}
			totalTokens += int64(len(r.Tokens))
			s := r.Stats
			if s.EmittedTokens != int64(len(r.Tokens)) {
				t.Fatalf("cfg#%d request %d: per-request EmittedTokens=%d != delivered %d",
					ci, i, s.EmittedTokens, len(r.Tokens))
			}
			sum.Steps += s.Steps
			sum.SuccessSteps += s.SuccessSteps
			sum.FallbackSteps += s.FallbackSteps
			sum.FailedSteps += s.FailedSteps
			sum.ProposedTokens += s.ProposedTokens
			sum.AcceptedTokens += s.AcceptedTokens
			sum.EmittedTokens += s.EmittedTokens
			sum.Rollbacks += s.Rollbacks
			sum.TimeoutFallbacks += s.TimeoutFallbacks
			sum.ErrorFallbacks += s.ErrorFallbacks
		}

		agg := svc.Stats()
		if agg.RequestsOK != int64(len(items)) {
			t.Fatalf("cfg#%d RequestsOK=%d want %d", ci, agg.RequestsOK, len(items))
		}
		if agg.InFlight != 0 {
			t.Fatalf("cfg#%d InFlight=%d want 0", ci, agg.InFlight)
		}
		if agg.EmittedTokens != totalTokens {
			t.Fatalf("cfg#%d aggregate EmittedTokens=%d != total delivered %d", ci, agg.EmittedTokens, totalTokens)
		}
		checks := []struct {
			name string
			got  int64
			want int64
		}{
			{"Steps", agg.SuccessSteps + agg.FallbackSteps + agg.FailedSteps, sum.Steps},
			{"SuccessSteps", agg.SuccessSteps, sum.SuccessSteps},
			{"FallbackSteps", agg.FallbackSteps, sum.FallbackSteps},
			{"FailedSteps", agg.FailedSteps, sum.FailedSteps},
			{"ProposedTokens", agg.ProposedTokens, sum.ProposedTokens},
			{"AcceptedTokens", agg.AcceptedTokens, sum.AcceptedTokens},
			{"EmittedTokens", agg.EmittedTokens, sum.EmittedTokens},
			{"Rollbacks", agg.Rollbacks, sum.Rollbacks},
			{"TimeoutFallbacks", agg.TimeoutFallbacks, sum.TimeoutFallbacks},
			{"ErrorFallbacks", agg.ErrorFallbacks, sum.ErrorFallbacks},
		}
		for _, c := range checks {
			if c.got != c.want {
				t.Fatalf("cfg#%d aggregate %s=%d != sum(per-request)=%d", ci, c.name, c.got, c.want)
			}
		}
		if agg.AcceptRate() > 1+1e-12 || agg.AcceptRate() < 0 {
			t.Fatalf("cfg#%d invalid accept rate %.4f", ci, agg.AcceptRate())
		}
		svc.Close()
	}
}

var errForcedNonTimeout = errors.New("forced non-timeout verify failure")

// flakyTarget fails verification for requests whose prompt starts with
// failMarker, and delegates to the real target otherwise. The failure is a
// plain error (never context.DeadlineExceeded), so it must be classified as
// verify_error rather than verify_timeout.
type flakyTarget struct {
	inner      *model.Target
	failMarker int
}

func (f *flakyTarget) Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error) {
	if len(history) > 0 && history[0] == f.failMarker {
		return nil, errForcedNonTimeout
	}
	return f.inner.Verify(ctx, history, candidates)
}

// TestNonTimeoutVerifyFailureSeparatedFromTimeoutAndIsolated: failed
// requests keep completing via vanilla fallback with verify_error recorded;
// concurrent healthy requests are unaffected; timeout and error counters are
// mutually exclusive per step and exact.
func TestNonTimeoutVerifyFailureSeparatedFromTimeoutAndIsolated(t *testing.T) {
	cfg := testSvcCfg(0.4)
	cfg.Decoder.VerifyTimeoutMS = 1000
	realTarget := model.NewTarget(cfg.Model)
	svc := newServiceWithVerifier(cfg, &flakyTarget{inner: realTarget, failMarker: 90})
	defer svc.Close()

	const badN, goodN = 5, 8
	results := make([]*Result, badN+goodN)
	var wg sync.WaitGroup
	for i := 0; i < badN; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = svc.Submit(context.Background(), Request{
				ID: "bad", Prompt: []int{90}, MaxNewTokens: 4, Seed: uint64(400 + i),
			})
		}(i)
	}
	for i := 0; i < goodN; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[badN+i] = svc.Submit(context.Background(), Request{
				ID: "good", Prompt: []int{1, 2}, MaxNewTokens: 6, Seed: uint64(500 + i),
			})
		}(i)
	}
	wg.Wait()

	var badSum, goodSum spec.Stats
	for i, r := range results {
		if r == nil || r.Err != nil {
			t.Fatalf("request %d returned err: %v", i, r)
		}
		if i < badN {
			if len(r.Tokens) != 4 {
				t.Fatalf("bad request %d got %d tokens, want 4", i, len(r.Tokens))
			}
			if r.Stats.ErrorFallbacks <= 0 {
				t.Fatalf("bad request %d recorded no verify_error: %+v", i, r.Stats)
			}
			if r.Stats.TimeoutFallbacks != 0 {
				t.Fatalf("non-timeout failure miscounted as timeout: %+v", r.Stats)
			}
			if r.Stats.FallbackSteps != r.Stats.ErrorFallbacks {
				t.Fatalf("every fallback must be a verify_error here: %+v", r.Stats)
			}
			if r.Stats.FailedSteps != 0 {
				t.Fatalf("verify error must not fail the request: %+v", r.Stats)
			}
			badSum.EmittedTokens += r.Stats.EmittedTokens
			badSum.ErrorFallbacks += r.Stats.ErrorFallbacks
			badSum.FallbackSteps += r.Stats.FallbackSteps
			badSum.Rollbacks += r.Stats.Rollbacks
		} else {
			if len(r.Tokens) != 6 {
				t.Fatalf("good request %d got %d tokens, want 6", i, len(r.Tokens))
			}
			if r.Stats.ErrorFallbacks != 0 || r.Stats.TimeoutFallbacks != 0 {
				t.Fatalf("healthy request %d recorded fallbacks: %+v", i, r.Stats)
			}
			goodSum.EmittedTokens += r.Stats.EmittedTokens
			goodSum.SuccessSteps += r.Stats.SuccessSteps
			// Healthy requests may still roll back on ordinary speculative
			// rejections; those rollbacks count too.
			goodSum.Rollbacks += r.Stats.Rollbacks
		}
	}

	agg := svc.Stats()
	if agg.RequestsOK != badN+goodN {
		t.Fatalf("RequestsOK=%d want %d", agg.RequestsOK, badN+goodN)
	}
	if agg.ErrorFallbacks != badSum.ErrorFallbacks || agg.ErrorFallbacks < badN {
		t.Fatalf("aggregate ErrorFallbacks=%d, bad-request sum=%d", agg.ErrorFallbacks, badSum.ErrorFallbacks)
	}
	if agg.TimeoutFallbacks != 0 {
		t.Fatalf("aggregate TimeoutFallbacks=%d want 0", agg.TimeoutFallbacks)
	}
	if agg.FallbackSteps != badSum.FallbackSteps {
		t.Fatalf("aggregate FallbackSteps=%d != %d", agg.FallbackSteps, badSum.FallbackSteps)
	}
	if agg.Rollbacks != badSum.Rollbacks+goodSum.Rollbacks {
		t.Fatalf("aggregate Rollbacks=%d != bad %d + good %d", agg.Rollbacks, badSum.Rollbacks, goodSum.Rollbacks)
	}
	if agg.EmittedTokens != badSum.EmittedTokens+goodSum.EmittedTokens {
		t.Fatalf("EmittedTokens=%d != %d+%d", agg.EmittedTokens, badSum.EmittedTokens, goodSum.EmittedTokens)
	}
	if agg.EmittedTokens != int64(badN*4+goodN*6) {
		t.Fatalf("EmittedTokens=%d want %d", agg.EmittedTokens, badN*4+goodN*6)
	}
	if agg.SuccessSteps != goodSum.SuccessSteps {
		t.Fatalf("SuccessSteps=%d want healthy sum %d", agg.SuccessSteps, goodSum.SuccessSteps)
	}

	// A rerun of one healthy request after the storm must be unaffected.
	r := svc.Submit(context.Background(), Request{ID: "after", Prompt: []int{1, 2}, MaxNewTokens: 6, Seed: 500})
	if r.Err != nil || len(r.Tokens) != 6 {
		t.Fatalf("post-failure healthy request: err=%v tokens=%v", r.Err, r.Tokens)
	}
	if r.Stats.ErrorFallbacks != 0 {
		t.Fatalf("post-failure healthy request saw verify errors: %+v", r.Stats)
	}
}

// TestLongRunStatsExactAndBounded: many long generations keep exact
// aggregate accounting (aggregate == sum, emitted == delivered) while the
// service schedules many requests; after Close all batcher goroutines drain.
func TestLongRunStatsExactAndBounded(t *testing.T) {
	cfg := testSvcCfg(0.5)
	cfg.CacheCapacity = model.ContextWindow
	cfg.BatchSize = 4
	cfg.Workers = 3
	svc := New(cfg)

	const reqs = 24
	const budget = 60
	results := make([]*Result, reqs)
	var wg sync.WaitGroup
	for i := 0; i < reqs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = svc.Submit(context.Background(), Request{
				ID: "long", Prompt: []int{1, 2}, MaxNewTokens: budget, Seed: uint64(9000 + i),
			})
		}(i)
	}
	wg.Wait()

	var sumEmitted int64
	var sumSteps int64
	for i, r := range results {
		if r == nil || r.Err != nil {
			t.Fatalf("request %d: %v", i, r)
		}
		if len(r.Tokens) > budget {
			t.Fatalf("request %d got %d tokens > budget %d", i, len(r.Tokens), budget)
		}
		if r.Stats.EmittedTokens != int64(len(r.Tokens)) {
			t.Fatalf("request %d EmittedTokens=%d != delivered %d", i, r.Stats.EmittedTokens, len(r.Tokens))
		}
		sumEmitted += r.Stats.EmittedTokens
		sumSteps += r.Stats.Steps
	}
	agg := svc.Stats()
	if agg.EmittedTokens != sumEmitted {
		t.Fatalf("EmittedTokens agg=%d sum=%d", agg.EmittedTokens, sumEmitted)
	}
	if agg.SuccessSteps+agg.FallbackSteps+agg.FailedSteps != sumSteps {
		t.Fatalf("step total agg=%d sum=%d", agg.SuccessSteps+agg.FallbackSteps+agg.FailedSteps, sumSteps)
	}
	if agg.InFlight != 0 {
		t.Fatalf("InFlight=%d want 0", agg.InFlight)
	}
	if agg.AcceptRate() > 1+1e-12 {
		t.Fatalf("accept rate above 1: %.4f", agg.AcceptRate())
	}

	before := runtime.NumGoroutine()
	svc.Close()
	// After Close the batcher loops must terminate; allow a short drain.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && runtime.NumGoroutine() > before {
		time.Sleep(5 * time.Millisecond)
	}
	if after := runtime.NumGoroutine(); after > before {
		t.Fatalf("goroutines leaked after Close: before=%d after=%d", before, after)
	}
}

// halfFlakyTarget fails verifications for marker 71 and makes marker 81 slow
// (latency driven by the real model config), in the same service run, so
// verify_error and verify_timeout are counted side by side.
type halfFlakyTarget struct {
	slow        *model.Target
	fast        *model.Target
	errorMarker int
	slowMarker  int
}

func (f *halfFlakyTarget) Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error) {
	if len(history) > 0 && history[0] == f.errorMarker {
		return nil, errForcedNonTimeout
	}
	if len(history) > 0 && history[0] == f.slowMarker {
		return f.slow.Verify(ctx, history, candidates)
	}
	return f.fast.Verify(ctx, history, candidates)
}

// TestTimeoutAndNonTimeoutCountedSeparately: in one service, one request
// class hits genuine deadline expiry (verify_timeout) while another class
// hits non-timeout errors (verify_error); counters are disjoint and exact,
// and both classes complete.
func TestTimeoutAndNonTimeoutCountedSeparately(t *testing.T) {
	cfg := testSvcCfg(0.4)
	cfg.Model.VerifyLatencyMS = 0
	cfg.Decoder.VerifyTimeoutMS = 15
	fast := model.NewTarget(cfg.Model)
	slowCfg := cfg.Model
	slowCfg.VerifyLatencyMS = 200
	slow := model.NewTarget(slowCfg)
	svc := newServiceWithVerifier(cfg, &halfFlakyTarget{
		slow: slow, fast: fast, errorMarker: 71, slowMarker: 81,
	})
	defer svc.Close()

	var wg sync.WaitGroup
	var timeoutRes, errorRes, goodRes *Result
	wg.Add(3)
	go func() {
		defer wg.Done()
		timeoutRes = svc.Submit(context.Background(), Request{ID: "to", Prompt: []int{81}, MaxNewTokens: 3, Seed: 11})
	}()
	go func() {
		defer wg.Done()
		errorRes = svc.Submit(context.Background(), Request{ID: "er", Prompt: []int{71}, MaxNewTokens: 3, Seed: 12})
	}()
	go func() {
		defer wg.Done()
		goodRes = svc.Submit(context.Background(), Request{ID: "ok", Prompt: []int{1, 2}, MaxNewTokens: 3, Seed: 13})
	}()
	wg.Wait()

	for name, r := range map[string]*Result{"timeout": timeoutRes, "error": errorRes, "good": goodRes} {
		if r == nil || r.Err != nil || len(r.Tokens) != 3 {
			t.Fatalf("%s request: %+v", name, r)
		}
	}
	if timeoutRes.Stats.TimeoutFallbacks != timeoutRes.Stats.FallbackSteps || timeoutRes.Stats.TimeoutFallbacks == 0 {
		t.Fatalf("timeout request stats wrong: %+v", timeoutRes.Stats)
	}
	if timeoutRes.Stats.ErrorFallbacks != 0 {
		t.Fatalf("timeout request miscounted as error: %+v", timeoutRes.Stats)
	}
	if errorRes.Stats.ErrorFallbacks != errorRes.Stats.FallbackSteps || errorRes.Stats.ErrorFallbacks == 0 {
		t.Fatalf("error request stats wrong: %+v", errorRes.Stats)
	}
	if errorRes.Stats.TimeoutFallbacks != 0 {
		t.Fatalf("error request miscounted as timeout: %+v", errorRes.Stats)
	}
	if goodRes.Stats.TimeoutFallbacks != 0 || goodRes.Stats.ErrorFallbacks != 0 {
		t.Fatalf("good request recorded fallbacks: %+v", goodRes.Stats)
	}

	agg := svc.Stats()
	if agg.TimeoutFallbacks != timeoutRes.Stats.TimeoutFallbacks {
		t.Fatalf("aggregate timeouts=%d want %d", agg.TimeoutFallbacks, timeoutRes.Stats.TimeoutFallbacks)
	}
	if agg.ErrorFallbacks != errorRes.Stats.ErrorFallbacks {
		t.Fatalf("aggregate errors=%d want %d", agg.ErrorFallbacks, errorRes.Stats.ErrorFallbacks)
	}
	if agg.EmittedTokens != 9 {
		t.Fatalf("aggregate EmittedTokens=%d want 9", agg.EmittedTokens)
	}
}
