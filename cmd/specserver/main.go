package main

import (
	"context"
	"fmt"

	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/service"
	"github.com/guoyyit/speculative-decoding-go/spec"
)

func main() {
	cfg := service.Config{
		Model: model.Config{
			VocabSize:       64,
			Seed:            42,
			DraftNoise:      0, // draft == target: all proposals are accepted
			EOSBias:         -3,
			TargetLatencyMS: 0,
			DraftLatencyMS:  0,
			VerifyLatencyMS: 0,
		},
		Decoder: spec.Config{DraftK: 5, VerifyTimeoutMS: 1000},
		Workers: 4,
	}

	svc := service.New(cfg)
	defer svc.Close()

	ctx := context.Background()
	prompt := []int{1, 2, 3}

	// Run several concurrent independent requests.
	results := make([]*service.Result, 4)
	done := make(chan struct{})
	for i := range results {
		i := i
		go func() {
			req := service.Request{
				ID:           fmt.Sprintf("r%d", i),
				Prompt:       prompt,
				MaxNewTokens: 12,
				Seed:         uint64(1000 + i),
			}
			results[i] = svc.Submit(ctx, req)
			done <- struct{}{}
		}()
	}
	for range results {
		<-done
	}

	for _, r := range results {
		status := "ok"
		if r.Err != nil {
			status = r.Err.Error()
		}
		fmt.Printf("request %s: tokens=%v (%s)\n", r.ID, r.Tokens, status)
	}

	st := svc.Stats()
	fmt.Printf("\nstats: accept-rate=%.3f tokens/step=%.3f success=%d fallback=%d(timeout=%d error=%d) failed=%d rollbacks=%d emitted=%d\n",
		st.AcceptRate(), st.TokensPerStep(),
		st.SuccessSteps, st.FallbackSteps, st.TimeoutFallbacks, st.ErrorFallbacks,
		st.FailedSteps, st.Rollbacks, st.EmittedTokens)

	// Reproducibility: same seed + same input must produce identical output.
	first := svc.Submit(ctx, service.Request{ID: "repro-a", Prompt: prompt, MaxNewTokens: 12, Seed: 77})
	second := svc.Submit(ctx, service.Request{ID: "repro-b", Prompt: prompt, MaxNewTokens: 12, Seed: 77})
	if equal(first.Tokens, second.Tokens) {
		fmt.Println("reproducibility: identical output for identical (seed, input)")
	} else {
		fmt.Printf("reproducibility: MISMATCH %v vs %v\n", first.Tokens, second.Tokens)
	}
}

func equal(a, b []int) bool {
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
