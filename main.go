package main

import (
	"context"
	"fmt"

	"github.com/guoyyit/speculative-decoding-go/engine"
	"github.com/guoyyit/speculative-decoding-go/model"
	"github.com/guoyyit/speculative-decoding-go/server"
)

func main() {
	cfg := model.Config{VocabSize: 16, Temperature: 1.0, Noise: 0.15}
	draft := model.NewDraft(cfg)
	target := model.NewTarget(cfg)

	srv := server.New(draft, target, 4, engine.Config{
		MaxDraftTokens:  4,
		MaxTokens:       24,
		VerifyTimeoutMS: 100,
	})
	defer srv.Close()

	ctx := context.Background()
	prompt := []model.TokenID{1, 2, 3}

	const n = 4
	chans := make([]<-chan server.Response, n)
	for i := 0; i < n; i++ {
		chans[i] = srv.Submit(ctx, server.Request{
			ID:        int64(i + 1),
			Prompt:    prompt,
			MaxTokens: 16,
			Seed:      42,
		})
	}

	for i, ch := range chans {
		resp := <-ch
		status := "success"
		if resp.Err != nil {
			status = "failure: " + resp.Err.Error()
		}
		fmt.Printf("request %d [%s]: %v\n", i+1, status, resp.Tokens)
	}

	s := srv.Stats().Snapshot()
	fmt.Println("--- diagnostics ---")
	fmt.Printf("speculative steps      : %d\n", s.Steps)
	fmt.Printf("acceptance rate        : %.3f\n", s.AcceptanceRate)
	fmt.Printf("avg tokens per step    : %.3f\n", s.AvgTokensPerStep)
	fmt.Printf("rollbacks              : %d\n", s.Rollbacks)
	fmt.Printf("fallbacks              : %d %v\n", s.FallbackCount, s.FallbackReasons)
	fmt.Printf("failures               : %d\n", s.FailureCount)
}
