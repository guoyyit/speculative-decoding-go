package service

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

var errServiceClosed = errors.New("service: closed")

type verifyCall struct {
	ctx        context.Context
	history    []int
	candidates []int
	result     chan callResult
}

type callResult struct {
	rows [][]float64
	err  error
}

type panicErr struct{ v any }

func (e panicErr) Error() string { return fmt.Sprintf("service: internal panic: %v", e.v) }

func panicError(v any) error { return panicErr{v: v} }

// verifyTarget is the model surface the batcher needs. In production it is
// *model.Target; tests may supply a deterministic failing implementation to
// exercise non-timeout verification failures.
type verifyTarget interface {
	Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error)
}

// batcher groups concurrent verification calls into bounded batches. The
// grouping only controls scheduling: each call's rows are computed from its
// own history/candidates by the stateless target model, so batching cannot
// change any request's distribution.
type batcher struct {
	target verifyTarget
	max    int
	window time.Duration

	inbox chan *verifyCall
	stop  chan struct{}
	wg    sync.WaitGroup
}

func newBatcher(target verifyTarget, max int, window time.Duration) *batcher {
	b := &batcher{
		target: target,
		max:    max,
		window: window,
		inbox:  make(chan *verifyCall, 64),
		stop:   make(chan struct{}),
	}
	b.wg.Add(1)
	go b.loop()
	return b
}

// Verify satisfies spec.Verifier.
func (b *batcher) Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error) {
	call := &verifyCall{
		ctx:        ctx,
		history:    history,
		candidates: candidates,
		result:     make(chan callResult, 1),
	}
	select {
	case b.inbox <- call:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-b.stop:
		return nil, errServiceClosed
	}
	select {
	case res := <-call.result:
		return res.rows, res.err
	case <-ctx.Done():
		// The batch worker may still deliver the (unused) result.
		return nil, ctx.Err()
	}
}

func (b *batcher) loop() {
	defer b.wg.Done()
	for {
		select {
		case <-b.stop:
			return
		case first := <-b.inbox:
			batch := b.gather(first)
			b.execute(batch)
		}
	}
}

// gather collects up to max calls or until the window elapses.
func (b *batcher) gather(first *verifyCall) []*verifyCall {
	batch := []*verifyCall{first}
	timer := time.NewTimer(b.window)
	defer timer.Stop()
	for len(batch) < b.max {
		select {
		case call := <-b.inbox:
			batch = append(batch, call)
		case <-timer.C:
			return batch
		case <-b.stop:
			return batch
		}
	}
	return batch
}

// execute computes one "parallel verification batch": all requests share the
// single batch latency window, but each gets rows derived solely from its
// own inputs. One request's cancellation does not affect any other.
func (b *batcher) execute(batch []*verifyCall) {
	// Determine the effective latency for this batch from the model config:
	// simulate one shared forward-pass wait, interruptible per request.
	var wg sync.WaitGroup
	for _, call := range batch {
		wg.Add(1)
		go func(call *verifyCall) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					call.result <- callResult{err: panicError(r)}
				}
			}()
			rows, err := b.target.Verify(call.ctx, call.history, call.candidates)
			select {
			case call.result <- callResult{rows: rows, err: err}:
			default:
				// Caller already gave up (e.g. canceled while queued);
				// drop the unused result instead of blocking the worker.
			}
		}(call)
	}
	wg.Wait()
}

func (b *batcher) close() {
	close(b.stop)
	b.wg.Wait()
}
