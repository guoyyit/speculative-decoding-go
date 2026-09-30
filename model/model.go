package model

import (
	"context"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"time"
)

// RNG is the minimal randomness surface consumed by the simulated models.
// Both math/rand.Rand and a seeded sampling.Sampler satisfy it.
type RNG interface {
	Float64() float64
}

// ContextWindow is the number of trailing history tokens that influence the
// simulated distributions. Caches with capacity >= ContextWindow never evict
// tokens that a model still depends on.
const ContextWindow = 8

// Config configures the deterministic simulated draft/target pair.
type Config struct {
	VocabSize       int
	Seed            uint64
	DraftNoise      float64
	EOSBias         float64
	TargetLatencyMS int
	DraftLatencyMS  int
	VerifyLatencyMS int
}

// Target is the deterministic simulated target model.
type Target struct {
	cfg Config
}

// Draft is the deterministic simulated draft model.
type Draft struct {
	cfg Config
}

// Proposal is a draft proposal: candidate tokens and the draft distribution
// that produced each candidate.
type Proposal struct {
	Tokens []int
	Q      [][]float64
}

func normalizeConfig(cfg Config) Config {
	if cfg.VocabSize <= 1 {
		cfg.VocabSize = 32
	}
	if cfg.DraftNoise < 0 {
		cfg.DraftNoise = 0
	}
	return cfg
}

// NewTarget creates a deterministic simulated target model.
func NewTarget(cfg Config) *Target { return &Target{cfg: normalizeConfig(cfg)} }

// NewDraft creates the matching draft model. With DraftNoise == 0 the draft
// distribution equals the target distribution at every context.
func NewDraft(cfg Config) *Draft { return &Draft{cfg: normalizeConfig(cfg)} }

func (t *Target) Size() int { return t.cfg.VocabSize }

// EOS returns the end-of-sequence token id (token 0).
func (t *Target) EOS() int { return 0 }

// Dist returns the full target categorical distribution p(. | history).
// The result depends only on the trailing ContextWindow tokens and Config,
// so identical histories always yield identical distributions.
func (t *Target) Dist(history []int) []float64 {
	return softmax(t.baseLogits(history))
}

// baseLogits builds the target logit vector from a hash of the context.
func (t *Target) baseLogits(history []int) []float64 {
	rng := t.contextRNG(history, 0x746172676574) // "target"
	logits := make([]float64, t.cfg.VocabSize)
	for i := range logits {
		// Spread logits moderately so distributions are non-trivial.
		logits[i] = 1.5 * rng.NormFloat64()
	}
	// Token 0 is EOS; bias it so long generations are possible.
	logits[0] += t.cfg.EOSBias
	return logits
}

func (t *Target) contextRNG(history []int, tag uint64) *rand.Rand {
	var buf [8]byte
	start := 0
	if len(history) > ContextWindow {
		start = len(history) - ContextWindow
	}
	// Hash only the visible suffix with a fresh FNV, so tokens outside the
	// context window cannot influence the distribution.
	h := fnv.New64a()
	putTag := func(v uint64) {
		for i := 0; i < 8; i++ {
			buf[i] = byte(v >> (8 * i))
		}
		_, _ = h.Write(buf[:])
	}
	putTag(t.cfg.Seed)
	putTag(tag)
	for _, tok := range history[start:] {
		putTag(uint64(tok) + 1)
	}
	lo := h.Sum64()
	putTag(lo ^ 0x9e3779b97f4a7c15)
	hi := h.Sum64()
	return rand.New(rand.NewPCG(lo, hi))
}

// Verify emulates one parallel target forward pass over the whole candidate
// sequence: it returns distributions at every candidate position plus the
// bonus position after the last candidate (len(candidates)+1 rows).
func (t *Target) Verify(ctx context.Context, history []int, candidates []int) ([][]float64, error) {
	latency := t.cfg.VerifyLatencyMS
	if latency <= 0 {
		latency = t.cfg.TargetLatencyMS
	}
	if err := sleepInterruptible(ctx, latency); err != nil {
		return nil, err
	}
	out := make([][]float64, 0, len(candidates)+1)
	cur := append([]int(nil), history...)
	for i := 0; i <= len(candidates); i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		out = append(out, t.Dist(cur))
		if i < len(candidates) {
			cur = append(cur, candidates[i])
		}
	}
	return out, nil
}

// Propose draws up to k draft tokens autoregressively. If EOS is sampled the
// proposal ends early; otherwise it returns k tokens and k distributions.
func (d *Draft) Propose(ctx context.Context, history []int, k int, rng RNG) (*Proposal, error) {
	latency := d.cfg.DraftLatencyMS
	if latency <= 0 {
		latency = d.cfg.TargetLatencyMS
	}
	if err := sleepInterruptible(ctx, latency); err != nil {
		return nil, err
	}
	if k < 0 {
		k = 0
	}
	cur := append([]int(nil), history...)
	tokens := make([]int, 0, k)
	q := make([][]float64, 0, k)
	for i := 0; i < k; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		dist := d.dist(cur)
		q = append(q, dist)
		tok := categorical(dist, rng.Float64())
		tokens = append(tokens, tok)
		cur = append(cur, tok)
		if tok == 0 { // EOS ends the sequence; do not propose past it.
			break
		}
	}
	return &Proposal{Tokens: tokens, Q: q}, nil
}

// dist returns the draft distribution at a context: target logits plus
// deterministic gaussian perturbation scaled by DraftNoise.
func (d *Draft) dist(history []int) []float64 {
	target := NewTarget(d.cfg)
	logits := target.baseLogits(history)
	if d.cfg.DraftNoise > 0 {
		rng := target.contextRNG(history, 0x6472616674) // "draft"
		for i := range logits {
			logits[i] += d.cfg.DraftNoise * rng.NormFloat64()
		}
	}
	return softmax(logits)
}

func categorical(probs []float64, u float64) int {
	var cumulative float64
	last := -1
	for i, p := range probs {
		if p > 0 {
			last = i
		}
		cumulative += p
		if u < cumulative {
			return i
		}
	}
	return last
}

func softmax(logits []float64) []float64 {
	max := math.Inf(-1)
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// sleepInterruptible simulates model latency but returns promptly when ctx
// is done, in slices of at most 10ms.
func sleepInterruptible(ctx context.Context, latencyMS int) error {
	if latencyMS <= 0 {
		return ctx.Err()
	}
	deadline := time.Now().Add(time.Duration(latencyMS) * time.Millisecond)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return nil
		}
		step := remaining
		if step > 10*time.Millisecond {
			step = 10 * time.Millisecond
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(step):
		}
	}
}
