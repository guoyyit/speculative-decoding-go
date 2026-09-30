package model

import (
	"context"
	"math"
	"time"
)

// TokenID identifies a token in the vocabulary.
type TokenID int

// Distribution is a probability mass function over the whole vocabulary.
type Distribution []float64

// Proposer is the draft model: it autoregressively proposes up to k tokens.
type Proposer interface {
	// Propose returns the draft distribution at every position plus the draft
	// tokens sampled from them. draw supplies the request-local random stream
	// used to sample draft tokens; the distributions themselves are
	// deterministic functions of the context (the simulated "weights").
	Propose(ctx context.Context, prefix []TokenID, k int, draw func() float64) (tokens []TokenID, dists []Distribution, err error)
	Name() string
	VocabSize() int
}

// Verifier is the target model: it verifies a whole candidate sequence in a
// single batched forward pass and returns the target distribution at every
// candidate position.
type Verifier interface {
	// Verify returns len(candidates)+1 distributions: dists[i] is
	// P_target(. | prefix + candidates[:i]) for i in 0..len(candidates).
	Verify(ctx context.Context, prefix []TokenID, candidates []TokenID) (dists []Distribution, err error)
	// Next is a single unbatched target step: P_target(. | prefix).
	Next(ctx context.Context, prefix []TokenID) (Distribution, error)
	Name() string
	VocabSize() int
}

// Config describes a deterministic simulated model.
type Config struct {
	VocabSize   int
	Seed        int64
	Temperature float64
	// Noise is how far the draft logits deviate from the target logits.
	// It is only used by the draft model; 0 makes the draft distribution
	// identical to the target distribution.
	Noise float64
	// Latency is the simulated wall-clock cost of one forward pass.
	LatencyMS int
}

// DraftModel is a deterministic local simulation of a draft model.
type DraftModel struct {
	cfg Config
}

// TargetModel is a deterministic local simulation of the target model.
type TargetModel struct {
	cfg Config
}

// NewDraft constructs an empty draft model skeleton.
func NewDraft(cfg Config) *DraftModel {
	cfg = normalizeConfig(cfg)
	return &DraftModel{cfg: cfg}
}

// NewTarget constructs an empty target model skeleton.
func NewTarget(cfg Config) *TargetModel {
	cfg = normalizeConfig(cfg)
	return &TargetModel{cfg: cfg}
}

func normalizeConfig(cfg Config) Config {
	if cfg.VocabSize <= 0 {
		cfg.VocabSize = DefaultVocabSize
	}
	if cfg.Temperature <= 0 {
		cfg.Temperature = 1
	}
	return cfg
}

const (
	// DefaultVocabSize is the vocabulary size used when unset.
	DefaultVocabSize = 16

	// saltBase is shared by draft and target: at noise 0 the draft produces
	// exactly the target distribution. saltNoise perturbs only the draft.
	saltBase  uint64 = 0x7A46E7
	saltNoise uint64 = 0xD4AFF
)

func (m *DraftModel) Name() string  { return "draft-sim" }
func (m *TargetModel) Name() string { return "target-sim" }

func (m *DraftModel) VocabSize() int  { return m.cfg.VocabSize }
func (m *TargetModel) VocabSize() int { return m.cfg.VocabSize }

// Propose proposes up to k draft tokens autoregressively. The draft token at
// every position is sampled deterministically from the draft distribution
// using a context-derived RNG, so the same prefix always yields the same
// proposal. It also returns the draft distribution at each position.
func (m *DraftModel) Propose(ctx context.Context, prefix []TokenID, k int, draw func() float64) ([]TokenID, []Distribution, error) {
	if k <= 0 {
		return nil, nil, nil
	}
	if err := wait(ctx, m.cfg.LatencyMS); err != nil {
		return nil, nil, err
	}
	tokens := make([]TokenID, 0, k)
	dists := make([]Distribution, 0, k)
	contextTokens := append([]TokenID(nil), prefix...)
	for i := 0; i < k; i++ {
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		default:
		}
		d := m.distribution(contextTokens)
		tok := Sample(d, draw())
		tokens = append(tokens, tok)
		dists = append(dists, d)
		contextTokens = append(contextTokens, tok)
	}
	return tokens, dists, nil
}

// Verify runs one batched forward pass over the whole candidate span and
// returns the target distribution at every candidate position i, i.e.
// P_target(. | prefix + candidates[:i]).
func (m *TargetModel) Verify(ctx context.Context, prefix []TokenID, candidates []TokenID) ([]Distribution, error) {
	if len(candidates) == 0 {
		return nil, nil
	}
	if err := wait(ctx, m.cfg.LatencyMS); err != nil {
		return nil, err
	}
	dists := make([]Distribution, len(candidates)+1)
	contextTokens := append([]TokenID(nil), prefix...)
	for i := 0; i <= len(candidates); i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		dists[i] = m.distribution(contextTokens)
		if i == len(candidates) {
			break
		}
		contextTokens = append(contextTokens, candidates[i])
	}
	return dists, nil
}

// Next is the unbatched single-token target step used by non-speculative
// generation and by tests: P_target(. | prefix).
func (m *TargetModel) Next(ctx context.Context, prefix []TokenID) (Distribution, error) {
	if err := wait(ctx, m.cfg.LatencyMS); err != nil {
		return nil, err
	}
	return m.distribution(prefix), nil
}

func (m *DraftModel) distribution(contextTokens []TokenID) Distribution {
	return contextDistribution(m.cfg, contextTokens, m.cfg.Noise)
}

func (m *TargetModel) distribution(contextTokens []TokenID) Distribution {
	return contextDistribution(m.cfg, contextTokens, 0)
}

func contextDistribution(cfg Config, contextTokens []TokenID, noise float64) Distribution {
	base := contextSeed(cfg.Seed, saltBase, contextTokens)
	logits := make([]float64, cfg.VocabSize)
	maxLogit := math.Inf(-1)
	for v := 0; v < cfg.VocabSize; v++ {
		z := unitRNG(splitmix64(base + uint64(v)))
		logit := (z - 0.5) * 2 / cfg.Temperature
		if noise != 0 {
			n := unitRNG(splitmix64(base ^ uint64(v) ^ saltNoise))
			logit += (n - 0.5) * 2 * noise * 2 / cfg.Temperature
		}
		logits[v] = logit
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	dist := make(Distribution, cfg.VocabSize)
	sum := 0.0
	for v := range logits {
		p := math.Exp(logits[v] - maxLogit)
		dist[v] = p
		sum += p
	}
	for v := range dist {
		dist[v] /= sum
	}
	return dist
}

func wait(ctx context.Context, latencyMS int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if latencyMS <= 0 {
		return nil
	}
	timer := time.NewTimer(time.Duration(latencyMS) * time.Millisecond)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// Sample draws one token index from a normalized distribution with r in [0,1).
func Sample(d Distribution, r float64) TokenID {
	cum := 0.0
	for v, p := range d {
		cum += p
		if r < cum {
			return TokenID(v)
		}
	}
	return TokenID(len(d) - 1)
}

func contextSeed(seed int64, salt uint64, contextTokens []TokenID) uint64 {
	state := uint64(seed) ^ salt
	for _, t := range contextTokens {
		state = splitmix64(state ^ (uint64(t) + 0x9E3779B97F4A7C15))
	}
	return state
}

func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

func unitRNG(x uint64) float64 {
	return float64(x>>11) / float64(uint64(1)<<53)
}
