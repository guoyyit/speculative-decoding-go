package sampling

import "math/rand/v2"

// Sampler draws categorical samples from a fixed, seeded RNG.
type Sampler struct {
	rng *rand.Rand
}

// New creates a sampler whose full output stream is determined by seed.
// Two samplers with the same seed produce identical draw sequences.
func New(seed uint64) *Sampler {
	// Split the single seed into two independent 64-bit words for PCG so
	// adjacent seeds do not produce correlated streams.
	s0 := splitmix64(seed ^ 0x9e3779b97f4a7c15)
	s1 := splitmix64(seed ^ 0xbf58476d1ce4e5b9)
	return &Sampler{rng: rand.New(rand.NewPCG(s0, s1))}
}

// Float64 exposes the underlying uniform stream for acceptance draws.
func (s *Sampler) Float64() float64 { return s.rng.Float64() }

// Sample returns an index sampled proportionally to probs.
// probs must be non-empty and sum to 1 (up to floating point error).
// The method uses one uniform draw exactly.
func (s *Sampler) Sample(probs []float64) int {
	u := s.rng.Float64()
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

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
