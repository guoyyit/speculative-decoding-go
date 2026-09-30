// Package sampling implements token sampling and the speculative-decoding
// acceptance rule.
package sampling

import (
	"context"
	"math"
	"math/rand/v2"

	"github.com/guoyyit/speculative-decoding-go/model"
)

// Source produces reproducible random bits for one request stream.
type Source interface {
	Float64() float64
}

// pcgSource is a per-request deterministic random stream. Each request owns
// an independent source, so concurrent requests never share random state and
// the same seed always reproduces the same sequence of draws.
type pcgSource struct {
	rng *rand.Rand
}

// NewSource builds a deterministic stream from a request seed.
func NewSource(seed uint64) Source {
	return &pcgSource{rng: rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))}
}

// draftStreamTag/targetStreamTag segregate randomness by role.
const (
	draftStreamTag  uint64 = 0xD4AFF
	targetStreamTag uint64 = 0x7A46E7
)

// NewStreamPair derives two independent deterministic streams from one
// request seed:

//   - draft: used only to sample draft proposal tokens;
//   - target: used for acceptance rolls, residual corrections, bonus tokens
//     and non-speculative fallback sampling.
//
// Because plain target-only generation consumes only the target stream, a
// speculative round can never shift the alignment of target samples: a
// timed-out verification falls back to a target draw that is identical to the
// target-only draw at the same position.
func NewStreamPair(seed uint64) (draft Source, target Source) {
	return NewSource(seed ^ draftStreamTag), NewSource(seed ^ targetStreamTag)
}

func (s *pcgSource) Float64() float64 { return s.rng.Float64() }

// AcceptProb is the speculative-sampling acceptance probability min(1, p/q),
// where p is the target probability and q the draft probability of the same
// token. There is no fixed threshold: the decision is driven entirely by the
// target model's probability mass.
func AcceptProb(p, q float64) float64 {
	switch {
	case q <= 0:
		// Zero-probability draft tokens are only reachable through the
		// residual path; treat them as certainly acceptable when p is also 0.
		if p <= 0 {
			return 1
		}
		return 1
	case p >= q:
		return 1
	default:
		return p / q
	}
}

// Sample draws one token from a normalized distribution.
func Sample(d model.Distribution, src Source) model.TokenID {
	return model.Sample(d, src.Float64())
}

// Residual returns the adjusted distribution p' = norm(max(0, p - q)) used to
// resample the correction token at the first rejected position.
func Residual(p, q model.Distribution) model.Distribution {
	res := make(model.Distribution, len(p))
	sum := 0.0
	for i := range p {
		v := p[i] - q[i]
		if v > 0 {
			res[i] = v
			sum += v
		}
	}
	if sum <= 0 || math.IsNaN(sum) {
		// Defensive fallback: target itself when distributions coincide.
		copy(res, p)
		return res
	}
	for i := range res {
		res[i] /= sum
	}
	return res
}

// StepOutcome is the result of one speculative proposal/verification step.
type StepOutcome struct {
	// Proposed is the draft token sequence.
	Proposed []model.TokenID
	// DraftDist[i] is q_i, the draft distribution at position i.
	DraftDist []model.Distribution
	// TargetDist[i] is p_i, the target distribution at position i, with
	// length len(Proposed)+1.
	TargetDist []model.Distribution
	// Accepted is the maximal accepted draft prefix.
	Accepted []model.TokenID
	// RejectedAt is the first rejected position, or -1 when every draft
	// token was accepted.
	RejectedAt int
	// Correction is the resampled token at RejectedAt. Valid only when
	// RejectedAt >= 0.
	Correction model.TokenID
	// Bonus is the extra target-sampled token produced when the whole draft
	// sequence is accepted.
	Bonus    model.TokenID
	HasBonus bool
}

// Step performs one speculative step: the draft proposes k tokens, the target
// verifies the whole span in one batched pass, accepted tokens form a
// contiguous prefix, and the first rejected position is resampled from the
// residual distribution. Rejected draft tokens never leave this function as
// accepted output.
func Step(ctx context.Context, proposer model.Proposer, verifier model.Verifier,
	prefix []model.TokenID, k int, draftSrc, targetSrc Source) (*StepOutcome, error) {

	proposed, draftDists, err := proposer.Propose(ctx, prefix, k, draftSrc.Float64)
	if err != nil {
		return nil, err
	}
	if len(proposed) == 0 {
		return &StepOutcome{RejectedAt: -1}, nil
	}
	targetDists, err := verifier.Verify(ctx, prefix, proposed)
	if err != nil {
		return nil, err
	}

	out := &StepOutcome{
		Proposed:   proposed,
		DraftDist:  draftDists,
		TargetDist: targetDists,
		RejectedAt: -1,
	}

	for i, tok := range proposed {
		p := targetDists[i][tok]
		q := draftDists[i][tok]
		if targetSrc.Float64() >= AcceptProb(p, q) {
			out.RejectedAt = i
			out.Accepted = append(out.Accepted, proposed[:i]...)
			residual := Residual(targetDists[i], draftDists[i])
			out.Correction = Sample(residual, targetSrc)
			return out, nil
		}
		out.Accepted = append(out.Accepted, tok)
	}

	// Every draft token was accepted: sample one bonus token from the final
	// target distribution produced by the same batched forward pass.
	out.Bonus = Sample(targetDists[len(proposed)], targetSrc)
	out.HasBonus = true
	return out, nil
}

// FallbackStep performs a single non-speculative target step. It is used when
// speculative verification times out or is unavailable.
func FallbackStep(ctx context.Context, verifier model.Verifier,
	prefix []model.TokenID, src Source) (model.TokenID, model.Distribution, error) {

	dist, err := verifier.Next(ctx, prefix)
	if err != nil {
		return 0, nil, err
	}
	return Sample(dist, src), dist, nil
}
