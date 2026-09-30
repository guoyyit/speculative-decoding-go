// Package cache holds per-request sequence/KV state with bounded rollback.
package cache

import "github.com/guoyyit/speculative-decoding-go/model"

// Snapshot is a restorable view of sequence cache state.
type Snapshot int

// Cache stores the committed token prefix of one request.
//
// A single backing slice is shared by committed and tentative tokens. A
// snapshot records a length, so rollback never retains stale arrays and the
// cache footprint is bounded by the longest sequence the request ever builds.
type Cache struct {
	tokens    []model.TokenID
	committed int
	vocabSize int
}

// New constructs an empty cache skeleton.
func New(vocabSize int) *Cache {
	return &Cache{vocabSize: vocabSize}
}

// Tokens returns the committed prefix.
func (c *Cache) Tokens() []model.TokenID {
	out := make([]model.TokenID, c.committed)
	copy(out, c.tokens[:c.committed])
	return out
}

// Append tentatively extends the sequence.
func (c *Cache) Append(tokens ...model.TokenID) {
	c.tokens = append(c.tokens, tokens...)
}

// Snapshot captures the current length/state.
func (c *Cache) Snapshot() Snapshot { return Snapshot(c.committed) }

// Rollback discards every token after the snapshot.
func (c *Cache) Rollback(s Snapshot) {
	n := int(s)
	if n < 0 {
		n = 0
	}
	if n > c.committed {
		n = c.committed
	}
	// Drop the tail entirely so rejected candidate state cannot linger.
	c.tokens = append([]model.TokenID(nil), c.tokens[:n]...)
	c.committed = n
}

// Commit keeps the tentatively appended prefix.
func (c *Cache) Commit() {
	c.committed = len(c.tokens)
}

// Len returns the committed sequence length.
func (c *Cache) Len() int { return c.committed }

// Reset clears all state and releases the backing array.
func (c *Cache) Reset() {
	c.tokens = nil
	c.committed = 0
}
