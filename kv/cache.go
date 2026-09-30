package kv

import "errors"

// ErrBadLength is returned by Truncate when the target length is invalid.
var ErrBadLength = errors.New("kv: truncate length out of range")

// Cache stores the accepted token prefix of one request and supports
// truncation to an accept point.
//
// Memory is bounded: when the stored prefix exceeds capacity the oldest
// tokens are evicted. The simulated models in this repository depend only on
// a bounded suffix of the history, so eviction never changes their output
// distributions (the model context window must not exceed capacity).
type Cache struct {
	capacity int
	tokens   []int
}

// New creates a bounded cache. A non-positive capacity means unbounded.
func New(capacity int) *Cache {
	return &Cache{capacity: capacity, tokens: make([]int, 0, 64)}
}

func (c *Cache) Len() int { return len(c.tokens) }

// Tokens returns a copy of the stored prefix; callers cannot mutate cache
// state through it.
func (c *Cache) Tokens() []int {
	out := make([]int, len(c.tokens))
	copy(out, c.tokens)
	return out
}

// Append adds tokens, evicting the oldest entries when over capacity.
func (c *Cache) Append(tokens ...int) {
	c.tokens = append(c.tokens, tokens...)
	if c.capacity > 0 && len(c.tokens) > c.capacity {
		drop := len(c.tokens) - c.capacity
		c.tokens = append(c.tokens[:0], c.tokens[drop:]...)
	}
}

// Truncate discards every token after position length, restoring the cache
// to the state of the accepted prefix. It never mutates retained tokens.
func (c *Cache) Truncate(length int) error {
	if length < 0 || length > len(c.tokens) {
		return ErrBadLength
	}
	c.tokens = c.tokens[:length]
	return nil
}

// Reset clears the cache and stores tokens, applying the same bounded
// eviction rules as Append. It is used to restore the pre-verification
// history plus an accepted prefix after a rejected speculative batch.
func (c *Cache) Reset(tokens []int) {
	c.tokens = c.tokens[:0]
	c.Append(tokens...)
}
