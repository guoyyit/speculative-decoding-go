package cache

import (
	"testing"

	"github.com/guoyyit/speculative-decoding-go/model"
)

func TestRollbackDiscardsRejectedCandidates(t *testing.T) {
	c := New(8)
	c.Append(1, 2)
	c.Commit()

	snap := c.Snapshot()
	c.Append(3, 4, 5) // tentative speculative span

	if got := c.Len(); got != 2 {
		t.Fatalf("Len before commit = %d, want 2", got)
	}

	c.Rollback(snap)
	got := c.Tokens()
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("after rollback tokens = %v, want [1 2]", got)
	}

	// A later generation must only see the accepted prefix.
	c.Append(6)
	c.Commit()
	got = c.Tokens()
	if len(got) != 3 || got[2] != 6 {
		t.Fatalf("rejected candidate influence lingered: %v", got)
	}
}

func TestCommitAndIsolation(t *testing.T) {
	c := New(4)
	c.Append(model.TokenID(7))
	c.Commit()
	toks := c.Tokens()
	toks[0] = 99 // mutating returned slice must not affect cache
	if c.Tokens()[0] != 7 {
		t.Fatal("Tokens() must return a defensive copy")
	}

	snap := c.Snapshot()
	c.Append(8, 9)
	c.Commit()
	c.Rollback(snap) // rollback cannot restore tokens committed earlier than snap
	if c.Len() != 1 {
		t.Fatalf("rollback past commit boundary moved committed state: %d", c.Len())
	}
}

func TestResetReleasesState(t *testing.T) {
	c := New(4)
	c.Append(1, 2, 3, 4)
	c.Commit()
	c.Reset()
	if c.Len() != 0 || len(c.Tokens()) != 0 {
		t.Fatal("Reset must empty the cache")
	}
}
