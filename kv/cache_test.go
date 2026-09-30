package kv

import "testing"

func TestAppendAndCopy(t *testing.T) {
	c := New(10)
	c.Append(1, 2, 3)
	got := c.Tokens()
	got[0] = 99
	if c.Tokens()[0] != 1 {
		t.Fatalf("Tokens() must return a copy; cache mutated to %v", c.Tokens())
	}
	if c.Len() != 3 {
		t.Fatalf("Len = %d, want 3", c.Len())
	}
}

func TestTruncateRollback(t *testing.T) {
	c := New(10)
	c.Append(1, 2, 3, 4, 5)
	if err := c.Truncate(3); err != nil {
		t.Fatalf("Truncate: %v", err)
	}
	got := c.Tokens()
	if len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Fatalf("after rollback got %v, want [1 2 3]", got)
	}
	if err := c.Truncate(4); err == nil {
		t.Fatal("Truncate beyond length must error")
	}
	if err := c.Truncate(-1); err == nil {
		t.Fatal("negative Truncate must error")
	}
}

func TestBoundedEviction(t *testing.T) {
	c := New(4)
	c.Append(1, 2, 3, 4, 5, 6)
	got := c.Tokens()
	if len(got) != 4 {
		t.Fatalf("Len = %d, want bounded 4", len(got))
	}
	want := []int{3, 4, 5, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("evicted suffix = %v, want %v", got, want)
		}
	}
}

func TestEvictThenTruncateBounded(t *testing.T) {
	c := New(4)
	c.Append(1, 2, 3, 4)
	c.Append(5, 6) // evicts 1,2 -> [3 4 5 6]
	if err := c.Truncate(3); err != nil {
		t.Fatalf("Truncate: %v", err)
	}
	got := c.Tokens()
	if len(got) != 3 || got[2] != 5 {
		t.Fatalf("got %v, want [3 4 5]", got)
	}
}
