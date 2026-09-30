package sampling

import "testing"

func TestSeededDeterminism(t *testing.T) {
	a := New(42)
	b := New(42)
	for i := 0; i < 100; i++ {
		if a.Float64() != b.Float64() {
			t.Fatal("same seed streams diverged")
		}
	}
}

func TestSampleFrequencies(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.3, 0.4}
	s := New(7)
	counts := make([]int, len(probs))
	const n = 40000
	for i := 0; i < n; i++ {
		counts[s.Sample(probs)]++
	}
	for i, want := range probs {
		got := float64(counts[i]) / n
		if got < want-0.02 || got > want+0.02 {
			t.Fatalf("bucket %d: %.3f not within range of %.3f", i, got, want)
		}
	}
}

func TestSampleDeterministic(t *testing.T) {
	probs := []float64{0.25, 0.25, 0.25, 0.25}
	a := New(99)
	b := New(99)
	for i := 0; i < 50; i++ {
		if a.Sample(probs) != b.Sample(probs) {
			t.Fatal("Sample not reproducible with fixed seed")
		}
	}
}
