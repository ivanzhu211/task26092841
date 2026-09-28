package prosa

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// stressHooks tracks residency invariants inside the hooks: an inference
// may only run while its model is loaded and not being unloaded.
type stressHooks struct {
	mu        sync.Mutex
	loaded    map[string]bool
	inferOK   int
	infered   map[int]bool // payloads (task indexes) that entered inference
	violation atomic.Value // first violation, string
}

func newStressHooks() *stressHooks {
	return &stressHooks{loaded: make(map[string]bool), infered: make(map[int]bool)}
}

func (h *stressHooks) fail(format string, args ...any) {
	h.violation.CompareAndSwap(nil, fmt.Sprintf(format, args...))
}

func (h *stressHooks) hooks() Hooks {
	return Hooks{
		Load: func(_ context.Context, model string) error {
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.loaded[model] {
				h.fail("model %s loaded twice", model)
			}
			h.loaded[model] = true
			return nil
		},
		Infer: func(_ context.Context, model string, payload any) error {
			h.mu.Lock()
			defer h.mu.Unlock()
			if !h.loaded[model] {
				h.fail("infer on non-resident model %s", model)
			}
			h.inferOK++
			h.infered[payload.(int)] = true
			return nil
		},
		Unload: func(_ context.Context, model string) error {
			h.mu.Lock()
			defer h.mu.Unlock()
			if !h.loaded[model] {
				h.fail("model %s unloaded while not resident", model)
			}
			h.loaded[model] = false
			return nil
		},
	}
}

func (h *stressHooks) stats() (inferOK int, infered map[int]bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make(map[int]bool, len(h.infered))
	for k, v := range h.infered {
		out[k] = v
	}
	return h.inferOK, out
}

// TestStressMixed randomizes submits, cancels, offlines and snapshots
// across many goroutines, checks invariants throughout, and verifies the
// terminal-state bookkeeping at the end. Repeated rounds must be stable.
func TestStressMixed(t *testing.T) {
	for round := 0; round < 4; round++ {
		t.Run(fmt.Sprintf("round-%d", round), func(t *testing.T) {
			runStressRound(t, int64(round+1))
		})
	}
}

func runStressRound(t *testing.T, seed int64) {
	const (
		nModels   = 20
		nTasks    = 20000
		maxConc   = 8
		maxModels = 4
		memLimit  = 10
	)
	h := newStressHooks()
	cfg := Config{MaxConcurrent: maxConc, MaxModels: maxModels, MemoryLimit: memLimit}
	s, err := New(cfg, h.hooks())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(seed))
	mems := make([]int64, nModels)
	for i := 0; i < nModels; i++ {
		mems[i] = int64(1 + rng.Intn(3)) // 1..3
		name := fmt.Sprintf("m%d", i)
		if err := s.RegisterModel(name, mems[i]); err != nil {
			t.Fatal(err)
		}
	}

	// Invariant sampler: every snapshot must be consistent with the limits.
	stop := make(chan struct{})
	var samplerWG sync.WaitGroup
	samplerWG.Add(1)
	go func() {
		defer samplerWG.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			snap := s.Snapshot()
			if snap.Running > maxConc {
				h.fail("running = %d > %d", snap.Running, maxConc)
			}
			if len(snap.Models) > maxModels {
				h.fail("resident = %d > %d", len(snap.Models), maxModels)
			}
			var mem int64
			for _, ms := range snap.Models {
				mem += ms.DeclaredMemory
				if ms.Running < 0 {
					h.fail("model %s running < 0", ms.Name)
				}
			}
			if mem > memLimit {
				h.fail("resident memory = %d > %d", mem, memLimit)
			}
			for p, n := range snap.QueuedByPriority {
				if n < 0 {
					h.fail("queued[%d] = %d < 0", p, n)
				}
			}
		}
	}()

	// Submitters.
	ids := make([]uint64, nTasks)
	accepted := make([]atomic.Bool, nTasks)
	var submitWG sync.WaitGroup
	for w := 0; w < 8; w++ {
		submitWG.Add(1)
		go func(w int) {
			defer submitWG.Done()
			r := rand.New(rand.NewSource(seed*100 + int64(w)))
			for i := w; i < nTasks; i += 8 {
				model := fmt.Sprintf("m%d", r.Intn(nModels))
				id, err := s.Submit(model, Priority(r.Intn(2)), i)
				if err == nil {
					ids[i] = id
					accepted[i].Store(true)
				}
			}
		}(w)
	}

	// Cancelers hammer random indexes while submits are in flight.
	var cancelWG sync.WaitGroup
	cancelRes := make([]atomic.Bool, nTasks)
	for w := 0; w < 4; w++ {
		cancelWG.Add(1)
		go func(w int) {
			defer cancelWG.Done()
			r := rand.New(rand.NewSource(seed*1000 + int64(w)))
			for k := 0; k < nTasks/2; k++ {
				i := r.Intn(nTasks)
				if accepted[i].Load() && s.Cancel(ids[i]) {
					cancelRes[i].Store(true)
				}
			}
		}(w)
	}

	// Offliner takes a few random models offline mid-flight.
	offlineWG := sync.WaitGroup{}
	offlineWG.Add(1)
	go func() {
		defer offlineWG.Done()
		r := rand.New(rand.NewSource(seed * 7))
		for _, mi := range r.Perm(nModels)[:3] {
			time.Sleep(time.Duration(r.Intn(5)) * time.Millisecond)
			_ = s.OfflineModel(fmt.Sprintf("m%d", mi))
		}
	}()

	submitWG.Wait()
	cancelWG.Wait()
	offlineWG.Wait()

	// Wait for quiescence: nothing queued, running, or loading.
	waitFor(t, "scheduler to drain", func() bool {
		snap := s.Snapshot()
		if snap.Running != 0 {
			return false
		}
		queued := 0
		for _, n := range snap.QueuedByPriority {
			queued += n
		}
		if queued != 0 {
			return false
		}
		for _, ms := range snap.Models {
			if ms.State == ModelLoading {
				return false
			}
		}
		return true
	})
	close(stop)
	samplerWG.Wait()

	if v := h.violation.Load(); v != nil {
		t.Fatalf("invariant violation: %s", v.(string))
	}

	// Terminal-state bookkeeping.
	inferOK, infered := h.stats()
	completed := 0
	for i := 0; i < nTasks; i++ {
		if !accepted[i].Load() {
			continue
		}
		info, ok := s.Task(ids[i])
		if !ok {
			t.Fatalf("task %d missing", ids[i])
		}
		switch info.State {
		case TaskCompleted:
			completed++
			if !infered[i] {
				t.Fatalf("task %d completed but never inferred", ids[i])
			}
		case TaskCanceled:
			if !cancelRes[i].Load() {
				t.Fatalf("task %d canceled but no Cancel reported it", ids[i])
			}
			if infered[i] {
				t.Fatalf("canceled task %d entered inference", ids[i])
			}
		case TaskFailed:
			// load failure is not exercised here; offline failures are fine
		default:
			t.Fatalf("task %d in non-terminal state %v after drain", ids[i], info.State)
		}
		if info.EndedAt.IsZero() {
			t.Fatalf("task %d missing end time", ids[i])
		}
	}
	if completed != inferOK {
		t.Fatalf("completed = %d, successful infer calls = %d", completed, inferOK)
	}
}

// TestPerf100k: with instant load/infer, 100k tasks across 50 models must
// all complete within 10 seconds.
func TestPerf100k(t *testing.T) {
	const (
		nTasks  = 100000
		nModels = 50
	)
	h := newStressHooks()
	cfg := Config{MaxConcurrent: 8, MaxModels: 4, MemoryLimit: 4}
	s, err := New(cfg, h.hooks())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < nModels; i++ {
		if err := s.RegisterModel(fmt.Sprintf("m%d", i), 1); err != nil {
			t.Fatal(err)
		}
	}
	start := time.Now()
	ids := make([]uint64, nTasks)
	for i := 0; i < nTasks; i++ {
		id, err := s.Submit(fmt.Sprintf("m%d", i%nModels), Priority(i%2), i)
		if err != nil {
			t.Fatal(err)
		}
		ids[i] = id
	}
	waitFor(t, "all 100k tasks to complete", func() bool {
		snap := s.Snapshot()
		queued := 0
		for _, n := range snap.QueuedByPriority {
			queued += n
		}
		return queued == 0 && snap.Running == 0
	})
	elapsed := time.Since(start)
	for _, id := range ids {
		if st := taskState(s, id); st != TaskCompleted {
			t.Fatalf("task %d state = %v, want completed", id, st)
		}
	}
	inferOK, _ := h.stats()
	if inferOK != nTasks {
		t.Fatalf("infer calls = %d, want %d", inferOK, nTasks)
	}
	if v := h.violation.Load(); v != nil {
		t.Fatalf("invariant violation: %s", v.(string))
	}
	if elapsed > 10*time.Second {
		t.Fatalf("100k tasks took %v, want < 10s", elapsed)
	}
	t.Logf("100k tasks completed in %v", elapsed)
}
