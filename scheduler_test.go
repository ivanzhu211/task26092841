package prosa

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func testConfig() Config {
	return Config{MaxConcurrent: 8, MaxModels: 4, MemoryLimit: 1 << 30}
}

// A task waiting for its model to load must not block later tasks whose
// model is already available.
func TestDispatchSkipsHeadWaitingForLoad(t *testing.T) {
	rec := newRecorder()
	s, err := New(testConfig(), rec.hooks())
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"slow", "fast"} {
		if err := s.RegisterModel(m, 1); err != nil {
			t.Fatal(err)
		}
	}
	slowGate := rec.gateLoad("slow")

	// Warm up "fast" so it is resident before the slow load begins.
	idWarm, err := s.Submit("fast", PriorityTranslator, "warmup")
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "warmup task to complete", func() bool {
		return taskState(s, idWarm) == TaskCompleted
	})

	idSlow, err := s.Submit("slow", PriorityReviewer, "slow-task")
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "slow model load to start", func() bool { return rec.count("load:slow") == 1 })

	idFast, err := s.Submit("fast", PriorityTranslator, "fast-task")
	if err != nil {
		t.Fatal(err)
	}
	// fast must complete even though slow is still loading.
	waitFor(t, "fast task to complete", func() bool {
		return taskState(s, idFast) == TaskCompleted
	})
	if got := taskState(s, idSlow); got != TaskPending {
		t.Fatalf("slow task state = %v, want pending while model loads", got)
	}
	close(slowGate)
	waitFor(t, "slow task to complete", func() bool {
		return taskState(s, idSlow) == TaskCompleted
	})
}

// With 4 resident models saturated by same-priority tasks, a task needing
// a 5th model must get the freed slot once the evicted model drains and the
// new model loads; the freed residency must not be stolen by another load.
func TestEvictionHandsFreedSlotToWaitingTask(t *testing.T) {
	rec := newRecorder()
	cfg := Config{MaxConcurrent: 8, MaxModels: 4, MemoryLimit: 4}
	s, err := New(cfg, rec.hooks())
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"m1", "m2", "m3", "m4", "m5", "m6"} {
		if err := s.RegisterModel(m, 1); err != nil {
			t.Fatal(err)
		}
	}
	// Occupy all 4 residency slots with one long-running task each.
	gates := map[string]chan struct{}{}
	holders := map[string]uint64{}
	for _, m := range []string{"m1", "m2", "m3", "m4"} {
		gates[m] = rec.gateInfer("hold-" + m)
		id, err := s.Submit(m, PriorityTranslator, "hold-"+m)
		if err != nil {
			t.Fatal(err)
		}
		holders[m] = id
	}
	waitFor(t, "4 resident models each running one task", func() bool {
		snap := s.Snapshot()
		return len(snap.Models) == 4 && snap.Running == 4
	})

	// Keep the resident models busy with a stream of same-priority tasks.
	busy := make([]uint64, 0, 16)
	for i := 0; i < 4; i++ {
		for _, m := range []string{"m1", "m2", "m3", "m4"} {
			id, err := s.Submit(m, PriorityTranslator, i)
			if err != nil {
				t.Fatal(err)
			}
			busy = append(busy, id)
		}
	}

	// The task needing the 5th model.
	id5, err := s.Submit("m5", PriorityTranslator, "need-m5")
	if err != nil {
		t.Fatal(err)
	}
	// A later task needing a 6th model must not steal the freed slot.
	id6, err := s.Submit("m6", PriorityTranslator, "need-m6")
	if err != nil {
		t.Fatal(err)
	}

	// Exactly one model must have been chosen for eviction (LRU: m1).
	waitFor(t, "one model marked pending unload", func() bool {
		n := 0
		for _, ms := range s.Snapshot().Models {
			if ms.State == ModelPendingUnload {
				n++
			}
		}
		return n == 1
	})
	var evicted string
	for _, ms := range s.Snapshot().Models {
		if ms.State == ModelPendingUnload {
			evicted = ms.Name
		}
	}
	// No new load may start while m1 is still draining.
	if n := rec.count("load:m5") + rec.count("load:m6"); n != 0 {
		t.Fatalf("load started before eviction completed: %d", n)
	}

	// Let the evicted model's running task finish; it unloads, m5 loads,
	// and the waiting task gets the freed execution slot.
	close(gates[evicted])
	{
		deadline := time.Now().Add(10 * time.Second)
		for {
			st := taskState(s, id5)
			if st == TaskRunning || st == TaskCompleted {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("m5 task never started; snapshot=%+v events=%v", s.Snapshot(), rec.events)
			}
			time.Sleep(time.Millisecond)
		}
	}
	if !rec.hasInOrder("unload:"+evicted, "load:m5") {
		t.Fatalf("expected unload:%s before load:m5, events: %v", evicted, rec.events)
	}
	if n := rec.count("load:m6"); n != 0 {
		t.Fatalf("m6 stole the freed residency slot")
	}

	// Drain everything.
	for _, m := range []string{"m1", "m2", "m3", "m4"} {
		if m != evicted {
			close(gates[m])
		}
	}
	all := append(busy, id5, id6)
	for _, id := range holders {
		all = append(all, id)
	}
	waitFor(t, "all tasks terminal", func() bool {
		for _, id := range all {
			st := taskState(s, id)
			if st != TaskCompleted && st != TaskFailed && st != TaskCanceled {
				return false
			}
		}
		return true
	})
	if got := taskState(s, id5); got != TaskCompleted {
		t.Fatalf("m5 task state = %v, want completed", got)
	}
	if got := taskState(s, id6); got != TaskCompleted {
		t.Fatalf("m6 task state = %v, want completed", got)
	}
}

// Freeing enough memory for a waiting task may require evicting several
// resident models; all chosen models drain before the load starts.
func TestEvictionOfMultipleModelsForMemory(t *testing.T) {
	rec := newRecorder()
	cfg := Config{MaxConcurrent: 8, MaxModels: 4, MemoryLimit: 8}
	s, err := New(cfg, rec.hooks())
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"a", "b", "c", "d"} {
		if err := s.RegisterModel(m, 2); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.RegisterModel("big", 6); err != nil {
		t.Fatal(err)
	}
	// Fill all 4 slots (8 memory) and let the tasks finish so models idle.
	for _, m := range []string{"a", "b", "c", "d"} {
		id, err := s.Submit(m, PriorityTranslator, nil)
		if err != nil {
			t.Fatal(err)
		}
		waitFor(t, m+" warmup done", func() bool { return taskState(s, id) == TaskCompleted })
	}
	// big needs 6 memory: must evict 3 models (3*2) to satisfy both the
	// slot (4->3) and memory (8+6-8=6) requirements.
	idBig, err := s.Submit("big", PriorityTranslator, nil)
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "big task to complete", func() bool { return taskState(s, idBig) == TaskCompleted })
	if got := rec.count("unload:"); got != 3 {
		t.Fatalf("unloads = %d, want 3", got)
	}
	snap := s.Snapshot()
	if len(snap.Models) != 2 { // big + one leftover
		t.Fatalf("resident models = %d, want 2", len(snap.Models))
	}
	var mem int64
	for _, ms := range snap.Models {
		mem += ms.DeclaredMemory
	}
	if mem > cfg.MemoryLimit {
		t.Fatalf("resident memory %d exceeds limit %d", mem, cfg.MemoryLimit)
	}
}

// When a load fails, every task waiting for that model fails with the
// reason, capacity is released, and a later submission retries the load.
func TestLoadFailureFailsWaitingTasksAndReleasesCapacity(t *testing.T) {
	var mu sync.Mutex
	loadCalls := 0
	firstLoad := make(chan struct{})
	hooks := Hooks{
		Load: func(_ context.Context, model string) error {
			mu.Lock()
			loadCalls++
			n := loadCalls
			mu.Unlock()
			if n == 1 {
				<-firstLoad // hold the first load until all tasks are queued
				return errors.New("disk gone")
			}
			return nil
		},
		Infer:  func(_ context.Context, _ string, _ any) error { return nil },
		Unload: func(_ context.Context, _ string) error { return nil },
	}
	s, err := New(testConfig(), hooks)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RegisterModel("m", 1); err != nil {
		t.Fatal(err)
	}
	var ids []uint64
	for i := 0; i < 5; i++ {
		id, err := s.Submit("m", PriorityTranslator, i)
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	waitFor(t, "first load to start", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return loadCalls == 1
	})
	close(firstLoad)
	waitFor(t, "all waiting tasks to fail", func() bool {
		for _, id := range ids {
			if taskState(s, id) != TaskFailed {
				return false
			}
		}
		return true
	})
	for _, id := range ids {
		info, _ := s.Task(id)
		if !strings.Contains(info.Err, "disk gone") {
			t.Fatalf("task %d err = %q, want load failure reason", id, info.Err)
		}
	}
	// Capacity was released: snapshot shows no resident model.
	if n := len(s.Snapshot().Models); n != 0 {
		t.Fatalf("resident models after load failure = %d, want 0", n)
	}
	// A new submission triggers a fresh load, which now succeeds.
	id, err := s.Submit("m", PriorityTranslator, nil)
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "retried task to complete", func() bool { return taskState(s, id) == TaskCompleted })
	mu.Lock()
	defer mu.Unlock()
	if loadCalls != 2 {
		t.Fatalf("load calls = %d, want 2", loadCalls)
	}
}

// Offlining a model with a running task: new submissions are rejected,
// queued tasks fail at once, the running task finishes, then it unloads.
func TestOfflineWhileRunning(t *testing.T) {
	rec := newRecorder()
	cfg := Config{MaxConcurrent: 1, MaxModels: 4, MemoryLimit: 1 << 30}
	s, err := New(cfg, rec.hooks())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RegisterModel("m", 1); err != nil {
		t.Fatal(err)
	}
	gate := rec.gateInfer("hold")
	idRun, err := s.Submit("m", PriorityTranslator, "hold")
	if err != nil {
		t.Fatal(err)
	}
	idQueued, err := s.Submit("m", PriorityTranslator, "queued")
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "task running", func() bool { return taskState(s, idRun) == TaskRunning })

	if err := s.OfflineModel("m"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Submit("m", PriorityTranslator, nil); !errors.Is(err, ErrModelOffline) {
		t.Fatalf("submit after offline = %v, want ErrModelOffline", err)
	}
	if got := taskState(s, idQueued); got != TaskFailed {
		t.Fatalf("queued task state = %v, want failed", got)
	}
	if n := rec.count("unload:m"); n != 0 {
		t.Fatalf("unload before running task drained: %d", n)
	}
	close(gate)
	waitFor(t, "unload after drain", func() bool { return rec.count("unload:m") == 1 })
	if got := taskState(s, idRun); got != TaskCompleted {
		t.Fatalf("running task state = %v, want completed", got)
	}
}

// Offlining a model while it loads: the load finishes, then the model
// unloads immediately without serving any task.
func TestOfflineWhileLoading(t *testing.T) {
	rec := newRecorder()
	s, err := New(testConfig(), rec.hooks())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RegisterModel("m", 1); err != nil {
		t.Fatal(err)
	}
	loadGate := rec.gateLoad("m")
	id, err := s.Submit("m", PriorityReviewer, nil)
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "load started", func() bool { return rec.count("load:m") == 1 })
	if err := s.OfflineModel("m"); err != nil {
		t.Fatal(err)
	}
	if got := taskState(s, id); got != TaskFailed {
		t.Fatalf("queued task state = %v, want failed", got)
	}
	close(loadGate)
	waitFor(t, "unload after load completes", func() bool { return rec.count("unload:m") == 1 })
	if !rec.hasInOrder("load:m", "unload:m") {
		t.Fatalf("expected load then unload, events: %v", rec.events)
	}
	if n := rec.count("infer:m"); n != 0 {
		t.Fatalf("offlined model served %d inferences", n)
	}
}

// Cancel racing dispatch: every task ends in exactly one terminal state,
// and a task reported canceled never enters inference.
func TestCancelRacingDispatch(t *testing.T) {
	rec := newRecorder()
	cfg := Config{MaxConcurrent: 4, MaxModels: 4, MemoryLimit: 1 << 30}
	s, err := New(cfg, rec.hooks())
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"a", "b", "c", "d"} {
		if err := s.RegisterModel(m, 1); err != nil {
			t.Fatal(err)
		}
	}
	models := []string{"a", "b", "c", "d"}
	const n = 2000
	ids := make([]uint64, n)
	canceled := make([]bool, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		id, err := s.Submit(models[i%4], Priority(i%2), i)
		if err != nil {
			t.Fatal(err)
		}
		ids[i] = id
	}
	for i := 0; i < n; i += 3 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			canceled[i] = s.Cancel(ids[i])
		}(i)
	}
	wg.Wait()
	waitFor(t, "all tasks terminal", func() bool {
		for _, id := range ids {
			st := taskState(s, id)
			if st != TaskCompleted && st != TaskFailed && st != TaskCanceled {
				return false
			}
		}
		return true
	})
	infered := make(map[any]bool)
	for _, p := range rec.inferedPayloads() {
		infered[p] = true
	}
	for i, id := range ids {
		info, _ := s.Task(id)
		if canceled[i] {
			if info.State != TaskCanceled {
				t.Fatalf("task %d reported canceled but state = %v", id, info.State)
			}
			if infered[i] {
				t.Fatalf("canceled task %d entered inference", id)
			}
			if !info.StartedAt.IsZero() {
				t.Fatalf("canceled task %d has a start time", id)
			}
		} else if info.State == TaskCanceled {
			t.Fatalf("task %d canceled but Cancel returned false", id)
		}
		if info.EndedAt.IsZero() {
			t.Fatalf("task %d missing end time", id)
		}
	}
}
