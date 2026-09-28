package prosa

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRegistrationValidation(t *testing.T) {
	if _, err := New(Config{MaxConcurrent: 0, MaxModels: 1, MemoryLimit: 1}, Hooks{}); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("bad config err = %v", err)
	}
	if _, err := New(testConfig(), Hooks{}); !errors.Is(err, ErrNilHooks) {
		t.Fatalf("nil hooks err = %v", err)
	}
	s, err := New(Config{MaxConcurrent: 1, MaxModels: 1, MemoryLimit: 10}, Hooks{
		Load:   func(context.Context, string) error { return nil },
		Infer:  func(context.Context, string, any) error { return nil },
		Unload: func(context.Context, string) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RegisterModel("too-big", 11); !errors.Is(err, ErrMemoryTooLarge) {
		t.Fatalf("oversize err = %v", err)
	}
	if err := s.RegisterModel("zero", 0); !errors.Is(err, ErrNonPositiveMemory) {
		t.Fatalf("zero mem err = %v", err)
	}
	if err := s.RegisterModel("m", 5); err != nil {
		t.Fatal(err)
	}
	if err := s.RegisterModel("m", 5); !errors.Is(err, ErrModelExists) {
		t.Fatalf("duplicate err = %v", err)
	}
	if _, err := s.Submit("ghost", PriorityTranslator, nil); !errors.Is(err, ErrModelUnknown) {
		t.Fatalf("unknown model err = %v", err)
	}
}

// Higher-priority tasks dispatch first; within a priority, earlier
// submissions dispatch first.
func TestPriorityThenFIFOOrder(t *testing.T) {
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
	if _, err := s.Submit("m", PriorityTranslator, "hold"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "holder running", func() bool { return rec.count("infer:m") == 1 })
	for _, p := range []string{"low1", "high1", "low2", "high2"} {
		prio := PriorityTranslator
		if strings.HasPrefix(p, "high") {
			prio = PriorityReviewer
		}
		if _, err := s.Submit("m", prio, p); err != nil {
			t.Fatal(err)
		}
	}
	close(gate)
	waitFor(t, "all tasks done", func() bool { return rec.count("infer:m") == 5 })
	got := rec.inferedPayloads()
	want := []any{"hold", "high1", "high2", "low1", "low2"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dispatch order = %v, want %v", got, want)
		}
	}
}

// An inference failure fails only that task, without retry, and the model
// keeps serving later tasks.
func TestInferFailureFailsOnlyThatTask(t *testing.T) {
	rec := newRecorder()
	s, err := New(testConfig(), Hooks{
		Load: func(context.Context, string) error { return nil },
		Infer: func(_ context.Context, _ string, payload any) error {
			rec.mu.Lock()
			rec.infered = append(rec.infered, payload)
			rec.mu.Unlock()
			if payload == "bad" {
				return errors.New("boom")
			}
			return nil
		},
		Unload: func(context.Context, string) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RegisterModel("m", 1); err != nil {
		t.Fatal(err)
	}
	idBad, err := s.Submit("m", PriorityTranslator, "bad")
	if err != nil {
		t.Fatal(err)
	}
	idGood, err := s.Submit("m", PriorityTranslator, "good")
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "both tasks terminal", func() bool {
		bad := taskState(s, idBad)
		good := taskState(s, idGood)
		return (bad == TaskFailed || bad == TaskCompleted) &&
			(good == TaskFailed || good == TaskCompleted)
	})
	info, _ := s.Task(idBad)
	if info.State != TaskFailed || !strings.Contains(info.Err, "boom") {
		t.Fatalf("bad task = %+v, want failed with cause", info)
	}
	if got := taskState(s, idGood); got != TaskCompleted {
		t.Fatalf("good task state = %v, want completed", got)
	}
	if n := len(rec.inferedPayloads()); n != 2 {
		t.Fatalf("infer calls = %d, want 2 (no retry)", n)
	}
	if info.StartedAt.Before(info.CreatedAt) || info.EndedAt.Before(info.StartedAt) {
		t.Fatalf("timestamps out of order: %+v", info)
	}
}

// A task waiting for an eviction must eventually run even while the
// resident models are kept busy by a stream of same-priority tasks.
func TestNoStarvationWhileEvictionPending(t *testing.T) {
	rec := newRecorder()
	cfg := Config{MaxConcurrent: 2, MaxModels: 2, MemoryLimit: 2}
	s, err := New(cfg, rec.hooks())
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"a", "b", "c"} {
		if err := s.RegisterModel(m, 1); err != nil {
			t.Fatal(err)
		}
	}
	gateA := rec.gateInfer("hold-a")
	gateB := rec.gateInfer("hold-b")
	holdA, _ := s.Submit("a", PriorityTranslator, "hold-a")
	holdB, _ := s.Submit("b", PriorityTranslator, "hold-b")
	waitFor(t, "both holders running", func() bool { return rec.count("infer:") == 2 })

	// Stream of same-priority tasks keeps the resident models busy.
	stop := make(chan struct{})
	var streamWG sync.WaitGroup
	streamWG.Add(1)
	var streamIDs sync.Map
	go func() {
		defer streamWG.Done()
		i := 0
		for {
			select {
			case <-stop:
				return
			default:
			}
			m := "a"
			if i%2 == 1 {
				m = "b"
			}
			if id, err := s.Submit(m, PriorityTranslator, i); err == nil {
				streamIDs.Store(id, struct{}{})
			}
			i++
			time.Sleep(time.Millisecond)
		}
	}()

	idC, err := s.Submit("c", PriorityTranslator, "need-c")
	if err != nil {
		t.Fatal(err)
	}
	// One of a/b gets evicted; release both holders so it can drain.
	waitFor(t, "one model evicted", func() bool {
		for _, ms := range s.Snapshot().Models {
			if ms.State == ModelPendingUnload {
				return true
			}
		}
		return false
	})
	close(gateA)
	close(gateB)
	waitFor(t, "waiting task completes despite the stream", func() bool {
		return taskState(s, idC) == TaskCompleted
	})
	close(stop)
	streamWG.Wait()
	for _, id := range []uint64{holdA, holdB} {
		if st := taskState(s, id); st != TaskCompleted {
			t.Fatalf("holder %d state = %v", id, st)
		}
	}
}
