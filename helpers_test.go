package prosa

import (
	"context"
	"sync"
	"testing"
	"time"
)

// waitFor polls cond until it holds or the deadline passes.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timeout waiting for %s", what)
}

// recorder hooks log every call and can gate load/infer per model.
type recorder struct {
	mu         sync.Mutex
	events     []string
	loadGates  map[string]chan struct{}
	inferGates map[any]chan struct{}
	infered    []any // payloads passed to Infer, in call order
}

func newRecorder() *recorder {
	return &recorder{
		loadGates:  make(map[string]chan struct{}),
		inferGates: make(map[any]chan struct{}),
	}
}

func (r *recorder) hooks() Hooks {
	return Hooks{
		Load: func(_ context.Context, model string) error {
			r.mu.Lock()
			r.events = append(r.events, "load:"+model)
			gate := r.loadGates[model]
			r.mu.Unlock()
			if gate != nil {
				<-gate
			}
			return nil
		},
		Infer: func(_ context.Context, model string, payload any) error {
			r.mu.Lock()
			r.events = append(r.events, "infer:"+model)
			r.infered = append(r.infered, payload)
			gate := r.inferGates[payload]
			r.mu.Unlock()
			if gate != nil {
				<-gate
			}
			return nil
		},
		Unload: func(_ context.Context, model string) error {
			r.mu.Lock()
			r.events = append(r.events, "unload:"+model)
			r.mu.Unlock()
			return nil
		},
	}
}

func (r *recorder) gateLoad(model string) chan struct{} {
	ch := make(chan struct{})
	r.mu.Lock()
	r.loadGates[model] = ch
	r.mu.Unlock()
	return ch
}

// gateInfer blocks Infer calls carrying the given payload until the
// returned channel is closed.
func (r *recorder) gateInfer(payload any) chan struct{} {
	ch := make(chan struct{})
	r.mu.Lock()
	r.inferGates[payload] = ch
	r.mu.Unlock()
	return ch
}

func (r *recorder) count(prefix string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, e := range r.events {
		if len(e) >= len(prefix) && e[:len(prefix)] == prefix {
			n++
		}
	}
	return n
}

// hasInOrder reports whether the events matching each prefix appear
// in the given relative order (other events may be interleaved).
func (r *recorder) hasInOrder(prefixes ...string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	next := 0
	for _, e := range r.events {
		if next < len(prefixes) && e == prefixes[next] {
			next++
		}
	}
	return next == len(prefixes)
}

func (r *recorder) inferedPayloads() []any {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]any, len(r.infered))
	copy(out, r.infered)
	return out
}

func taskState(s *Scheduler, id uint64) TaskState {
	info, ok := s.Task(id)
	if !ok {
		return TaskPending
	}
	return info.State
}
