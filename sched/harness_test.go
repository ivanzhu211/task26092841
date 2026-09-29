package sched

import (
	"context"
	"sync"
	"testing"
	"time"
)

// fakeRuntime 可控的加载/推理/卸载钩子，用于测试。
type fakeRuntime struct {
	mu          sync.Mutex
	blockInfer  bool // 推理默认阻塞，等待 releaseInfer
	loadBlock   map[string]chan struct{}
	inferBlock  map[uint64]chan struct{}
	released    map[uint64]bool
	loadErr     map[string]error
	inferErr    map[uint64]error
	loadCalls   []string
	unloadCalls []string
	inferCalls  int
}

func newFakeRuntime() *fakeRuntime {
	return &fakeRuntime{
		loadBlock:  make(map[string]chan struct{}),
		inferBlock: make(map[uint64]chan struct{}),
		released:   make(map[uint64]bool),
		loadErr:    make(map[string]error),
		inferErr:   make(map[uint64]error),
	}
}

func (r *fakeRuntime) hooks() Hooks {
	return Hooks{Load: r.load, Infer: r.infer, Unload: r.unload}
}

func (r *fakeRuntime) load(_ context.Context, name string) error {
	r.mu.Lock()
	r.loadCalls = append(r.loadCalls, name)
	ch := r.loadBlock[name]
	r.mu.Unlock()
	if ch != nil {
		<-ch
	}
	r.mu.Lock()
	err := r.loadErr[name]
	r.mu.Unlock()
	return err
}

func (r *fakeRuntime) infer(_ context.Context, _ string, id uint64) error {
	if r.blockInfer {
		r.mu.Lock()
		ch := r.inferBlock[id]
		if ch == nil {
			ch = make(chan struct{})
			r.inferBlock[id] = ch
		}
		r.mu.Unlock()
		<-ch
	}
	r.mu.Lock()
	r.inferCalls++
	err := r.inferErr[id]
	r.mu.Unlock()
	return err
}

func (r *fakeRuntime) unload(name string) {
	r.mu.Lock()
	r.unloadCalls = append(r.unloadCalls, name)
	r.mu.Unlock()
}

// blockLoad 让指定模型的下一次及后续加载阻塞，直到 releaseLoad。
func (r *fakeRuntime) blockLoad(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.loadBlock[name] = make(chan struct{})
}

func (r *fakeRuntime) releaseLoad(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if ch, ok := r.loadBlock[name]; ok {
		close(ch)
		delete(r.loadBlock, name)
	}
}

func (r *fakeRuntime) releaseInfer(id uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released[id] {
		return
	}
	r.released[id] = true
	ch := r.inferBlock[id]
	if ch == nil {
		ch = make(chan struct{})
		r.inferBlock[id] = ch
	}
	close(ch)
}

func (r *fakeRuntime) loadCount(name string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, l := range r.loadCalls {
		if l == name {
			n++
		}
	}
	return n
}

func (r *fakeRuntime) unloadCount(name string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, u := range r.unloadCalls {
		if u == name {
			n++
		}
	}
	return n
}

func (r *fakeRuntime) inferCallCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.inferCalls
}

// waitFor 轮询直到条件满足，超时失败。
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timeout waiting for: %s", what)
}

func taskStatus(s *Scheduler, id uint64) TaskStatus {
	info, _ := s.Task(id)
	return info.Status
}

func modelPhase(s *Scheduler, name string) (ModelPhase, bool) {
	for _, m := range s.Snapshot().Models {
		if m.Name == name {
			return m.Phase, true
		}
	}
	return 0, false
}
