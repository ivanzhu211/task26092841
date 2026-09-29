package sched

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

// 取消只对还没开始的任务有效；报告已取消的任务绝不会进入推理。
func TestCancel(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 1, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "M", 10)

	running, _ := s.Submit("M", PriorityTranslator)
	waitFor(t, "running", func() bool { return taskStatus(s, running) == TaskRunning })
	queued, _ := s.Submit("M", PriorityTranslator)

	if s.Cancel(running) {
		t.Fatalf("running task cannot be cancelled")
	}
	if !s.Cancel(queued) {
		t.Fatalf("queued task should be cancellable")
	}
	if st := taskStatus(s, queued); st != TaskCancelled {
		t.Fatalf("want cancelled, got %v", st)
	}
	if s.Cancel(queued) {
		t.Fatalf("already cancelled")
	}
	rt.releaseInfer(running)
	waitFor(t, "running done", func() bool { return taskStatus(s, running) == TaskCompleted })
	if s.Cancel(running) {
		t.Fatalf("completed task cannot be cancelled")
	}
	if n := rt.inferCallCount(); n != 1 {
		t.Fatalf("cancelled task must never enter inference, inferCalls=%d", n)
	}
}

// 取消与派发大量同时发生。
func TestCancelRacesDispatch(t *testing.T) {
	rt := newFakeRuntime()
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	for i := 0; i < 4; i++ {
		mustRegister(t, s, fmt.Sprintf("M%d", i), 10)
	}
	const n = 2000
	ids := make([]uint64, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			id, err := s.Submit(fmt.Sprintf("M%d", i%4), Priority(i%2))
			if err != nil {
				t.Errorf("submit: %v", err)
				return
			}
			ids[i] = id
		}()
		if i%2 == 0 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				s.Cancel(uint64(i + 1)) // 取消可能尚未提交的 ID，结果只能是已取消或已开始之一
			}()
		}
	}
	wg.Wait()
	waitFor(t, "all terminal", func() bool {
		for _, id := range ids {
			if id == 0 {
				return false
			}
			st := taskStatus(s, id)
			if st != TaskCompleted && st != TaskCancelled {
				return false
			}
		}
		return true
	})
	// 已取消的从未进入推理；已完成数等于推理调用数。
	completed, cancelled := 0, 0
	for _, id := range ids {
		switch taskStatus(s, id) {
		case TaskCompleted:
			completed++
		case TaskCancelled:
			cancelled++
		}
	}
	if completed != rt.inferCallCount() {
		t.Fatalf("completed=%d != inferCalls=%d", completed, rt.inferCallCount())
	}
	if completed+cancelled != n {
		t.Fatalf("every task exactly one terminal state: %d+%d != %d", completed, cancelled, n)
	}
}

// 加载失败：在等这个模型的任务全部失败并带上原因，名额和内存释放，
// 之后提交的任务会重新加载。
func TestLoadFailure(t *testing.T) {
	rt := newFakeRuntime()
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "BAD", 10)
	mustRegister(t, s, "OK", 10)

	loadErr := errors.New("disk corrupted")
	rt.loadErr["BAD"] = loadErr
	t1, _ := s.Submit("BAD", PriorityTranslator)
	t2, _ := s.Submit("BAD", PriorityReviewer)
	waitFor(t, "t1 failed", func() bool { return taskStatus(s, t1) == TaskFailed })
	waitFor(t, "t2 failed", func() bool { return taskStatus(s, t2) == TaskFailed })
	info, _ := s.Task(t1)
	if !errors.Is(info.Err, loadErr) {
		t.Fatalf("failure should carry the load reason, got %v", info.Err)
	}
	// 名额和内存释放。
	if got := len(s.Snapshot().Models); got != 0 {
		t.Fatalf("residency should be released, got %d models", got)
	}
	// 修复后重新提交会重新加载并成功。
	delete(rt.loadErr, "BAD")
	t3, _ := s.Submit("BAD", PriorityTranslator)
	waitFor(t, "t3 completed", func() bool { return taskStatus(s, t3) == TaskCompleted })
	if n := rt.loadCount("BAD"); n != 2 {
		t.Fatalf("BAD should be loaded twice, got %d", n)
	}
	// 其他模型不受影响。
	t4, _ := s.Submit("OK", PriorityTranslator)
	waitFor(t, "t4 completed", func() bool { return taskStatus(s, t4) == TaskCompleted })
}

// 推理失败只让该任务失败，不重试。
func TestInferFailure(t *testing.T) {
	rt := newFakeRuntime()
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "M", 10)
	inferErr := errors.New("oom during inference")
	t1, _ := s.Submit("M", PriorityTranslator)
	rt.inferErr[t1] = inferErr
	t2, _ := s.Submit("M", PriorityTranslator)
	waitFor(t, "t1 failed", func() bool { return taskStatus(s, t1) == TaskFailed })
	waitFor(t, "t2 completed", func() bool { return taskStatus(s, t2) == TaskCompleted })
	info, _ := s.Task(t1)
	if !errors.Is(info.Err, inferErr) {
		t.Fatalf("want infer error, got %v", info.Err)
	}
	if n := rt.inferCallCount(); n != 2 {
		t.Fatalf("no retry: inferCalls=%d", n)
	}
}

// 下线在跑模型：新提交拒绝，排队任务立即失败，在跑的跑完后卸载。
func TestOfflineRunningModel(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 1, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "M", 10)

	running, _ := s.Submit("M", PriorityTranslator)
	waitFor(t, "running", func() bool { return taskStatus(s, running) == TaskRunning })
	queued, _ := s.Submit("M", PriorityTranslator)

	s.Offline("M")
	if _, err := s.Submit("M", PriorityTranslator); !errors.Is(err, ErrModelOffline) {
		t.Fatalf("submit after offline should be rejected, got %v", err)
	}
	if st := taskStatus(s, queued); st != TaskFailed {
		t.Fatalf("queued task should fail immediately, got %v", st)
	}
	if st := taskStatus(s, running); st != TaskRunning {
		t.Fatalf("running task should keep running, got %v", st)
	}
	if n := rt.unloadCount("M"); n != 0 {
		t.Fatalf("should not unload while running")
	}
	rt.releaseInfer(running)
	waitFor(t, "unloaded", func() bool { return rt.unloadCount("M") == 1 })
	waitFor(t, "running completed", func() bool { return taskStatus(s, running) == TaskCompleted })
	if got := len(s.Snapshot().Models); got != 0 {
		t.Fatalf("model should be gone after unload, got %d", got)
	}
}

// 下线加载中的模型：加载完立即卸载。
func TestOfflineLoadingModel(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "M", 10)

	rt.blockLoad("M")
	queued, _ := s.Submit("M", PriorityTranslator)
	waitFor(t, "load started", func() bool { return rt.loadCount("M") == 1 })

	s.Offline("M")
	if st := taskStatus(s, queued); st != TaskFailed {
		t.Fatalf("queued task should fail immediately, got %v", st)
	}
	rt.releaseLoad("M")
	waitFor(t, "unloaded right after load", func() bool { return rt.unloadCount("M") == 1 })
	if got := len(s.Snapshot().Models); got != 0 {
		t.Fatalf("model should be gone, got %d", got)
	}
	if n := rt.inferCallCount(); n != 0 {
		t.Fatalf("no inference should happen on offlined model")
	}
}
