package sched

import (
	"errors"
	"testing"
)

// 排头任务等加载时，后面模型可用的任务照常开始。
func TestHeadWaitingForLoadDoesNotBlockOthers(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "A", 10)
	mustRegister(t, s, "B", 10)

	// 让 B 先驻留。
	idB0, _ := s.Submit("B", PriorityTranslator)
	rt.releaseInfer(idB0)
	waitFor(t, "B ready", func() bool {
		p, ok := modelPhase(s, "B")
		return ok && p == ModelReady
	})

	// A 的加载卡住，排头任务等加载。
	rt.blockLoad("A")
	idA, _ := s.Submit("A", PriorityTranslator)
	waitFor(t, "A load started", func() bool { return rt.loadCount("A") == 1 })

	// 后续 B 的任务应照常开始。
	idB, _ := s.Submit("B", PriorityTranslator)
	waitFor(t, "B task running", func() bool { return taskStatus(s, idB) == TaskRunning })
	if st := taskStatus(s, idA); st != TaskPending {
		t.Fatalf("head task should still be pending, got %v", st)
	}

	rt.releaseLoad("A")
	rt.releaseInfer(idA)
	rt.releaseInfer(idB)
	waitFor(t, "A task completed", func() bool { return taskStatus(s, idA) == TaskCompleted })
}

// 派发顺序：高优先级在前，同级按提交先后。
func TestDispatchOrder(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 1, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "M", 10)

	first, _ := s.Submit("M", PriorityTranslator)
	waitFor(t, "first running", func() bool { return taskStatus(s, first) == TaskRunning })

	t1, _ := s.Submit("M", PriorityTranslator)
	t2, _ := s.Submit("M", PriorityTranslator)
	r1, _ := s.Submit("M", PriorityReviewer)
	r2, _ := s.Submit("M", PriorityReviewer)

	// 审校员优先，按提交先后；再译员按提交先后。
	want := []uint64{r1, r2, t1, t2}
	rt.releaseInfer(first)
	for _, id := range want {
		waitFor(t, "task running", func() bool { return taskStatus(s, id) == TaskRunning })
		rt.releaseInfer(id)
		waitFor(t, "task completed", func() bool { return taskStatus(s, id) == TaskCompleted })
	}
}

// 声明内存超过总上限拒绝注册；未注册模型拒绝提交。
func TestRegisterAndSubmitValidation(t *testing.T) {
	rt := newFakeRuntime()
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	if err := s.RegisterModel("huge", 101); !errors.Is(err, ErrModelTooLarge) {
		t.Fatalf("want ErrModelTooLarge, got %v", err)
	}
	if err := s.RegisterModel("ok", 100); err != nil {
		t.Fatalf("register: %v", err)
	}
	if _, err := s.Submit("ghost", PriorityTranslator); !errors.Is(err, ErrModelNotRegistered) {
		t.Fatalf("want ErrModelNotRegistered, got %v", err)
	}
	if _, err := s.Submit("ok", PriorityTranslator); err != nil {
		t.Fatalf("submit: %v", err)
	}
}

func mustRegister(t *testing.T, s *Scheduler, name string, mem int64) {
	t.Helper()
	if err := s.RegisterModel(name, mem); err != nil {
		t.Fatalf("register %s: %v", name, err)
	}
}
