package sched

import (
	"testing"
)

// 驻留模型被同级新任务占着时，需要新模型的任务在被挑中模型手上的任务跑完、
// 加载完成后拿到空出的执行位，且腾出的位置不被别的加载抢走。
func TestEvictionForNewModelTask(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 2, MaxResident: 2, MaxMemory: 20, Hooks: rt.hooks()})
	mustRegister(t, s, "M1", 10)
	mustRegister(t, s, "M2", 10)
	mustRegister(t, s, "M3", 10)

	// M1、M2 驻留并各占一个执行位（推理阻塞中）。
	a1, _ := s.Submit("M1", PriorityTranslator)
	b1, _ := s.Submit("M2", PriorityTranslator)
	waitFor(t, "a1 running", func() bool { return taskStatus(s, a1) == TaskRunning })
	waitFor(t, "b1 running", func() bool { return taskStatus(s, b1) == TaskRunning })

	// M3 的加载先堵住，便于观察腾位过程。
	rt.blockLoad("M3")
	c1, _ := s.Submit("M3", PriorityTranslator)

	// 同级新任务继续涌向 M1、M2。
	d1, _ := s.Submit("M1", PriorityTranslator)
	d2, _ := s.Submit("M2", PriorityTranslator)

	// 被挑中的应是 M1（都有任务在跑，M1 最久没接新任务）；M1 不再接新任务。
	waitFor(t, "M1 unloading", func() bool {
		p, ok := modelPhase(s, "M1")
		return ok && p == ModelUnloading
	})
	if p, _ := modelPhase(s, "M2"); p != ModelReady {
		t.Fatalf("M2 should stay ready, got %v", p)
	}

	// M1 手上的任务跑完 → M1 卸载 → M3 开始加载。
	rt.releaseInfer(a1)
	waitFor(t, "M1 unloaded", func() bool { return rt.unloadCount("M1") == 1 })
	waitFor(t, "M3 load started", func() bool { return rt.loadCount("M3") == 1 })

	// 腾出的位置没有被 M1 的重新加载抢走。
	if n := rt.loadCount("M1"); n != 1 {
		t.Fatalf("M1 should not reload while M3 load pending, loadCount=%d", n)
	}

	// 空出的执行位先给了模型可用的 d2；M3 加载完成后 c1 排在使用空位的最前。
	rt.releaseLoad("M3")
	waitFor(t, "M3 ready", func() bool {
		p, ok := modelPhase(s, "M3")
		return ok && p == ModelReady
	})
	rt.releaseInfer(b1)
	waitFor(t, "c1 running", func() bool { return taskStatus(s, c1) == TaskRunning })
	if st := taskStatus(s, d1); st == TaskRunning {
		t.Fatalf("later same-priority task should not jump ahead of c1")
	}

	// 收尾：全部放行，所有任务到达终态。
	for _, id := range []uint64{c1, d1, d2} {
		rt.releaseInfer(id)
	}
	waitFor(t, "all terminal", func() bool {
		for _, id := range []uint64{a1, b1, c1, d1, d2} {
			st := taskStatus(s, id)
			if st != TaskCompleted && st != TaskFailed && st != TaskCancelled {
				return false
			}
		}
		return true
	})
}

// 内存不够时连腾多个模型。
func TestEvictMultipleForMemory(t *testing.T) {
	rt := newFakeRuntime()
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 30, Hooks: rt.hooks()})
	for _, name := range []string{"S1", "S2", "S3"} {
		mustRegister(t, s, name, 10)
	}
	mustRegister(t, s, "BIG", 25)

	// 三个小模型驻留（内存合计 30，占满）。
	for _, name := range []string{"S1", "S2", "S3"} {
		id, _ := s.Submit(name, PriorityTranslator)
		waitFor(t, name+" done", func() bool { return taskStatus(s, id) == TaskCompleted })
	}
	if got := len(s.Snapshot().Models); got != 3 {
		t.Fatalf("want 3 residents, got %d", got)
	}

	// BIG 需要 25 内存，必须连腾三个小模型。
	id, _ := s.Submit("BIG", PriorityTranslator)
	waitFor(t, "BIG completed", func() bool { return taskStatus(s, id) == TaskCompleted })
	for _, name := range []string{"S1", "S2", "S3"} {
		if n := rt.unloadCount(name); n != 1 {
			t.Fatalf("%s should be unloaded once, got %d", name, n)
		}
	}
	snap := s.Snapshot()
	if len(snap.Models) != 1 || snap.Models[0].Name != "BIG" {
		t.Fatalf("want only BIG resident, got %+v", snap.Models)
	}
}

// 腾位顺序：先挑没有任务在跑的，再挑有任务在跑的；各自按最久没接新任务的在前。
func TestEvictionPrefersIdleThenLRU(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 4, MaxResident: 4, MaxMemory: 40, Hooks: rt.hooks()})
	for _, name := range []string{"M1", "M2", "M3", "M4", "M5", "M6"} {
		mustRegister(t, s, name, 10)
	}

	// M1、M2 驻留且空闲（M1 最久没接新任务），M3 驻留且有任务在跑，M4 驻留且最新。
	i1, _ := s.Submit("M1", PriorityTranslator)
	rt.releaseInfer(i1)
	waitFor(t, "i1 done", func() bool { return taskStatus(s, i1) == TaskCompleted })
	i2, _ := s.Submit("M2", PriorityTranslator)
	rt.releaseInfer(i2)
	waitFor(t, "i2 done", func() bool { return taskStatus(s, i2) == TaskCompleted })
	i3, _ := s.Submit("M3", PriorityTranslator)
	waitFor(t, "i3 running", func() bool { return taskStatus(s, i3) == TaskRunning })
	i4, _ := s.Submit("M4", PriorityTranslator)
	rt.releaseInfer(i4)
	waitFor(t, "i4 done", func() bool { return taskStatus(s, i4) == TaskCompleted })
	if got := len(s.Snapshot().Models); got != 4 {
		t.Fatalf("want 4 residents, got %d", got)
	}

	// M5 到来腾一个位置：应挑 M1（无任务在跑且最久没接新任务）。
	i5, _ := s.Submit("M5", PriorityTranslator)
	rt.releaseInfer(i5)
	waitFor(t, "i5 done", func() bool { return taskStatus(s, i5) == TaskCompleted })
	if n := rt.unloadCount("M1"); n != 1 {
		t.Fatalf("M1 should be evicted, unloadCount=%d", n)
	}
	if n := rt.unloadCount("M3"); n != 0 {
		t.Fatalf("busy M3 should not be evicted while idle models exist")
	}

	// M6 到来再腾一个：应挑 M2（剩下的空闲里最久没接新任务的），仍不动 M3。
	i6, _ := s.Submit("M6", PriorityTranslator)
	rt.releaseInfer(i6)
	waitFor(t, "i6 done", func() bool { return taskStatus(s, i6) == TaskCompleted })
	if n := rt.unloadCount("M2"); n != 1 {
		t.Fatalf("M2 should be evicted, unloadCount=%d", n)
	}
	if n := rt.unloadCount("M3"); n != 0 {
		t.Fatalf("busy M3 should not be evicted while idle models exist")
	}
	rt.releaseInfer(i3)
}
