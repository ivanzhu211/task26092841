package sched

import (
	"testing"
)

// 整体画面与单任务查询：排队数、在跑数、驻留模型状态、时间戳。
func TestSnapshotAndTaskQuery(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 2, MaxResident: 4, MaxMemory: 100, Hooks: rt.hooks()})
	mustRegister(t, s, "A", 10)
	mustRegister(t, s, "B", 20)

	// A 驻留并占满执行位；B 加载中。
	a1, _ := s.Submit("A", PriorityTranslator)
	a2, _ := s.Submit("A", PriorityReviewer)
	rt.blockLoad("B")
	b1, _ := s.Submit("B", PriorityReviewer)
	waitFor(t, "a1 running", func() bool { return taskStatus(s, a1) == TaskRunning })
	waitFor(t, "a2 running", func() bool { return taskStatus(s, a2) == TaskRunning })
	waitFor(t, "B loading", func() bool { return rt.loadCount("B") == 1 })
	q1, _ := s.Submit("A", PriorityTranslator)
	q2, _ := s.Submit("A", PriorityReviewer)

	snap := s.Snapshot()
	if snap.Running != 2 {
		t.Fatalf("running = %d, want 2", snap.Running)
	}
	if snap.Queued[PriorityTranslator] != 1 || snap.Queued[PriorityReviewer] != 2 {
		t.Fatalf("queued = %v, want {0:1 1:2}", snap.Queued)
	}
	phases := map[string]ModelPhase{}
	running := map[string]int{}
	mems := map[string]int64{}
	for _, m := range snap.Models {
		phases[m.Name] = m.Phase
		running[m.Name] = m.Running
		mems[m.Name] = m.Memory
	}
	if phases["A"] != ModelReady || phases["B"] != ModelLoading {
		t.Fatalf("phases = %v", phases)
	}
	if running["A"] != 2 || mems["A"] != 10 || mems["B"] != 20 {
		t.Fatalf("model info wrong: running=%v mem=%v", running, mems)
	}

	// 单任务查询：时间戳与状态。
	info, ok := s.Task(a1)
	if !ok || info.Status != TaskRunning {
		t.Fatalf("task a1: %+v ok=%v", info, ok)
	}
	if info.CreatedAt.IsZero() || info.StartedAt.IsZero() || !info.EndedAt.IsZero() {
		t.Fatalf("running task timestamps wrong: %+v", info)
	}
	qinfo, _ := s.Task(q1)
	if qinfo.Status != TaskPending || !qinfo.StartedAt.IsZero() {
		t.Fatalf("pending task should have no start time: %+v", qinfo)
	}
	if _, ok := s.Task(99999); ok {
		t.Fatalf("unknown task should not be found")
	}

	// 结束后有结束时间。
	rt.releaseInfer(a1)
	rt.releaseInfer(a2)
	rt.releaseLoad("B")
	rt.releaseInfer(b1)
	rt.releaseInfer(q1)
	rt.releaseInfer(q2)
	waitFor(t, "a1 completed", func() bool { return taskStatus(s, a1) == TaskCompleted })
	info, _ = s.Task(a1)
	if info.EndedAt.IsZero() || info.EndedAt.Before(info.StartedAt) {
		t.Fatalf("completed task timestamps wrong: %+v", info)
	}
}
