package sched

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// 已驻留的模型一直被同级新任务占着，也不能让等腾位置的任务无限期等下去。
func TestNoStarvationUnderContinuousSamePriorityLoad(t *testing.T) {
	rt := newFakeRuntime()
	rt.blockInfer = true
	s := New(Config{MaxConcurrent: 2, MaxResident: 2, MaxMemory: 20, Hooks: rt.hooks()})
	mustRegister(t, s, "M1", 10)
	mustRegister(t, s, "M2", 10)
	mustRegister(t, s, "M3", 10)

	// M1、M2 驻留，各有一个在跑、一个排队的同级任务。
	a1, _ := s.Submit("M1", PriorityTranslator)
	b1, _ := s.Submit("M2", PriorityTranslator)
	waitFor(t, "a1 running", func() bool { return taskStatus(s, a1) == TaskRunning })
	waitFor(t, "b1 running", func() bool { return taskStatus(s, b1) == TaskRunning })
	a2, _ := s.Submit("M1", PriorityTranslator)
	b2, _ := s.Submit("M2", PriorityTranslator)

	// 需要第 5 个模型的任务到来；M1、M2 有更早的排队任务，暂时受保护。
	x, _ := s.Submit("M3", PriorityTranslator)
	if n := rt.loadCount("M3"); n != 0 {
		t.Fatalf("M3 should not load while earlier queued tasks protect residents")
	}

	// M2 的同级新任务持续到来。
	b3, _ := s.Submit("M2", PriorityTranslator)
	b4, _ := s.Submit("M2", PriorityTranslator)

	// M1 的排队任务跑起来后，M1 不再受保护，被挑中腾位。
	rt.releaseInfer(a1)
	waitFor(t, "a2 running", func() bool { return taskStatus(s, a2) == TaskRunning })
	waitFor(t, "M1 unloading", func() bool {
		p, ok := modelPhase(s, "M1")
		return ok && p == ModelUnloading
	})
	rt.releaseInfer(a2)
	waitFor(t, "M1 unloaded", func() bool { return rt.unloadCount("M1") == 1 })
	waitFor(t, "M3 load started", func() bool { return rt.loadCount("M3") == 1 })

	// M3 加载完成后，X 拿到下一个空出的执行位（比 M2 的新任务靠前）。
	waitFor(t, "M3 ready", func() bool {
		p, ok := modelPhase(s, "M3")
		return ok && p == ModelReady
	})
	rt.releaseInfer(b1)
	waitFor(t, "x running", func() bool { return taskStatus(s, x) == TaskRunning })
	if st := taskStatus(s, b3); st == TaskRunning {
		t.Fatalf("later same-priority task should not jump ahead of x")
	}

	for _, id := range []uint64{x, b2, b3, b4} {
		rt.releaseInfer(id)
	}
	waitFor(t, "all terminal", func() bool {
		for _, id := range []uint64{a1, a2, b1, b2, b3, b4, x} {
			st := taskStatus(s, id)
			if st != TaskCompleted && st != TaskFailed && st != TaskCancelled {
				return false
			}
		}
		return true
	})
}

// 随机提交、取消、下线的混合压力测试，全程检查不变量。
func TestStressMixed(t *testing.T) {
	for round := 0; round < 3; round++ {
		t.Run(fmt.Sprintf("round%d", round), func(t *testing.T) {
			stressRound(t, int64(1000+round))
		})
	}
}

func stressRound(t *testing.T, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	const (
		numModels = 50
		numTasks  = 20000
		maxConc   = 8
		maxRes    = 4
		maxMem    = int64(60)
	)
	models := make([]string, numModels)
	for i := range models {
		models[i] = fmt.Sprintf("M%02d", i)
	}

	var inferMu sync.Mutex
	inferRan := make(map[uint64]bool)
	inferSuccess := 0
	var inferCur int32
	var inferErrAtomic int32
	var loadMu sync.Mutex
	loadRng := rand.New(rand.NewSource(seed + 7))

	hooks := Hooks{
		Load: func(_ context.Context, m string) error {
			loadMu.Lock()
			defer loadMu.Unlock()
			if loadRng.Intn(20) == 0 {
				return errors.New("load failed")
			}
			return nil
		},
		Infer: func(_ context.Context, _ string, id uint64) error {
			cur := atomic.AddInt32(&inferCur, 1)
			if cur > maxConc {
				t.Errorf("running %d exceeds %d", cur, maxConc)
			}
			inferMu.Lock()
			inferRan[id] = true
			inferMu.Unlock()
			if atomic.AddInt32(&inferErrAtomic, 0)%50 == 49 {
				atomic.AddInt32(&inferCur, -1)
				return errors.New("infer failed")
			}
			inferMu.Lock()
			inferSuccess++
			inferMu.Unlock()
			atomic.AddInt32(&inferCur, -1)
			return nil
		},
		Unload: func(string) {},
	}
	s := New(Config{MaxConcurrent: maxConc, MaxResident: maxRes, MaxMemory: maxMem, Hooks: hooks})
	mem := make(map[string]int64)
	for _, m := range models {
		mem[m] = 5 + rng.Int63n(20)
		if err := s.RegisterModel(m, mem[m]); err != nil {
			t.Fatalf("register %s: %v", m, err)
		}
	}

	// 不变量采样：整体画面必须始终满足驻留与执行位约束。
	var stopSample int32
	var sampleWG sync.WaitGroup
	sampleWG.Add(1)
	go func() {
		defer sampleWG.Done()
		for atomic.LoadInt32(&stopSample) == 0 {
			snap := s.Snapshot()
			if snap.Running > maxConc {
				t.Errorf("running %d exceeds %d", snap.Running, maxConc)
			}
			if len(snap.Models) > maxRes {
				t.Errorf("residents %d exceeds %d", len(snap.Models), maxRes)
			}
			var total int64
			for _, mi := range snap.Models {
				total += mi.Memory
				if mi.Running < 0 {
					t.Errorf("model %s running %d < 0", mi.Name, mi.Running)
				}
			}
			if total > maxMem {
				t.Errorf("resident memory %d exceeds %d", total, maxMem)
			}
			for p, q := range snap.Queued {
				if q < 0 {
					t.Errorf("queued[%v] = %d < 0", p, q)
				}
			}
		}
	}()

	// 提交、取消、下线并发进行。
	ids := make([]uint64, numTasks)
	var idMu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(seed + int64(w)*31))
			for i := w; i < numTasks; i += 8 {
				id, err := s.Submit(models[r.Intn(numModels)], Priority(r.Intn(2)))
				if err != nil {
					continue // 模型已下线，拒绝是预期行为
				}
				idMu.Lock()
				ids[i] = id
				idMu.Unlock()
			}
		}(w)
	}
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(seed + int64(w)*97))
			for i := 0; i < numTasks/2; i++ {
				s.Cancel(uint64(r.Intn(numTasks) + 1))
			}
		}(w)
	}
	// 随机下线一部分模型。
	offlined := make(map[string]bool)
	for i := 0; i < 10; i++ {
		m := models[rng.Intn(numModels)]
		offlined[m] = true
		wg.Add(1)
		go func(m string, d time.Duration) {
			defer wg.Done()
			time.Sleep(d)
			s.Offline(m)
		}(m, time.Duration(rng.Intn(20))*time.Millisecond)
	}
	wg.Wait()

	// 等所有已提交任务到达终态。
	idMu.Lock()
	submitted := make([]uint64, 0, numTasks)
	for _, id := range ids {
		if id != 0 {
			submitted = append(submitted, id)
		}
	}
	idMu.Unlock()
	waitFor(t, "all terminal", func() bool {
		for _, id := range submitted {
			switch taskStatus(s, id) {
			case TaskCompleted, TaskFailed, TaskCancelled:
			default:
				return false
			}
		}
		return true
	})
	atomic.StoreInt32(&stopSample, 1)
	sampleWG.Wait()

	// 每个任务恰好一个终态；已完成数等于成功的推理调用次数；
	// 报告已取消的任务从未进入推理。
	completed := 0
	for _, id := range submitted {
		info, ok := s.Task(id)
		if !ok {
			t.Fatalf("task %d missing", id)
		}
		if info.EndedAt.IsZero() {
			t.Fatalf("task %d terminal but no end time", id)
		}
		switch info.Status {
		case TaskCompleted:
			completed++
			if info.StartedAt.IsZero() {
				t.Fatalf("task %d completed but no start time", id)
			}
		case TaskCancelled:
			inferMu.Lock()
			ran := inferRan[id]
			inferMu.Unlock()
			if ran {
				t.Fatalf("cancelled task %d entered inference", id)
			}
			if !info.StartedAt.IsZero() {
				t.Fatalf("cancelled task %d has start time", id)
			}
		case TaskFailed:
		default:
			t.Fatalf("task %d not terminal: %v", id, info.Status)
		}
	}
	inferMu.Lock()
	success := inferSuccess
	inferMu.Unlock()
	if completed != success {
		t.Fatalf("completed=%d != successful infer calls=%d", completed, success)
	}
	t.Logf("submitted=%d completed=%d offlined=%d", len(submitted), completed, len(offlined))
}

// 加载和推理都立即返回时，10 万个任务 10 秒内全部跑完。
func TestThroughput100k(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	const (
		numModels = 50
		numTasks  = 100000
	)
	hooks := Hooks{
		Load:   func(context.Context, string) error { return nil },
		Infer:  func(context.Context, string, uint64) error { return nil },
		Unload: func(string) {},
	}
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 40, Hooks: hooks})
	for i := 0; i < numModels; i++ {
		mustRegister(t, s, fmt.Sprintf("M%02d", i), 10)
	}
	ids := make([]uint64, numTasks)
	start := time.Now()
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w; i < numTasks; i += 8 {
				id, err := s.Submit(fmt.Sprintf("M%02d", i%numModels), Priority(i%2))
				if err != nil {
					t.Errorf("submit: %v", err)
					return
				}
				ids[i] = id
			}
		}(w)
	}
	wg.Wait()
	waitFor(t, "all completed", func() bool {
		return s.Snapshot().Running == 0 &&
			s.Snapshot().Queued[PriorityTranslator] == 0 &&
			s.Snapshot().Queued[PriorityReviewer] == 0
	})
	elapsed := time.Since(start)
	for _, id := range ids {
		if st := taskStatus(s, id); st != TaskCompleted {
			t.Fatalf("task %d not completed: %v", id, st)
		}
	}
	if elapsed > 10*time.Second {
		t.Fatalf("100k tasks took %v, want < 10s", elapsed)
	}
	t.Logf("100k tasks in %v", elapsed)
}

// 排队变长时，全部命中已驻留模型的提交也不能明显变慢。
func TestThroughputAllResident(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	const numTasks = 100000
	hooks := Hooks{
		Load:   func(context.Context, string) error { return nil },
		Infer:  func(context.Context, string, uint64) error { return nil },
		Unload: func(string) {},
	}
	s := New(Config{MaxConcurrent: 8, MaxResident: 4, MaxMemory: 40, Hooks: hooks})
	for i := 0; i < 4; i++ {
		mustRegister(t, s, fmt.Sprintf("M%d", i), 10)
	}
	ids := make([]uint64, numTasks)
	start := time.Now()
	for i := 0; i < numTasks; i++ {
		id, err := s.Submit(fmt.Sprintf("M%d", i%4), Priority(i%2))
		if err != nil {
			t.Fatalf("submit: %v", err)
		}
		ids[i] = id
	}
	waitFor(t, "all completed", func() bool {
		snap := s.Snapshot()
		return snap.Running == 0 &&
			snap.Queued[PriorityTranslator] == 0 &&
			snap.Queued[PriorityReviewer] == 0
	})
	elapsed := time.Since(start)
	for _, id := range ids {
		if st := taskStatus(s, id); st != TaskCompleted {
			t.Fatalf("task %d not completed: %v", id, st)
		}
	}
	if elapsed > 10*time.Second {
		t.Fatalf("100k resident tasks took %v, want < 10s", elapsed)
	}
	t.Logf("100k resident tasks in %v", elapsed)
}
