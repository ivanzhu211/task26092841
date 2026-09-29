package sched

import (
	"context"
	"fmt"
	"sort"
	"time"
)

// 内部调度逻辑：派发、加载、腾位、卸载。
// 除特别说明外，本文件所有方法都必须在持有 s.mu 时调用。

// dispatchLocked 在任何状态变化后调用，推进派发与加载安排。
func (s *Scheduler) dispatchLocked() {
	s.startAvailableLocked()
	s.arrangeLoadLocked()
}

// startAvailableLocked 有空闲执行位时，按派发顺序挑模型可用的任务开始，
// 不因前面有任务在等模型而空等。
func (s *Scheduler) startAvailableLocked() {
	for s.running < s.cfg.MaxConcurrent {
		t := s.firstAvailableLocked()
		if t == nil {
			return
		}
		s.startTaskLocked(t)
	}
}

// firstAvailableLocked 返回派发顺序中第一个模型可用的任务。
func (s *Scheduler) firstAvailableLocked() *task {
	for pi := numPriorities - 1; pi >= 0; pi-- {
		var best *task
		for _, m := range s.models {
			if !m.available() {
				continue
			}
			e := m.pending[pi].Front()
			if e == nil {
				continue
			}
			t := e.Value.(*task)
			if best == nil || t.id < best.id {
				best = t
			}
		}
		if best != nil {
			return best
		}
	}
	return nil
}

// startTaskLocked 占用一个执行位开始任务。
func (s *Scheduler) startTaskLocked(t *task) {
	s.removePendingLocked(t)
	t.status = TaskRunning
	t.startedAt = time.Now()
	s.running++
	m := s.models[t.model]
	m.running++
	s.assignSeq++
	m.lastAssigned = s.assignSeq
	go s.runInfer(t)
}

// arrangeLoadLocked 为缺模型的任务安排加载：一次只安排一个，
// 名额和内存不够就为它腾位置，腾出的资源只留给它。
func (s *Scheduler) arrangeLoadLocked() {
	if s.loading {
		return
	}
	if s.loadTarget == nil {
		t := s.firstMissingLocked()
		if t == nil {
			return
		}
		s.loadTarget = t
	}
	target := s.models[s.loadTarget.model]
	s.evictForLocked(target)
	s.tryStartLoadLocked(target)
}

// firstMissingLocked 返回派发顺序中第一个缺模型的任务
// （模型没驻留或待卸载；加载中的不算缺，等它加载完即可）。
func (s *Scheduler) firstMissingLocked() *task {
	if s.missingCount == 0 {
		return nil
	}
	for pi := numPriorities - 1; pi >= 0; pi-- {
		for e := s.pending[pi].Front(); e != nil; e = e.Next() {
			t := e.Value.(*task)
			if !t.missing {
				continue
			}
			return t
		}
	}
	return nil
}

// evictForLocked 为 target 的加载腾出名额和内存。
// 先挑没有任务在跑的驻留模型，再挑有任务在跑的，各自按最久没接新任务的在前。
// 加载中的不挑；挑中后不撤回；挑不够就等。
// 有比 target 更早的排队任务的模型受保护：否则刚加载好的模型会在它的任务
// 拿到执行位之前就被另一个等腾位置的任务踢出去，两个模型来回装卸，谁也跑不上。
func (s *Scheduler) evictForLocked(target *model) {
	// 待卸载模型终将释放，按已释放计入。
	freeCnt, freeMem := 0, int64(0)
	for _, m := range s.models {
		if m.resident && m.phase == ModelUnloading {
			freeCnt++
			freeMem += m.memory
		}
	}
	needRoom := func() bool {
		return s.residentCnt-freeCnt+1 > s.cfg.MaxResident ||
			s.residentMem-freeMem+target.memory > s.cfg.MaxMemory
	}
	if !needRoom() {
		return
	}
	var cands []*model
	for _, m := range s.models {
		if !m.resident || m.phase != ModelReady || m == target {
			continue
		}
		if s.protectedLocked(m) {
			continue
		}
		cands = append(cands, m)
	}
	sort.Slice(cands, func(i, j int) bool {
		a, b := cands[i], cands[j]
		if (a.running == 0) != (b.running == 0) {
			return a.running == 0 // 没有在跑任务的在前
		}
		return a.lastAssigned < b.lastAssigned // 最久没接新任务的在前
	})
	for _, c := range cands {
		if !needRoom() {
			break
		}
		c.phase = ModelUnloading
		s.markMissingLocked(c, true)
		freeCnt++
		freeMem += c.memory
		s.maybeUnloadLocked(c)
	}
}

// protectedLocked 报告模型是否有比 loadTarget 更早的排队任务。
func (s *Scheduler) protectedLocked(m *model) bool {
	if s.loadTarget == nil {
		return false
	}
	for pi := 0; pi < numPriorities; pi++ {
		if e := m.pending[pi].Front(); e != nil {
			if e.Value.(*task).id < s.loadTarget.id {
				return true
			}
		}
	}
	return false
}

// tryStartLoadLocked 名额和内存够就为 loadTarget 的模型发起加载。
func (s *Scheduler) tryStartLoadLocked(target *model) {
	if target.resident {
		return // 旧副本仍在卸载，同一模型同一时刻只存在一份
	}
	if s.residentCnt+1 > s.cfg.MaxResident ||
		s.residentMem+target.memory > s.cfg.MaxMemory {
		return
	}
	target.resident = true
	target.phase = ModelLoading
	target.unloadStarted = false
	s.residentCnt++
	s.residentMem += target.memory
	s.loading = true
	go s.runLoad(target)
}

// runLoad 执行加载钩子，完成后回到临界区收尾。
func (s *Scheduler) runLoad(m *model) {
	err := s.cfg.Hooks.Load(context.Background(), m.name)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loading = false
	if !m.resident || m.phase != ModelLoading {
		return // 防御：状态已被其他路径改变
	}
	if s.loadTarget != nil && s.loadTarget.model == m.name {
		s.loadTarget = nil
	}
	if err != nil {
		// 加载失败：在等这个模型的任务全部失败并带上原因，
		// 名额和内存释放，之后提交的任务会重新加载。
		s.releaseResidentLocked(m)
		s.failPendingLocked(m, fmt.Errorf("sched: load model %s: %w", m.name, err))
		s.dispatchLocked()
		return
	}
	if m.offline {
		// 下线时正在加载的，加载完立即卸载。
		m.phase = ModelUnloading
		s.maybeUnloadLocked(m)
	} else {
		m.phase = ModelReady
		s.markMissingLocked(m, false)
		s.assignSeq++
		m.lastAssigned = s.assignSeq
	}
	s.dispatchLocked()
}

// maybeUnloadLocked 模型不再接新任务且没有在跑任务时发起卸载。
func (s *Scheduler) maybeUnloadLocked(m *model) {
	if !m.resident || m.phase != ModelUnloading || m.running > 0 || m.unloadStarted {
		return
	}
	m.unloadStarted = true
	go s.runUnload(m)
}

// runUnload 执行卸载钩子，完成后释放驻留名额和内存。
func (s *Scheduler) runUnload(m *model) {
	s.cfg.Hooks.Unload(m.name)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseResidentLocked(m)
	s.dispatchLocked()
}

// releaseResidentLocked 卸载完成，释放驻留名额和内存。
func (s *Scheduler) releaseResidentLocked(m *model) {
	m.resident = false
	m.unloadStarted = false
	s.residentCnt--
	s.residentMem -= m.memory
}
