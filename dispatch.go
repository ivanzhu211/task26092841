package prosa

import (
	"container/heap"
	"context"
	"sort"
	"time"
)

// lessTask reports whether a dispatches before b:
// higher priority first, then earlier submission (lower id) first.
func lessTask(a, b *task) bool {
	if a.priority != b.priority {
		return a.priority > b.priority
	}
	return a.id < b.id
}

// rescheduleLocked advances the state machine after any event.
// Caller holds s.mu.
func (s *Scheduler) rescheduleLocked() {
	s.dispatchReadyLocked()
	s.maybeLoadLocked()
	s.maybeUnloadLocked()
}

// dispatchReadyLocked starts pending tasks whose model is available,
// in dispatch order, while execution slots remain.
func (s *Scheduler) dispatchReadyLocked() {
	for s.running < s.cfg.MaxConcurrent {
		var best *task
		for _, m := range s.models {
			if m.state != ModelAvailable || m.pending.Len() == 0 {
				continue
			}
			if t := m.pending[0]; best == nil || lessTask(t, best) {
				best = t
			}
		}
		if best == nil {
			return
		}
		m := s.models[best.model]
		heap.Pop(&m.pending)
		s.queuedByPrio[best.priority]--
		best.state = TaskRunning
		best.started = time.Now()
		s.running++
		m.running++
		m.lastUse = best.started
		go s.runInfer(m, best)
	}
}

// maybeLoadLocked starts loads for tasks that need a model, in dispatch
// order. Loads may overlap, but while a task is waiting for capacity to
// be freed for it (its load not yet started), no other load may start:
// the freed residency is reserved for the load target.
func (s *Scheduler) maybeLoadLocked() {
	for {
		if s.loadTarget != nil && (s.loadTarget.offline || s.loadTarget.pending.Len() == 0) {
			s.loadTarget = nil
		}
		if s.loadTarget != nil {
			m := s.loadTarget
			if s.fitsLocked(m) {
				s.loadTarget = nil
				s.startLoadLocked(m)
				continue
			}
			s.evictForLocked(m)
			return
		}
		t := s.earliestWaitingLocked()
		if t == nil {
			return
		}
		m := s.models[t.model]
		if s.fitsLocked(m) {
			s.startLoadLocked(m)
			return
		}
		s.loadTarget = m
		s.evictForLocked(m)
		return
	}
}

// earliestWaitingLocked returns the first task in dispatch order whose
// model is not resident at all. Tasks on draining models wait for the
// unload to finish and are reconsidered afterwards.
func (s *Scheduler) earliestWaitingLocked() *task {
	var best *task
	for _, m := range s.models {
		if m.offline || m.state != ModelNotResident || m.pending.Len() == 0 {
			continue
		}
		if t := m.pending[0]; best == nil || lessTask(t, best) {
			best = t
		}
	}
	return best
}

// fitsLocked reports whether m may start loading now.
func (s *Scheduler) fitsLocked(m *model) bool {
	return s.residentN < s.cfg.MaxModels && s.residentMem+m.declared <= s.cfg.MemoryLimit
}

// evictForLocked marks available models for unload until the target fits,
// preferring idle models, then least-recently-used ones. Already-draining
// models count as freed. If even evicting every candidate is not enough,
// nothing new is evicted and the scheduler waits for running tasks.
// Evictions, once made, are never withdrawn.
func (s *Scheduler) evictForLocked(target *model) {
	needN := s.residentN - (s.cfg.MaxModels - 1)
	needMem := s.residentMem + target.declared - s.cfg.MemoryLimit
	freeN := 0
	freeMem := int64(0)
	var cands []*model
	for _, m := range s.models {
		switch m.state {
		case ModelPendingUnload:
			freeN++
			freeMem += m.declared
		case ModelAvailable:
			cands = append(cands, m)
		}
	}
	if freeN >= needN && freeMem >= needMem {
		return
	}
	sort.Slice(cands, func(i, j int) bool {
		ri, rj := cands[i].running > 0, cands[j].running > 0
		if ri != rj {
			return rj // models without running tasks first
		}
		return cands[i].lastUse.Before(cands[j].lastUse)
	})
	var picked []*model
	for _, c := range cands {
		if freeN >= needN && freeMem >= needMem {
			break
		}
		picked = append(picked, c)
		freeN++
		freeMem += c.declared
	}
	if freeN < needN || freeMem < needMem {
		return // cannot free enough yet; wait
	}
	for _, c := range picked {
		c.state = ModelPendingUnload
	}
}

// maybeUnloadLocked starts unloads for drained models.
func (s *Scheduler) maybeUnloadLocked() {
	for _, m := range s.models {
		if m.state == ModelPendingUnload && m.running == 0 && !m.unloading {
			m.unloading = true
			go s.runUnload(m)
		}
	}
}

// startLoadLocked marks m as loading and spawns the load hook.
func (s *Scheduler) startLoadLocked(m *model) {
	m.state = ModelLoading
	s.residentN++
	s.residentMem += m.declared
	go s.runLoad(m)
}

func (s *Scheduler) runLoad(m *model) {
	err := s.hooks.Load(context.Background(), m.name)
	s.mu.Lock()
	defer s.mu.Unlock()
	if err != nil {
		m.state = ModelNotResident
		s.residentN--
		s.residentMem -= m.declared
		now := time.Now()
		for m.pending.Len() > 0 {
			t := heap.Pop(&m.pending).(*task)
			s.failPendingLocked(t, "load of model "+m.name+" failed: "+err.Error(), now)
		}
		s.rescheduleLocked()
		return
	}
	if m.offline {
		m.state = ModelPendingUnload
		s.rescheduleLocked()
		return
	}
	m.state = ModelAvailable
	s.rescheduleLocked()
}

func (s *Scheduler) runInfer(m *model, t *task) {
	err := s.hooks.Infer(context.Background(), m.name, t.payload)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.running--
	m.running--
	t.ended = time.Now()
	if err != nil {
		t.state = TaskFailed
		t.err = "infer on model " + m.name + " failed: " + err.Error()
	} else {
		t.state = TaskCompleted
	}
	s.rescheduleLocked()
}

func (s *Scheduler) runUnload(m *model) {
	_ = s.hooks.Unload(context.Background(), m.name)
	s.mu.Lock()
	defer s.mu.Unlock()
	m.unloading = false
	m.state = ModelNotResident
	s.residentN--
	s.residentMem -= m.declared
	s.rescheduleLocked()
}
