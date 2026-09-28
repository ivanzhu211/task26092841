package prosa

import (
	"container/heap"
	"errors"
	"sort"
	"sync"
	"time"
)

var (
	ErrModelExists       = errors.New("prosa: model already registered")
	ErrModelUnknown      = errors.New("prosa: model not registered")
	ErrModelOffline      = errors.New("prosa: model is offline")
	ErrMemoryTooLarge    = errors.New("prosa: declared memory exceeds limit")
	ErrInvalidConfig     = errors.New("prosa: invalid config")
	ErrNilHooks          = errors.New("prosa: hooks must be non-nil")
	ErrNonPositiveMemory = errors.New("prosa: declared memory must be positive")
)

// Scheduler schedules inference tasks and manages model residency.
// All state lives in process memory. Safe for concurrent use.
type Scheduler struct {
	cfg   Config
	hooks Hooks

	mu           sync.Mutex
	models       map[string]*model
	tasks        map[uint64]*task
	nextID       uint64
	running      int   // in-flight inferences
	residentN    int   // models counting against MaxModels
	residentMem  int64 // memory counting against MemoryLimit
	loadTarget   *model
	queuedByPrio map[Priority]int
}

// New creates a Scheduler. All hooks must be non-nil.
func New(cfg Config, hooks Hooks) (*Scheduler, error) {
	if cfg.MaxConcurrent <= 0 || cfg.MaxModels <= 0 || cfg.MemoryLimit <= 0 {
		return nil, ErrInvalidConfig
	}
	if hooks.Load == nil || hooks.Infer == nil || hooks.Unload == nil {
		return nil, ErrNilHooks
	}
	return &Scheduler{
		cfg:          cfg,
		hooks:        hooks,
		models:       make(map[string]*model),
		tasks:        make(map[uint64]*task),
		queuedByPrio: make(map[Priority]int),
	}, nil
}

// RegisterModel registers a model with its declared memory footprint.
func (s *Scheduler) RegisterModel(name string, declaredMem int64) error {
	if declaredMem <= 0 {
		return ErrNonPositiveMemory
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if declaredMem > s.cfg.MemoryLimit {
		return ErrMemoryTooLarge
	}
	if _, ok := s.models[name]; ok {
		return ErrModelExists
	}
	s.models[name] = &model{name: name, declared: declaredMem, state: ModelNotResident}
	return nil
}

// OfflineModel takes a model offline: new submissions are rejected,
// queued tasks for it fail, and the resident copy unloads once drained.
// It is idempotent for an already-offline model.
func (s *Scheduler) OfflineModel(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.models[name]
	if !ok {
		return ErrModelUnknown
	}
	if m.offline {
		return nil
	}
	m.offline = true
	now := time.Now()
	for m.pending.Len() > 0 {
		t := heap.Pop(&m.pending).(*task)
		s.failPendingLocked(t, "model "+name+" is offline", now)
	}
	if m.state == ModelAvailable {
		m.state = ModelPendingUnload
	}
	s.rescheduleLocked()
	return nil
}

// Submit enqueues a task and returns its ID.
func (s *Scheduler) Submit(modelName string, prio Priority, payload any) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.models[modelName]
	if !ok {
		return 0, ErrModelUnknown
	}
	if m.offline {
		return 0, ErrModelOffline
	}
	s.nextID++
	t := &task{
		id:       s.nextID,
		model:    modelName,
		priority: prio,
		payload:  payload,
		state:    TaskPending,
		created:  time.Now(),
	}
	s.tasks[t.id] = t
	heap.Push(&m.pending, t)
	s.queuedByPrio[prio]++
	s.rescheduleLocked()
	return t.id, nil
}

// Cancel cancels a pending task. It reports whether the task was
// canceled (true) or had already started or finished (false).
func (s *Scheduler) Cancel(id uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tasks[id]
	if !ok || t.state != TaskPending {
		return false
	}
	m := s.models[t.model]
	heap.Remove(&m.pending, t.heapIdx)
	t.state = TaskCanceled
	t.ended = time.Now()
	s.queuedByPrio[t.priority]--
	s.rescheduleLocked()
	return true
}

// Task returns a point-in-time view of one task.
func (s *Scheduler) Task(id uint64) (TaskInfo, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tasks[id]
	if !ok {
		return TaskInfo{}, false
	}
	return t.info(), true
}

// Snapshot returns a consistent view of the whole scheduler.
func (s *Scheduler) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap := Snapshot{
		QueuedByPriority: make(map[Priority]int, len(s.queuedByPrio)),
		Running:          s.running,
	}
	for p, n := range s.queuedByPrio {
		snap.QueuedByPriority[p] = n
	}
	for _, m := range s.models {
		if m.state != ModelNotResident {
			snap.Models = append(snap.Models, m.info())
		}
	}
	sort.Slice(snap.Models, func(i, j int) bool { return snap.Models[i].Name < snap.Models[j].Name })
	return snap
}

func (t *task) info() TaskInfo {
	return TaskInfo{
		ID:        t.id,
		Model:     t.model,
		Priority:  t.priority,
		State:     t.state,
		Err:       t.err,
		CreatedAt: t.created,
		StartedAt: t.started,
		EndedAt:   t.ended,
	}
}

// failPendingLocked moves a pending task to Failed. Caller holds s.mu.
func (s *Scheduler) failPendingLocked(t *task, reason string, now time.Time) {
	t.state = TaskFailed
	t.err = reason
	t.ended = now
	s.queuedByPrio[t.priority]--
}
