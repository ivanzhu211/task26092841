package sched

import (
	"container/list"
	"context"
	"errors"
	"sync"
	"time"
)

// ErrModelTooLarge 声明内存超过总上限，拒绝注册。
var ErrModelTooLarge = errors.New("sched: declared memory exceeds capacity")

// ErrModelNotRegistered 模型未注册。
var ErrModelNotRegistered = errors.New("sched: model not registered")

// ErrModelOffline 模型已下线，拒绝新提交。
var ErrModelOffline = errors.New("sched: model is offline")

// 优先级桶数量：译员、审校员。
const numPriorities = 2

func prioIndex(p Priority) int {
	if p >= PriorityReviewer {
		return 1
	}
	return 0
}

// task 内部任务记录。任务终身保留在 tasks 表中供查询。
type task struct {
	id     uint64
	model  string
	prio   Priority
	status TaskStatus
	err    error

	createdAt time.Time
	startedAt time.Time
	endedAt   time.Time

	globalElem *list.Element // 在 Scheduler.pending[prio] 中
	modelElem  *list.Element // 在 model.pending[prio] 中
	missing    bool          // 模型没驻留或待卸载
}

// model 内部模型记录。注册即存在；resident 表示占用驻留名额。
type model struct {
	name   string
	memory int64

	offline bool // 已下线：拒绝新提交

	resident      bool       // 是否占用驻留名额（加载中/可用/待卸载）
	phase         ModelPhase // 仅 resident 时有意义
	running       int        // 在该模型上跑的任务数
	lastAssigned  uint64     // 最近一次接新任务的序号，用于腾位排序
	unloadStarted bool       // Unload 钩子已发出，等待完成

	pending [numPriorities]*list.List // 等待该模型的任务（按提交先后）
}

func (m *model) available() bool {
	return m.resident && m.phase == ModelReady
}

// Scheduler 任务调度与模型驻留模块，所有方法并发安全。
type Scheduler struct {
	cfg Config

	mu           sync.Mutex
	nextID       uint64
	assignSeq    uint64 // 接新任务序号，腾位时按“最久没接新任务”排序
	tasks        map[uint64]*task
	models       map[string]*model
	pending      [numPriorities]*list.List // 全局排队（按提交先后）
	pendingCount [numPriorities]int
	missingCount int   // 排队中缺模型的任务数
	running      int   // 总在跑数
	residentCnt  int   // 驻留模型数（含加载中、待卸载）
	residentMem  int64 // 驻留声明内存合计
	loading      bool  // 有模型正在加载：一次只安排一个加载
	loadTarget   *task // 正在为其安排加载的任务，名额和内存只留给它
}

// New 创建模块。
func New(cfg Config) *Scheduler {
	s := &Scheduler{
		cfg:    cfg,
		tasks:  make(map[uint64]*task),
		models: make(map[string]*model),
	}
	for i := range s.pending {
		s.pending[i] = list.New()
	}
	return s
}

// RegisterModel 注册模型。声明内存超过总上限时返回 ErrModelTooLarge。
// 重复注册为无操作。
func (s *Scheduler) RegisterModel(name string, memory int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.models[name]; ok {
		return nil
	}
	if memory > s.cfg.MaxMemory {
		return ErrModelTooLarge
	}
	m := &model{name: name, memory: memory}
	for i := range m.pending {
		m.pending[i] = list.New()
	}
	s.models[name] = m
	return nil
}

// Submit 提交任务，立即返回任务 ID。
func (s *Scheduler) Submit(modelName string, prio Priority) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.models[modelName]
	if !ok {
		return 0, ErrModelNotRegistered
	}
	if m.offline {
		return 0, ErrModelOffline
	}
	s.nextID++
	t := &task{
		id:        s.nextID,
		model:     modelName,
		prio:      prio,
		status:    TaskPending,
		createdAt: time.Now(),
	}
	if isMissing(m) {
		t.missing = true
		s.missingCount++
	}
	pi := prioIndex(prio)
	t.globalElem = s.pending[pi].PushBack(t)
	t.modelElem = m.pending[pi].PushBack(t)
	s.pendingCount[pi]++
	s.tasks[t.id] = t
	s.dispatchLocked()
	return t.id, nil
}

// Cancel 取消尚未开始的任务。返回 true 表示已取消；
// 返回 false 表示任务已开始或已结束。
func (s *Scheduler) Cancel(id uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tasks[id]
	if !ok || t.status != TaskPending {
		return false
	}
	s.removePendingLocked(t)
	t.status = TaskCancelled
	t.endedAt = time.Now()
	if s.loadTarget == t {
		s.loadTarget = nil
	}
	s.dispatchLocked()
	return true
}

// Offline 下线模型：新提交拒绝，排队任务立即失败，
// 在跑的跑完后卸载，加载中的加载完立即卸载。
func (s *Scheduler) Offline(modelName string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.models[modelName]
	if !ok || m.offline {
		return
	}
	m.offline = true
	s.failPendingLocked(m, ErrModelOffline)
	if m.resident && m.phase == ModelReady {
		m.phase = ModelUnloading
		s.maybeUnloadLocked(m)
	}
	s.dispatchLocked()
}

// Task 查询单个任务。
func (s *Scheduler) Task(id uint64) (TaskInfo, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tasks[id]
	if !ok {
		return TaskInfo{}, false
	}
	return TaskInfo{
		ID:        t.id,
		Model:     t.model,
		Priority:  t.prio,
		Status:    t.status,
		Err:       t.err,
		CreatedAt: t.createdAt,
		StartedAt: t.startedAt,
		EndedAt:   t.endedAt,
	}, true
}

// Snapshot 查询整体画面，返回同一时刻的一致状态。
func (s *Scheduler) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap := Snapshot{
		Queued: map[Priority]int{
			PriorityTranslator: s.pendingCount[0],
			PriorityReviewer:   s.pendingCount[1],
		},
		Running: s.running,
	}
	for _, m := range s.models {
		if !m.resident {
			continue
		}
		snap.Models = append(snap.Models, ModelInfo{
			Name:    m.name,
			Phase:   m.phase,
			Memory:  m.memory,
			Running: m.running,
		})
	}
	return snap
}

// removePendingLocked 把任务从排队结构中移除。
func (s *Scheduler) removePendingLocked(t *task) {
	pi := prioIndex(t.prio)
	s.pending[pi].Remove(t.globalElem)
	s.models[t.model].pending[pi].Remove(t.modelElem)
	s.pendingCount[pi]--
	if t.missing {
		t.missing = false
		s.missingCount--
	}
}

// isMissing 报告模型对排队任务而言是否算缺：没驻留或待卸载。
func isMissing(m *model) bool {
	return !m.resident || m.phase == ModelUnloading
}

// markMissingLocked 模型驻留状态变化时，同步它所有排队任务的缺模型标记。
func (s *Scheduler) markMissingLocked(m *model, missing bool) {
	for pi := 0; pi < numPriorities; pi++ {
		for e := m.pending[pi].Front(); e != nil; e = e.Next() {
			t := e.Value.(*task)
			if t.missing == missing {
				continue
			}
			t.missing = missing
			if missing {
				s.missingCount++
			} else {
				s.missingCount--
			}
		}
	}
}

// failPendingLocked 让等待该模型的全部排队任务失败。
func (s *Scheduler) failPendingLocked(m *model, err error) {
	for pi := 0; pi < numPriorities; pi++ {
		for e := m.pending[pi].Front(); e != nil; {
			next := e.Next()
			t := e.Value.(*task)
			s.removePendingLocked(t)
			t.status = TaskFailed
			t.err = err
			t.endedAt = time.Now()
			if s.loadTarget == t {
				s.loadTarget = nil
			}
			e = next
		}
	}
}

// finishTaskLocked 推理结束，记录终态并触发后续调度。
func (s *Scheduler) finishTaskLocked(t *task, err error) {
	if err != nil {
		t.status = TaskFailed
		t.err = err
	} else {
		t.status = TaskCompleted
	}
	t.endedAt = time.Now()
	s.running--
	m := s.models[t.model]
	m.running--
	if m.resident && m.phase == ModelUnloading {
		s.maybeUnloadLocked(m)
	}
	s.dispatchLocked()
}

// runInfer 在执行位上跑推理，完成后回到临界区收尾。
func (s *Scheduler) runInfer(t *task) {
	err := s.cfg.Hooks.Infer(context.Background(), t.model, t.id)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.finishTaskLocked(t, err)
}
