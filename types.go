package prosa

import (
	"context"
	"time"
)

// Priority of a submitted task. Higher value dispatches earlier.
type Priority int

const (
	// PriorityTranslator is the regular priority for translator tasks.
	PriorityTranslator Priority = 0
	// PriorityReviewer is the higher priority for reviewer tasks.
	PriorityReviewer Priority = 1
)

// TaskState is the lifecycle state of a task.
type TaskState int

const (
	TaskPending   TaskState = iota // queued, not started
	TaskRunning                    // inference in flight
	TaskCompleted                  // inference succeeded
	TaskFailed                     // load/inference failed or model offlined
	TaskCanceled                   // canceled before start
)

// ModelState is the residency state of a model.
type ModelState int

const (
	ModelNotResident ModelState = iota
	ModelLoading
	ModelAvailable
	ModelPendingUnload // evicted or offlined, draining or unloading
)

// Config tunes the scheduler limits.
type Config struct {
	MaxConcurrent int   // max simultaneous inferences (execution slots)
	MaxModels     int   // max resident models (loading counts, released after unload)
	MemoryLimit   int64 // max sum of declared memory of resident models
}

// Hooks are supplied by the caller; all must be non-nil.
// They must be safe to call concurrently and may block.
type Hooks struct {
	Load   func(ctx context.Context, model string) error
	Infer  func(ctx context.Context, model string, payload any) error
	Unload func(ctx context.Context, model string) error
}

// TaskInfo is a point-in-time view of one task.
type TaskInfo struct {
	ID        uint64
	Model     string
	Priority  Priority
	State     TaskState
	Err       string // failure reason when State == TaskFailed
	CreatedAt time.Time
	StartedAt time.Time // zero until inference starts
	EndedAt   time.Time // zero until a terminal state
}

// ModelSnapshot is a point-in-time view of one resident model.
type ModelSnapshot struct {
	Name           string
	State          ModelState
	DeclaredMemory int64
	Running        int
}

// Snapshot is a consistent view of the whole scheduler.
type Snapshot struct {
	QueuedByPriority map[Priority]int
	Running          int
	Models           []ModelSnapshot
}
