package prosa

import "time"

// task is the internal mutable record for one submitted task.
type task struct {
	id       uint64
	model    string
	priority Priority
	payload  any
	state    TaskState
	err      string
	created  time.Time
	started  time.Time
	ended    time.Time
	heapIdx  int // index inside its model's pending heap, -1 when removed
}

// model is the internal record for a registered model.
type model struct {
	name     string
	declared int64
	offline  bool

	state     ModelState
	running   int       // in-flight inferences on this model
	lastUse   time.Time // last time a task started on this model
	unloading bool      // an unload hook call is in flight

	pending taskHeap // queued tasks waiting for this model
}

func (m *model) info() ModelSnapshot {
	return ModelSnapshot{
		Name:           m.name,
		State:          m.state,
		DeclaredMemory: m.declared,
		Running:        m.running,
	}
}
