// Package sched 提供离线推理的任务调度与模型驻留管理。
// 状态全部保存在进程内存中，由上层翻译服务直接以库的方式调用。
package sched

import (
	"context"
	"time"
)

// Priority 任务优先级，数值越大优先级越高。
type Priority int

const (
	// PriorityTranslator 译员任务。
	PriorityTranslator Priority = 0
	// PriorityReviewer 审校员任务，高于译员。
	PriorityReviewer Priority = 1
)

// TaskStatus 任务状态。
type TaskStatus int

const (
	TaskPending   TaskStatus = iota // 排队中
	TaskRunning                     // 推理中
	TaskCompleted                   // 已完成
	TaskFailed                      // 失败
	TaskCancelled                   // 已取消
)

// ModelPhase 驻留模型状态。
type ModelPhase int

const (
	ModelLoading   ModelPhase = iota // 加载中
	ModelReady                       // 可用
	ModelUnloading                   // 待卸载（不再接新任务，跑完卸载）
)

// Hooks 由上层在创建模块时传入的三个动作。
type Hooks struct {
	// Load 加载模型，可能耗时几十秒。
	Load func(ctx context.Context, model string) error
	// Infer 执行一次推理，可能耗时几分钟。
	Infer func(ctx context.Context, model string, taskID uint64) error
	// Unload 卸载模型。
	Unload func(model string)
}

// Config 创建模块时的配置。
type Config struct {
	MaxConcurrent int   // 同时运行的任务上限（执行位）
	MaxResident   int   // 驻留模型数量上限
	MaxMemory     int64 // 驻留模型声明内存合计上限
	Hooks         Hooks
}

// TaskInfo 单个任务的查询结果。
type TaskInfo struct {
	ID        uint64
	Model     string
	Priority  Priority
	Status    TaskStatus
	Err       error
	CreatedAt time.Time
	StartedAt time.Time // 未开始为零值
	EndedAt   time.Time // 未结束为零值
}

// ModelInfo 单个驻留模型的状态。
type ModelInfo struct {
	Name    string
	Phase   ModelPhase
	Memory  int64
	Running int
}

// Snapshot 整体画面，是同一时刻的一致状态。
type Snapshot struct {
	Queued  map[Priority]int // 各优先级排队数
	Running int              // 总在跑数
	Models  []ModelInfo      // 各驻留模型
}
