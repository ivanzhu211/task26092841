# prosa

离线推理的任务调度与模型驻留模块。以库的形式提供给上层翻译服务，
所有状态保存在进程内存中，不落盘、不监听端口。

## 功能

- 模型先注册（名称 + 声明内存）才能使用；声明内存超过上限拒绝注册。
- 加载 / 推理 / 卸载三个动作由上层在创建时以钩子注入，钩子可阻塞，
  调度器在独立 goroutine 中调用，提交、取消、下线、查看均立即返回。
- 任务带优先级（审校员 > 译员），派发顺序：高优先级在前，同级按提交先后。
- 最多 `MaxConcurrent` 个任务同时推理；驻留模型最多 `MaxModels` 个、
  声明内存合计不超过 `MemoryLimit`。模型从开始加载起占用名额，
  卸载完成才释放；同一模型同一时刻只存在一份，可同时服务多个任务。
- 有空闲执行位时，按派发顺序挑第一个模型可用的任务开始，
  不因排头任务在等模型而空等。
- 缺模型的任务按派发顺序安排加载：名额和内存够就立即加载；
  不够就为它腾位置——腾位置期间其他任务一律不开始加载，
  腾出的名额只留给它。驱逐先挑没有任务在跑的驻留模型，
  再挑有任务在跑的，各自按最久没接新任务的在前；驱逐决定不撤回，
  因此驻留模型被同级新任务持续占用时，等腾位置的任务也不会饿死。
- 加载失败：等该模型的任务全部失败并带原因，名额释放，
  之后的提交会重新加载。推理失败只让该任务失败，不重试。
- 取消只对未开始的任务有效；报告已取消的任务绝不会进入推理。
- 下线模型：新提交直接拒绝，排队任务立即失败，在跑的跑完后卸载，
  正在加载的加载完立即卸载。
- 每个任务最终恰好是已完成 / 失败 / 已取消之一，
  记录创建、开始、结束时间。`Snapshot` 返回同一时刻的一致整体画面。

## 用法

```go
s, err := prosa.New(prosa.Config{
    MaxConcurrent: 8,
    MaxModels:     4,
    MemoryLimit:   48 << 30,
}, prosa.Hooks{
    Load:   func(ctx context.Context, model string) error { ... },
    Infer:  func(ctx context.Context, model string, payload any) error { ... },
    Unload: func(ctx context.Context, model string) error { ... },
})

err = s.RegisterModel("qwen-7b-q4", 6<<30)
id, err := s.Submit("qwen-7b-q4", prosa.PriorityReviewer, payload)
ok := s.Cancel(id)              // true=已取消；false=已开始或已结束
info, ok := s.Task(id)          // 单任务状态与时间戳
snap := s.Snapshot()            // 各优先级排队数、在跑数、驻留模型状态
err = s.OfflineModel("qwen-7b-q4")
```

## 测试

```sh
go test ./...        # 确定性验收测试 + 随机混合压力测试 + 10 万任务性能测试
go test -race ./...  # 竞态检测
```
