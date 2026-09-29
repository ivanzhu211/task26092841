# prosa/sched

离线推理的任务调度与模型驻留模块。以库的形式被上层翻译服务调用，
状态全部保存在进程内存中，不落盘、不监听端口。

## 用法

```go
s := sched.New(sched.Config{
    MaxConcurrent: 8,          // 推理执行位
    MaxResident:   4,          // 驻留模型数上限
    MaxMemory:     64 << 30,   // 驻留模型声明内存合计上限
    Hooks: sched.Hooks{
        Load:   myLoad,   // func(ctx, model) error
        Infer:  myInfer,  // func(ctx, model, taskID) error
        Unload: myUnload, // func(model)
    },
})

s.RegisterModel("qwen-7b-q4", 8<<30) // 声明内存超上限会被拒绝

id, _ := s.Submit("qwen-7b-q4", sched.PriorityReviewer) // 审校员优先于译员
s.Cancel(id)        // 只对未开始的任务有效
s.Offline("qwen-7b-q4") // 下线：新提交拒绝，排队失败，在跑跑完后卸载

info, _ := s.Task(id) // 单任务：状态 + 创建/开始/结束时间
snap := s.Snapshot()  // 整体画面：各优先级排队数、在跑数、驻留模型状态
```

## 调度规则要点

- 派发顺序：高优先级在前，同级按提交先后；排头等加载不阻塞后面模型可用的任务。
- 缺模型的任务按派发顺序一次只安排一个加载；名额和内存不够就为它腾位置，
  腾出的资源只留给它，它的加载开始之前其他任务一律不开始加载。
- 腾位先挑没有任务在跑的驻留模型，再挑有任务在跑的，各自按最久没接新任务的在前；
  加载中的不挑，挑不够就等；挑中后不撤回。有更早排队任务的模型受保护，
  避免两个模型来回装卸却谁也跑不上。
- 模型从开始加载起算占用，卸载完成才释放；同一模型同一时刻只存在一份。
- 加载失败：等它的任务全部失败并带原因，资源释放，后续提交会重新加载。
- 推理失败只让该任务失败，不重试。

## 测试

```sh
go test ./sched/            # 全部测试（含 10 万任务吞吐）
go test ./sched/ -race -count=3
go test ./sched/ -short     # 跳过吞吐测试
```
