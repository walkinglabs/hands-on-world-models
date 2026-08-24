# 运行证据与 24GB 验收

## 当前结论

仓库内 CPU/短训练 smoke 可运行；各章最后一份动手要求的完整神经网络训练，其 24GB 验收记录目前为 **0 个**。

仓库另保存一项 CPU toy 参考运行：`runs/reference/position-dynamics/`。它证明 learned dynamics 加 Planner 在 PixelWorld 上优于随机动作，但既不是完整的 Dreamer 闭环，也不构成 24GB 验收。

## 必填运行清单

每次完整训练在 `runs/<experiment>/manifest.json` 保存：

```json
{
  "experiment": "dreamer-loop-pixelworld",
  "route": "decision-and-planning",
  "seed": 0,
  "dataset": "pixelworld-v1",
  "split": "episode-seed-v1",
  "command": "python train.py ...",
  "started_at": "ISO-8601",
  "wall_time_seconds": 0,
  "device": "cuda",
  "gpu": "GPU exact name",
  "cuda": "version",
  "peak_allocated_mb": 0,
  "peak_reserved_mb": 0,
  "checkpoint_sha256": "...",
  "notes": ""
}
```

## 验收条件

- 从干净环境完成，不是截断 smoke；
- 单卡 peak reserved 不超过 22GB，给系统留出余量；
- 必做配方尽量不超过 24 小时；
- 提交训练/验证曲线、最终指标和失败图集；
- checkpoint 与数据 artifact 有 SHA256；
- 中断、OOM 和重跑必须保留记录。

只有满足以上条件，对应实验才算通过 24GB 验收。
