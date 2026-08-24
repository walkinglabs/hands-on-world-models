# 参考运行证据

`runs/reference/` 只保存课程维护者实际运行过、可以由仓库脚本重新生成的轻量证据。它与学生自己的 `runs/<experiment>/` 分开。

- `position-dynamics/`：CPU toy 闭环，证明从图片测量的状态、learned dynamics 与 Planner 能改善 PixelWorld 行动；不是 Dreamer-lite，也不是 24GB 验收。

重新运行：

```bash
python scripts/run_position_dynamics_reference.py --output runs/reference/position-dynamics
```

每个目录至少包含 `metrics.json`、`manifest.json` 与 checkpoint。二进制 checkpoint 很小时可以提交；大型神经 checkpoint 只记录外部 artifact 地址与 SHA256。
