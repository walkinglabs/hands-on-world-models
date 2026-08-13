# B1–C2 · 互动视频与 JEPA 实验

路线 B 与 C 使用同一份 PixelWorld 数据。路线 B 要还原可观看画面，路线 C 只预测特征；共用数据能让两种目标直接比较。

## 路线 B：两份 Notebook

### B1　做出第一台视频模型

`B1-compress-and-predict-video.ipynb` 依次完成：

```text
检查动作时间 → 复制帧基线 → VQ tokenizer → token AR → 一步画面评价
```

这一份先把最短闭环跑通。你会同时看到重建 loss、码本使用数、token accuracy 和解码后物体方向。某一个数字变好，不代表整台模型已经可控。

### B2　让模型连续运行，再拆开比较

`B2-make-video-controllable.ipynb` 只增加三个困难：

```text
动作注入消融 → teacher-forced / free rollout → 逐帧不同噪声
```

动作消融使用 `no-action / additive / FiLM`；生成实验固定同一历史替换动作；Diffusion Forcing 小节为视频中每一帧分别指定噪声等级，并构造 `x / epsilon / v` 目标。它只验证接口，不冒充大型系统复现。

CPU smoke 证明 shape、梯度和数据流可以运行。PA1-B 还要保存曲线、失败片段、端到端延迟和 checkpoint。

## 路线 C：两份 Notebook

`C1-learn-video-features.ipynb`：

```text
video patch → online/target encoder → mask → EMA → collapse 检查
```

`C2-test-and-control-features.ipynb`：

```text
linear probe → action conditioning → 反事实 feature → 一步动作选择
```

C2 的 probe 按 episode seed 分开训练与测试。被动视频与动作条件结果也要分开报告：没有动作的数据不能证明 controllability。

## 运行

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_routes_bc -v
```

完成一条路线即可进入对应 PA，不要求同时完成 B 与 C。
