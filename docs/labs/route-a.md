# A1–A2 · 决策与规划实验

路线 A 第一次使用 PyTorch。共同基础仍只需要 NumPy；选择本路线后再安装神经依赖。

```bash
python -m pip install -r requirements-neural.txt
```

## A1：学习一个 latent world

路径：

```text
notebooks/02_decision/A1-learn-a-latent-world.ipynb
```

A1 使用 4 段项目内 PixelWorld episode 做 CPU smoke。它逐步检查：

```text
CNN embedding
→ RSSM prior/posterior
→ reconstruction/reward/continue/KL
→ 15 次更新
→ posterior 与 prior 的 shape
```

这不是完整 Dreamer 训练。可见结果是 loss 下降、各个 head 的数值和一次 prior/posterior 对照。

## A2：在想象中行动

路径：

```text
notebooks/02_decision/A2-act-in-imagination.ipynb
```

A2 从真实 posterior state 出发，让 Actor 采样 5 步动作，RSSM prior 推演 latent，Critic 给出 value，再计算 TD-λ target。

Notebook 会完成一次 Actor 与 Critic 更新，并检查参数确实改变。它只证明训练接口连通，不报告真实环境 return。

## Smoke 与 PA1 的区别

| 项目 | A1/A2 smoke     | PA1-A                       |
| ---- | --------------- | --------------------------- |
| 数据 | 4 段 PixelWorld | 大一些 PixelWorld；选做 DMC |
| 训练 | 数十步          | 直到形成稳定曲线            |
| 目的 | 检查接口与梯度  | 检查真实 return 和样本效率  |
| 资源 | CPU             | 单张 24GB 目标              |
| 结论 | 代码路径可运行  | 模型是否帮助行动            |

运行神经 smoke：

```bash
python -m unittest tests.test_neural -v
```

完成两份 Notebook 后进入 [PA1-A](/assignments/pa1-a)。
