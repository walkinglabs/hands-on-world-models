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

## A2：先证明模型能帮助行动，再进入 latent imagination

路径：

```text
notebooks/02_decision/A2-act-in-imagination.ipynb
```

A2 先完成一条容易检查的控制链。它从 PixelWorld 图片量出方块位置，学习 `position + action → next position`，在 learned dynamics 中用 beam search 试动作，再回到真实 PixelWorld 执行第一步。Notebook 会在同一组起点上比较 learned MPC 与随机动作的成功率和最终距离。

这条位置模型不是 Dreamer。它的作用是先给出一份下游证据：预测模型确实能被 Planner 使用。随后，A2 才把可解释位置换成 RSSM latent，从真实 posterior state 出发，让 Actor 采样 5 步动作，RSSM prior 推演 latent，Critic 给出 value，再计算 TD-λ target。

Notebook 会完成一次 Actor 与 Critic 更新，并检查参数确实改变。位置模型部分有真实环境控制结果；RSSM 部分仍只证明训练接口连通，不能写成 Dreamer-lite 已完成。

## Smoke 与 PA1 的区别

| 项目 | A1/A2 smoke                                 | PA1-A                                       |
| ---- | ------------------------------------------- | ------------------------------------------- |
| 数据 | 4 段 PixelWorld                             | 大一些 PixelWorld；选做 DMC                 |
| 训练 | 数十步                                      | 直到形成稳定曲线                            |
| 目的 | 检查控制增益、接口与梯度                    | 检查完整 latent policy 的 return 和样本效率 |
| 资源 | CPU                                         | 单张 24GB 目标                              |
| 结论 | 可解释 dynamics 能帮助行动；RSSM 接口可运行 | latent 模型与策略是否形成稳定闭环           |

运行神经 smoke：

```bash
python -m unittest tests.test_control tests.test_neural -v
```

完成两份 Notebook 后进入 [PA1-A](/assignments/pa1-a)。
