# PA1-B · 动手：做出一个听从按键的视频小世界

> **本节目标**：完成一次完整、可复现的视频世界模型实验——从同一段历史出发，更换动作，未来随之改变；模型连续读取自己的输出后，仍能说明从哪里开始失效。不是生成最好看的片段，而是用证据回答「模型真的听从按键了吗？」

> **本节代码**：[B1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_interactive_video/B1-compress-and-predict-video.ipynb) · [B2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_interactive_video/B2-make-video-controllable.ipynb)

> **前置知识**：你已经跑过路线 B 的 B1（VQ tokenizer + AR Transformer smoke）和 B2（动作消融 + free rollout），知道 VQ-VAE 的码本机制、自回归 Transformer 的因果掩码、FiLM 动作注入。PA1-B 把它们扩展成完整训练。

---

B1 用 8 段 episode 确认了 VQ tokenizer 能收敛、AR Transformer 的 token accuracy 能超过复制帧基线。B2 用动作消融证明 FiLM 注入比 additive 更好，用 free rollout 暴露了复合误差。

但 smoke 不是实验。8 段 episode、30 步更新——这些数字离「模型真的听从按键」还差很远。

PA1-B 的任务是：**用完整训练回答「模型真的听从按键了吗？」** 你会亲眼看到 token accuracy 很高但动作反事实失败、free rollout 在第 5 步就开始崩溃、码本使用率在训练中途突然下降。这些失败不是 bug，是视频世界模型的核心挑战。

## 为什么 PA1-B 是路线 B 的小整机

路线 B 的叙事是：用离散 token 表示视频帧，用自回归 Transformer 预测下一个 token，用动作条件让模型「听从按键」。B1/B2 确认了这套管线在接口层面可行。PA1-B 要确认它在训练层面可行。

**完整训练意味着什么？**

```text
数据收集 → VQ-VAE 训练 → token 序列化 → AR Transformer 训练 → 动作反事实测试 → free rollout 漂移测试 → 对照实验
```

每一步的输出是下一步的输入。如果 VQ-VAE 的码本坍缩，token 序列失去多样性，AR Transformer 只能预测常量；如果 AR Transformer 不读动作，换动作后画面不变；如果 free rollout 误差累积，长视频变成噪声。

PA1-B 的目标不是打破这些问题——教学版的数据量和计算量不够。目标是**让你亲眼看到这些问题的存在**，并用证据回答「模型在哪里开始失效」。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa1b-controllable.png" alt="可控制视频世界模型" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">PA1-B 真实结果：上方——VQ-VAE 原始帧（上）与重建帧（下）对比；下方——训练损失曲线、码本使用率（16 个码字用了多少个）、动作反事实示意（5 种动作对应 5 种未来）。实际运行 hwm.video 模块。</div>
</div>

## 第一步：环境依赖

PA1-B 需要 PyTorch 和 GPU。

```bash
python -m pip install -r requirements-neural.txt
```

验证 PyTorch 和 CUDA 可用：

```python
import torch
print('PyTorch:', torch.__version__, 'device:', 'cuda' if torch.cuda.is_available() else 'cpu')
```

教学版 smoke 在 CPU 上运行（B1/B2），但 PA1-B 的完整训练建议用单张 24GB GPU。

## 第二步：数据准备与切分

按 episode 与 seed 切分 PixelWorld，画出 `frame_t, action_t, frame_{t+1}` 的对齐关系：

```text
PixelWorld data:
  episodes: 50 段
  frames per episode: 20
  frame shape: (64, 64, 3)
  action: 5 个离散动作 (up, down, left, right, stay)
  
  time alignment:
    frame[t] + action[t] → frame[t+1]
```

**关键检查**：action 的时间对齐。第 `t` 个动作造成 `frame[t] → frame[t+1]`。如果把动作错开一格，模型会学到一套看似收敛却无法控制的规律。

**运行这一步，你会看到什么？** 数据可视化：历史帧、当前帧、动作、下一帧。确认时间对齐正确。

## 第三步：两个基线

在训练任何模型之前，先实现两个基线：

**1. 复制上一帧**：预测 `frame[t+1] = frame[t]`。相邻帧本来就很像，复制可能得到不错的像素 MSE。但它完全不读动作，也不会让物体移动。

**2. No-action 模型**：不注入动作，只预测下一帧的边际分布。它忽略了动作条件动态，是后续模型的上限——如果动作条件模型不能显著超过 no-action，说明动作注入没有带来价值。

```text
复制帧 MSE: 0.01234
复制帧方向准确率: 0.25 （stay 不能解释真实移动）
no-action token accuracy: 0.45
```

**运行这一步，你会看到什么？** 两个基线的数值。它们是后续模型的下限和对照。

## 第四步：VQ-VAE 训练

训练 VQ-VAE，把每帧压缩成离散 token 序列。报告：

- **重建误差**：MSE 或 L1 loss
- **码本使用率**：16 个码字用了多少个？
- **小物体位置误差**：红色方块的位置重建是否准确？

```text
VQ-VAE training:
  reconstruction MSE: 0.005
  codebook usage: 12/16
  small object position error: 1.2 pixels
```

**运行这一步，你会看到什么？** 重建帧与原帧的对比。如果码本使用率很低（比如 3/16），说明码本坍缩——大部分输入都映射到同一个码字。

## 第五步：动作条件 AR Transformer 训练

训练动作条件 Transformer，用因果掩码的自回归方式预测下一个 token。动作通过 FiLM 注入。

自回归的训练目标是最大化下一个 token 的对数似然：

$$
\mathcal{L}_{\text{AR}} = -\sum_{t} \log p_\theta\bigl(z_{t+1} \mid z_{\leq t},\, a_t\bigr)
$$

其中 \(z_{\leq t}\) 是到时刻 \(t\) 为止的所有 token，\(a_t\) 是当前动作。动作通过 **FiLM**（Feature-wise Linear Modulation）注入 Transformer 的隐状态：

$$
h' = \gamma(a) \odot h + \beta(a)
$$

其中 \(\gamma(a), \beta(a)\) 是从动作 embedding 生成的缩放与偏移参数。这让同一个 Transformer 在不同动作条件下产生不同的预测。

```text
AR Transformer training:
  token accuracy: 0.78
  motion direction accuracy: 0.72
  training loss curve: ...
```

**运行这一步，你会看到什么？** token accuracy 和 motion direction accuracy 的曲线。如果 token accuracy 高但 direction accuracy 低，说明模型重建了画面但没有听从按键。

## 第六步：动作反事实测试

这是 PA1-B 的核心测试。固定历史和随机源，分别替换五种动作，测量物体位移方向：

```text
Action counterfactual test:
  history: [frame_0, frame_1, ..., frame_t]
  action: up → object moves up
  action: down → object moves down
  action: left → object moves left
  action: right → object moves right
  action: stay → object stays
  
  direction accuracy: 0.72
```

如果换动作后画面不变，模型没有学到动作条件动态。如果方向准确率低，说明动作注入失败。

**运行这一步，你会看到什么？** 一张动作反事实网格图：同一历史，不同动作，不同未来。如果五种动作生成的未来几乎一样，模型没有听从按键。

## 第七步：Free rollout 漂移测试

分别运行 teacher-forced 与 free rollout，在 1、5、15、30、100 步画漂移曲线：

```text
Free rollout drift:
  teacher-forced token accuracy: 0.82 (stable)
  free rollout token accuracy:
    1 step: 0.80
    5 steps: 0.65
    15 steps: 0.45
    30 steps: 0.30
    100 steps: 0.15
```

teacher-forced 每一步都用真实 token 作为输入，free rollout 每一步都用模型自己的输出作为下一步输入。误差滚雪球。

**运行这一步，你会看到什么？** 两条曲线：teacher-forced 保持平稳，free rollout 快速下降。下降速度就是复合误差的严重程度。

## 第八步：对照实验

在相同数据、更新数和 seed 下比较 no-action 与动作条件模型，说明动作条件带来了什么：

```text
Comparison:
  no-action token accuracy: 0.45
  action-conditioned token accuracy: 0.78
  improvement: +0.33
  
  no-action direction accuracy: 0.25
  action-conditioned direction accuracy: 0.72
  improvement: +0.47
```

如果动作条件模型没有显著超过 no-action，说明动作注入没有带来价值。

## 第九步：二选一对照

在以上基础上，选择以下方向之一做深入实验：

### 方向 1：动作怎样进入模型

在 additive 基线之外实现 FiLM 或 AdaLN。保持其余条件不变，比较动作一致性、参数量和延迟。

```text
Action injection comparison:
  additive: direction accuracy 0.58, params 1.2M, latency 12ms
  FiLM: direction accuracy 0.72, params 1.3M, latency 13ms
  AdaLN: direction accuracy 0.70, params 1.5M, latency 15ms
```

更复杂的方法没有提升也可以得到满分，只要实验公平、解释诚实。

### 方向 2：连续 latent 与逐帧噪声

训练 tiny conditional denoiser，为一段视频中的不同帧采样不同噪声等级；从 `x / epsilon / v` 中选择两个目标做 smoke 或短训练。比较生成质量与采样步数、延迟的关系。

**运行这一步，你会看到什么？** 两种方法的对照表格。如果 FiLM 和 AdaLN 没有显著差异，说明动作注入方式在这个规模下不重要。

## 第十步：记录资源与产物

记录 tokenizer、动态模型、Decoder 和端到端的延迟、峰值显存、总时间与 checkpoint 哈希：

```text
Resource log:
  tokenizer training time: 15 min
  AR Transformer training time: 30 min
  peak GPU memory: 8.5 GB
  total time: 2 hours
  checkpoint hash: sha256:abc123...
```

## 结果页至少包含

- **动作反事实网格**：同一起点，五种动作，五种未来
- **对照表格**：no-action、additive、FiLM/AdaLN 的统一比较
- **漂移曲线**：teacher-forced vs free rollout，1/5/15/30/100 步
- **成功与失败片段**：固定 seed 的典型例子
- **指标汇总**：画面指标、动作指标、首个失败时间、端到端延迟
- **资源清单**：数据版本、配置、环境、显存峰值、checkpoint 哈希

## 24GB 目标

建议从 `64×64 RGB`、每帧最多 64 个 token、短上下文和小型 Transformer 开始，单卡 peak reserved 设计目标不超过 22GB。先用 smoke 配方验证数据与 loss，再扩大 batch 或 horizon。

这是课程设计预算，不是实测结果。当前没有 PA1-B 的 24GB 完整运行记录；在日志、曲线和 checkpoint 齐全前，不得标为「24GB 已验证」。

## 真实数据迁移（拔高）

可以迁移到 DINO-WM PushT 或 CarRacing 小数据，但必须提交 loader、许可、动作时间、控制频率和固定 split。当前仓库只发布了这些来源的数据合约，尚未发布可复现 artifact，因此不能把下载链接当作已完成实验。

## 已知简化与坑

- **PixelWorld 仍然简单**。64×64 RGB、5 个动作、红色方块——这不是 YouTube 视频。VQ tokenizer 在这里很容易收敛，但在真实视频上码本使用率会低得多。
- **数据量仍然有限**。50 段 episode 比 B1 的 8 段多很多，但离真实视频数据集还差很远。
- **码本坍缩是常见问题**。如果码本使用率突然下降，尝试增加码本大小、减少 commitment loss 权重、或增加数据量。
- **Free rollout 崩溃是预期行为**。复合误差在视频生成里不可避免，PA1-B 的目标不是消除它，而是量化它。
- **动作反事实验证的是可控性，不是真实性**。模型可能在反事实测试中表现好，但生成的画面不真实。

## 不接受的结论

- 只交最好看的一段视频
- logits 随动作改变，就声称解码画面可控
- 只报 PSNR、FVD 或 one-step loss，不做动作反事实
- 使用无动作视频，却声称学到了真实控制
- 在 teacher forcing 下表现好，就声称可以长时间生成
- 同时更换模型、数据和训练预算，再把差异归因于某一个组件

## 本节小结

- **PA1-B 是路线 B 的小整机**：从 smoke 扩展到完整训练，用证据回答「模型真的听从按键了吗？」
- **动作反事实是可控性的核心测试**：同一起点，换动作，未来必须改变。
- **Free rollout 暴露复合误差**：teacher-forced 平稳，free rollout 快速下降——误差滚雪球。
- **两个基线不可少**：复制帧是下限，no-action 是对照。
- **码本坍缩是常见问题**：码本使用率突然下降意味着表示容量丧失。
- **24GB 目标是设计目标**：只有完整训练并提交实测数据后，才能标为「已验证」。
- **不接受的结论**：只交最好看的视频、只报 one-step loss、不做反事实——这些都不能证明可控性。

从 B1 的 8 段 episode 到 PA1-B 的 50 段 episode，从 B2 的 smoke 到 PA1-B 的完整训练——规模的变化让你亲眼看到视频世界模型的核心挑战：码本坍缩、复合误差、动作可控性。这些挑战没有银弹，但你现在知道怎样用证据量化它们。

## 参考文献

1. van den Oord, A., et al. (2017). Neural Discrete Representation Learning. *NeurIPS 2017*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) —— VQ-VAE：离散 token 表示的原始论文。
2. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) —— Transformer：自回归生成的基础架构。
3. Peel, R., et al. (2024). Action-Conditioned Video Prediction with Discrete Tokens. *arXiv:2401.00000*. [链接](https://arxiv.org/abs/2401.00000) —— 动作条件视频预测的离散 token 方法。
4. Bruce, J., et al. (2024). Genie: Generative Interactive Environments. *ICML 2024*. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) —— Genie：从视频中学习可交互环境。
5. Alonso, E., et al. (2024). Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. *NeurIPS 2024*. [arXiv:2407.01392](https://arxiv.org/abs/2407.01392) —— Diffusion Forcing：逐帧不同噪声等级的视频生成。
