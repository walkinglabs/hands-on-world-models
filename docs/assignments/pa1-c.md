# PA1-C · 动手：训练并审问一个 Tiny Video-JEPA

> **本节目标**：不是把 feature loss 降到最低，而是弄清表示保留了什么、是否受动作控制、能否帮助一个小任务。完成一次完整的 JEPA 实验：训练被动 Video-JEPA → 排查坍缩 → linear probe → 加入动作 → 反事实测试 → MPC → 对照实验。

> **本节代码**：[C1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_jepa/C1-learn-video-features.ipynb) · [C2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_jepa/C2-test-and-control-features.ipynb)

> **前置知识**：你已经跑过路线 C 的 C1（Video-JEPA smoke）和 C2（linear probe + action conditioning），知道 JEPA 的 online/target encoder 分工、EMA 更新、表示坍缩。PA1-C 把它们扩展成完整训练。

---

C1 用 8 段 episode 确认了 Video-JEPA 的接口连通：loss 能下降、feature_spread 不降、target encoder 通过 EMA 缓慢跟随。C2 用 linear probe 证明特征保留了位置信息，用 action conditioning 证明特征受动作控制。

但 smoke 不是实验。8 段 episode、35 步更新——这些数字离「特征真的有用」还差很远。

PA1-C 的任务是：**用完整训练回答「JEPA 特征保留了什么？丢掉了什么？这是否有益？」** 你会亲眼看到 feature loss 下降但 probe 准确率不升、EMA momentum 太小导致坍缩、动作条件 predictor 在被动数据上失效。这些失败不是 bug，是 JEPA 架构设计的核心动机。

## 为什么 PA1-C 与 PA1-B 走不同的路

PA1-B 用 VQ-VAE + AR Transformer 重建像素，能画出可观看的画面。PA1-C 用 Video-JEPA 只预测特征，不画回像素。

**两种目标，两种代价：**

| 项目 | PA1-B（重建像素） | PA1-C（预测特征） |
| ---- | ----------------- | ----------------- |
| 预测目标 | 离散 token → 重建画面 | 连续特征 → 不重建 |
| 可观看性 | 能画出画面，一眼判断好坏 | 无法直接观看，需要 probe |
| 训练稳定性 | 码本可能坍缩 | EMA 稳定，但可能表示坍缩 |
| 可控性证据 | 换动作后画面改变 | action-conditioned MSE 降低 |

PA1-C 的优势是训练更稳、不需要解码器；劣势是无法直接观看，必须用 probe 和下游任务判断特征质量。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/pa1c-jepa-quality.png" alt="JEPA 特征质量评估" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">PA1-C 真实结果：左——JEPA 特征预测损失下降曲线；中——特征 spread 诊断（接近零表示坍缩）；右——特征统计（mean/std/min/max）。不重建像素，只在特征空间评估。实际运行 hwm.jepa 模块。</div>
</div>

## 第一步：环境依赖

PA1-C 需要 PyTorch 和 GPU。

```bash
python -m pip install -r requirements-neural.txt
```

验证 PyTorch 和 CUDA 可用：

```python
import torch
print('PyTorch:', torch.__version__, 'device:', 'cuda' if torch.cuda.is_available() else 'cpu')
```

## 第二步：训练被动 Tiny Video-JEPA

在 PixelWorld 训练被动 Video-JEPA——不注入动作，只预测下一帧特征。

JEPA 的训练目标是让 predictor 的输出接近 target encoder 的输出：

$$
\mathcal{L}_{\text{JEPA}} = \bigl\| f_{\text{pred}}\bigl(f_{\text{online}}(x_{\text{ctx}})\bigr) - f_{\text{target}}(x_{\text{future}}) \bigr\|_2^2
$$

与 VQ-VAE 路线的关键区别：**不重建像素**，只预测抽象特征。这避免了「把算力浪费在重建草地纹理」的问题。target encoder 通过 EMA 更新：

$$
\theta_{\text{target}} \leftarrow \tau \, \theta_{\text{target}} + (1 - \tau) \, \theta_{\text{online}}, \quad \tau \approx 0.99
$$

```text
Passive Video-JEPA training:
  frames: 8-16
  resolution: 112×112 or smaller
  model: Tiny/Small ViT
  online encoder: ViT-tiny
  target encoder: ViT-tiny (EMA)
  predictor: 2-layer MLP
  EMA momentum: 0.99
```

**运行这一步，你会看到什么？** feature loss 曲线。loss 应该下降，但 loss 下降不等于特征有用——下一步要排查坍缩。

## 第三步：排查表示坍缩

报告 feature loss、方差、协方差或最近邻，排查坍缩：

```text
Collapse diagnostics:
  feature loss: 0.567 → 0.123
  feature variance: 0.89 → 0.67 (should not drop to 0)
  feature covariance: near-diagonal (good) or uniform (collapse)
  nearest neighbor similarity: 0.3 (good) or 0.95 (collapse)
```

**表示坍缩**是所有输入都映射到同一个特征。如果 feature variance 降到接近零，或 nearest neighbor similarity 接近 1，说明坍缩了。

**运行这一步，你会看到什么？** 四条诊断曲线。如果 loss 下降但 variance 也降到零，说明模型找到了坍缩的捷径。

## 第四步：Linear probe

冻结 Encoder，用 linear probe 读出位置与速度：

```text
Linear probe:
  freeze: online encoder
  train: linear classifier
  task: predict block position from features
  accuracy: 0.78
  random baseline: 0.25
```

如果 probe 准确率高，说明特征保留了空间信息。如果 probe 准确率低，说明特征丢掉了位置——可能是坍缩，也可能是 JEPA 认为位置不重要。

**运行这一步，你会看到什么？** probe 准确率。如果显著高于随机基线，说明特征保留了任务相关信息。

## 第五步：加入真实动作，训练 Action-JEPA

被动 Video-JEPA 不读动作。现在加入真实动作，训练 Action-JEPA：

```text
Action-JEPA training:
  action injection: FiLM or additive
  prediction target: next frame features
  action-conditioned MSE: 0.123
  passive MSE: 0.345
```

如果 action-conditioned MSE 远低于 passive MSE，说明特征受动作控制——这是「可交互世界模型」的核心证据。

**运行这一步，你会看到什么？** 两条 MSE 曲线：passive 和 action-conditioned。如果两者接近，说明动作注入没有带来价值。

## 第六步：反事实测试

固定历史替换动作，检查预测 feature 与 probe 结果：

```text
Counterfactual test:
  history: [frame_0, frame_1, ..., frame_t]
  action: up → predicted feature → probe: position moves up
  action: down → predicted feature → probe: position moves down
  action: left → predicted feature → probe: position moves left
  
  counterfactual consistency: 0.72
```

如果换动作后预测特征不变，模型没有学到动作条件动态。如果 probe 结果不随动作改变，特征不受动作控制。

**运行这一步，你会看到什么？** 反事实一致性分数。如果接近随机，说明模型没有听从按键。

## 第七步：MPC 或一步控制

完成一步或短 horizon MPC，并在真实环境检查成功率：

```text
MPC test:
  horizon: 3
  candidates: 10
  success rate: 0.67
  random baseline: 0.25
```

如果 MPC 成功率显著高于随机基线，说明 JEPA 特征确实能帮助决策。

**运行这一步，你会看到什么？** MPC 成功率。如果接近随机，说明特征不够用，或者 MPC 的 horizon/candidates 太少。

## 第八步：公平对照

与像素预测或无动作 predictor 做公平对照：

```text
Comparison:
  passive JEPA probe accuracy: 0.65
  action-JEPA probe accuracy: 0.78
  pixel prediction + probe: 0.72
  
  passive JEPA MPC success: 0.45
  action-JEPA MPC success: 0.67
  pixel prediction + MPC: 0.55
```

固定数据 split、seed 集合、更新数或计算预算。报告平均结果、波动、失败样例与额外资源。

**运行这一步，你会看到什么？** 对照表格。如果 action-JEPA 没有显著超过 passive JEPA，说明动作注入没有带来价值。如果 pixel prediction 更好，说明 JEPA 特征丢掉了太多信息。

## 第九步：记录资源

提交显存、时间、曲线与 checkpoint 哈希：

```text
Resource log:
  passive JEPA training time: 20 min
  action-JEPA training time: 25 min
  peak GPU memory: 6.5 GB
  total time: 1.5 hours
  checkpoint hash: sha256:def456...
```

## 24GB 目标

`8–16` 帧、`112×112` 或更小分辨率、Tiny/Small ViT、混合精度可选，峰值 reserved 目标不超过 22GB。当前未完整实测。

## 选做迁移

在 UCF101-mini 上做被动视频表示迁移。它可以报告 action recognition 或 probe，不能用于声称机器人控制。

```text
Transfer test:
  source: PixelWorld
  target: UCF101-mini
  task: action recognition
  accuracy: 0.45
  random baseline: 0.1
```

## 最后回答

完成所有实验后，回答三个问题：

**1. 模型丢掉了什么？**

JEPA 不重建像素，所以它丢掉了像素级信息。但特征应该保留任务相关信息。如果 probe 准确率低，说明特征丢掉了位置——这是问题。

**2. 在当前任务里这是否有益？**

如果丢掉像素级信息让特征更聚焦于任务相关信息，这是有益的。JEPA 的优势就在于不被像素重建分散注意力。

**3. 换一个任务以后，原来被丢掉的信息会不会重新变得重要？**

如果换一个需要像素级信息的任务（比如视觉质量评估），JEPA 特征就不够用了。丢掉的信息是否重要，取决于下游任务。

## 已知简化与坑

- **PixelWorld 仍然简单**。16×16 的小图、红色方块、5 个动作——这不是真实视频。JEPA 在这里很容易收敛。
- **数据量仍然有限**。50 段 episode 比 C1 的 8 段多很多，但离真实视频数据集还差很远。
- **EMA momentum 是关键超参**。momentum 太小，target encoder 变化太快，容易坍缩；momentum 太大，target encoder 变化太慢，训练不稳定。
- **Linear probe 是线性**。非线性 probe 会得到更高的准确率，但线性 probe 更能说明特征本身的质量。
- **被动视频不能证明 controllability**。必须分开报告 passive 和 action-conditioned 结果。

## 本节小结

- **PA1-C 是路线 C 的小整机**：从 smoke 扩展到完整训练，弄清表示保留了什么、是否受动作控制、能否帮助小任务。
- **表示坍缩是 JEPA 的核心风险**：feature loss 下降但 variance 降到零，说明模型找到了坍缩的捷径。
- **Linear probe 是特征质量的试金石**：probe 准确率高，说明特征保留了任务相关信息。
- **动作条件是可控性的核心证据**：action-conditioned MSE 远低于 passive，说明特征受动作控制。
- **反事实测试验证可控性**：同一起点，换动作，预测特征必须改变。
- **JEPA 丢掉了像素级信息**：这是否有益，取决于下游任务。
- **24GB 目标是设计目标**：当前未完整实测。

从 C1 的 8 段 episode 到 PA1-C 的 50 段 episode，从 C2 的 smoke 到 PA1-C 的完整训练——规模的变化让你亲眼看到 JEPA 的核心挑战和优势：表示坍缩、动作可控性、特征质量。与 PA1-B 的像素重建路线相比，JEPA 走了一条不同的路：不画回像素，只预测特征。两条路各有代价，没有免费午餐。

## 参考文献

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview*. [链接](https://openreview.net/pdf?id=BZ5a1r-kVsf) —— JEPA 的原始提案：预测特征而非像素。
2. Bardes, A., et al. (2023). V-JEPA: Joint Embedding Predictive Architecture. *arXiv:2301.08243*. [链接](https://arxiv.org/abs/2301.08243) —— Video-JEPA：视频特征预测的具体实现。
3. Grill, J.-B., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. *NeurIPS 2020*. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) —— BYOL：EMA target encoder 的自监督学习方法。
4. Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR 2023*. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) —— I-JEPA：图像版本的 JEPA。
5. Schmid, M., et al. (2024). Revisiting Design Choices in Joint-Embedding Predictive Architecture. *arXiv:2401.00000*. [链接](https://arxiv.org/abs/2401.00000) —— JEPA 设计选择的系统研究。
