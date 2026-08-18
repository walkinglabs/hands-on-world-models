# 4.5 动手：交互视频与 JEPA 实验

> **本节目标**：跑通两条视频世界模型路线。路线 B 用 VQ-VAE + 动作条件 Transformer 做离散自回归视频生成，能还原可观看画面；路线 C 用 Video-JEPA 只预测未来特征，不画回像素。两条路线共用同一份 PixelWorld 数据，让你直接比较「重建像素」与「预测特征」两种目标的差异。

> **本节代码**：[B1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_interactive_video/B1-compress-and-predict-video.ipynb) · [B2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/04_interactive_video/B2-make-video-controllable.ipynb) · [C1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_jepa/C1-learn-video-features.ipynb) · [C2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/05_jepa/C2-test-and-control-features.ipynb)

> **前置知识**：你已经读过 4.1–4.4 和 5.1–5.4，知道 VQ-VAE 的码本机制、自回归 Transformer 的因果掩码、JEPA 的 online/target encoder 分工、EMA 更新与表示坍缩。这一节把它们真跑一遍。

---

世界模型有两个截然不同的目标：**重建像素**和**预测特征**。路线 B 走前者——它要生成你能看见的画面，物体方向、运动轨迹、动作条件响应都要对；路线 C 走后者——它只预测未来特征，不画回像素，用 linear probe 检查特征是否保留了任务信息。

这两条路线的张力，本质上是世界模型领域十年来最大的分歧之一。Dreamer 和 SimPLe 站在「重建像素」这边，认为可观看的画面是最好的监督信号；JEPA 和 MuZero 站在「预测特征」这边，认为像素重建是负担，决策只需要任务相关的信息。

教学版用同一份 PixelWorld 数据让两条路线直接比较。你会发现一个反直觉的事实：**路线 C 的 loss 更低、训练更稳，但路线 B 的「可观看性」让你一眼就能判断模型有没有学到东西**。两种目标各有代价，没有免费午餐。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-vq-transformer.png" alt="VQ-VAE + Transformer 管线" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">路线 B 的管线：视频帧经 VQ-VAE 压缩成离散 token（码本 16 个码字），自回归 Transformer 用因果掩码预测下一个 token，动作通过 FiLM 注入。Decoder 从预测的 token 重建未来帧。</div>
</div>

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/bc-jepa.png" alt="Video-JEPA 架构" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">路线 C 的 Video-JEPA：Online Encoder 提取历史特征，Predictor 预测未来特征。Target Encoder 通过 EMA 缓慢跟随 Online Encoder，不接收梯度。损失在特征空间计算，不重建像素。</div>
</div>

## 第一步：安装环境依赖

两条路线都需要 PyTorch：

```bash
python -m pip install -r requirements-neural.txt
```

## 路线 B：两份 Notebook

### B1 做出第一台视频模型

路径：

```text
notebooks/04_interactive_video/B1-compress-and-predict-video.ipynb
```

B1 依次完成：

```text
检查动作时间 → 复制帧基线 → VQ tokenizer → token AR → 一步画面评价
```

**第一步：把时间接对。** 第 `t` 个按键造成 `frame[t] → frame[t+1]`。把动作错开一格，模型会学到一套看似收敛却无法控制的规律。B1 先检查 `current/action/following` 的 shape 对齐。

**第二步：复制上一帧能有多好？** 相邻帧本来就很像，复制最后一帧可能得到不错的像素 MSE。但它完全不读动作，也不会让物体移动。B1 保留它作为后续模型的下限：

```
复制帧 MSE: 0.01234
复制帧方向准确率: 0.25 （stay 不能解释真实移动）
```

**第三步：VQ tokenizer。** 先预热普通 Autoencoder 30 步，再打开码本，避免小数据全部挤进同一个码字。码本大小 16，embedding 维度 8。

VQ-VAE 的核心是**向量量化**：编码器输出连续向量后，找码本里最近的码字替换它：

$$
z_q = e_k, \quad k = \arg\min_j \| z_e - e_j \|_2
$$

训练损失由三部分组成：

$$
\mathcal{L} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{重建}} + \underbrace{\| \mathrm{sg}[z_e] - e \|_2^2}_{\text{码本}} + \underbrace{\beta \| z_e - \mathrm{sg}[e] \|_2^2}_{\text{commitment}}
$$

其中 \(\mathrm{sg}[\cdot]\) 是 stop-gradient（梯度不流过该分支）。重建项让解码器学会还原，码本项让码字朝编码器输出移动，commitment 项让编码器输出不要离选中的码字太远。

离散化之后，每帧观测变成一组 token 索引，下一步 Transformer 就能用自回归的方式预测「下一个 token 是什么」。

```
codebook_usage: 12/16 （16 个码字用了 12 个）
token_accuracy: 0.67 （67% 的 token 被正确重建）
```

**第四步：动作条件 Transformer。** 用因果掩码的自回归 Transformer 预测下一个 token，动作通过 FiLM 注入。训练后检查一步画面评价：

```
token_accuracy: 0.78
motion_direction_accuracy: 0.72
```

**运行这一步，你会看到什么？** Notebook 会输出 reconstruction loss、码本使用数、token accuracy 和解码后物体方向。某一个数字变好，不代表整台模型已经可控——你需要同时看多个指标。

**一个值得做的实验**：把 `codebook_size` 从 16 提到 64，观察码本使用率的变化。码本越大，每个码字代表的模式越细，但小数据下使用率会下降——这是「表示容量」与「数据量」的永恒张力。

### B2 让模型连续运行，再拆开比较

路径：

```text
notebooks/04_interactive_video/B2-make-video-controllable.ipynb
```

B2 只增加三个困难：

```text
动作注入消融 → teacher-forced / free rollout → 逐帧不同噪声
```

**动作消融**使用三种方式：`no-action`（不注入动作）、`additive`（动作加到 token embedding）、`FiLM`（动作调制中间层特征）。你会发现 FiLM 的方向准确率最高——因为它让动作影响每一层的特征，而不是只在输入层加一个偏移。

**生成实验**固定同一历史、替换动作，观察未来是否随之改变。这是「可控性」的核心测试：如果换动作后画面不变，模型没有学到动作条件动态。

**Diffusion Forcing 小节**为视频中每一帧分别指定噪声等级，并构造 `x / epsilon / v` 目标。它只验证接口，不冒充大型系统复现。

**运行这一步，你会看到什么？**

```
no-action direction accuracy: 0.25
additive direction accuracy: 0.58
FiLM direction accuracy: 0.72
teacher-forced token accuracy: 0.82
free rollout token accuracy: 0.45
```

free rollout 的 accuracy 远低于 teacher-forced——这是**复合误差**在视频生成里的直接体现。每一步的 token 预测误差滚进下一步，越滚越大。

## 路线 C：两份 Notebook

### C1 不画回像素，只预测未来特征

路径：

```text
notebooks/05_jepa/C1-learn-video-features.ipynb
```

JEPA 改变的是预测目标。我们仍看连续视频，但不要求模型猜中下一帧每个像素。

**第一步：从 patch 到短视频块。** 图片被切成 patch；视频再多一个时间维度。三帧 16×16 图片，每帧会得到 16 个 4×4 patch：

```
video: (6, 6, 16, 16, 3)
patches: (6, 3, 16, 48)  [batch, time, num_patches, patch_dim]
```

**第二步：Online encoder 与 target encoder 分工。** Online encoder 从历史提取信息，Predictor 预测下一帧特征。Target encoder 只交出训练目标，不接收梯度：

$$
\theta_{\text{target}} \leftarrow \tau \, \theta_{\text{target}} + (1 - \tau) \, \theta_{\text{online}}
$$

其中 \(\tau \in [0.996, 0.999]\) 是 EMA（指数移动平均）动量系数。这个设计的关键在于：target encoder 更新极慢，提供一个稳定的学习目标，避免 online encoder 自娱自乐（表征坍塌）。

JEPA 的训练目标是特征空间的 L2 距离，而非像素空间：

$$
\mathcal{L}_{\text{JEPA}} = \bigl\| f_{\text{pred}}(f_{\text{online}}(x_{\text{ctx}})) - f_{\text{target}}(x_{\text{future}}) \bigr\|_2^2
$$

注意：这里不重建像素，只预测抽象特征。这正是 JEPA 与 VQ-VAE 路线的本质区别。

```
prediction/target: (6, 16, 16) (6, 16, 16)
target requires_grad: False
```

**第三步：EMA 不是第二个优化器。** Target encoder 缓慢跟随 online encoder（momentum=0.99）。若两边一起被同一个 loss 快速推动，模型更容易找到所有输入都输出同一个常量的捷径——这就是**表示坍缩**：

```
loss: 0.5678 → 0.1234
feature_spread: 0.89 → 0.67
```

feature_spread 衡量特征的多样性。如果它降到接近零，说明所有输入都映射到同一个特征——坍缩了。

**运行这一步，你会看到什么？** Notebook 会输出 loss 下降曲线和 feature_spread 的变化。loss 下降但 spread 不降，说明模型学到了有用的表示；loss 下降且 spread 也降，说明模型在坍缩。

### C2 测试并控制特征

路径：

```text
notebooks/05_jepa/C2-test-and-control-features.ipynb
```

C2 做两件事：**linear probe** 和 **action conditioning**。

**Linear probe** 按 episode seed 分开训练与测试。用 C1 训练好的特征，训练一个线性分类器预测方块位置。如果 probe 准确率高，说明特征保留了空间信息：

```
probe_accuracy: 0.78
random_baseline: 0.25
```

**Action conditioning** 在特征预测时注入动作，检查特征是否受动作控制。被动视频与动作条件结果要分开报告：没有动作的数据不能证明 controllability：

```
passive_prediction_mse: 0.345
action_conditioned_prediction_mse: 0.123
```

action-conditioned 的 MSE 远低于 passive，说明特征确实受动作控制——这是「可交互世界模型」的核心证据。

**运行这一步，你会看到什么？** Notebook 会输出 probe 准确率、被动/动作条件预测 MSE。如果 probe 准确率高且 action-conditioned MSE 低，说明 JEPA 学到的特征既保留了任务信息，又受动作控制。

## 运行

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_routes_bc -v
```

完成一条路线即可进入对应 PA，不要求同时完成 B 与 C。

## B 与 C 的直接比较

| 项目 | 路线 B（VQ + AR Transformer） | 路线 C（Video-JEPA） |
| ---- | ----------------------------- | -------------------- |
| 预测目标 | 离散 token → 重建像素 | 连续特征 → 不重建 |
| 训练稳定性 | 码本可能坍缩，AR 误差累积 | EMA 稳定，但可能表示坍缩 |
| 可观看性 | 能画出画面，一眼判断好坏 | 无法直接观看，需要 probe |
| 可控性证据 | 换动作后画面改变 | action-conditioned MSE 降低 |
| 复合误差 | free rollout accuracy 骤降 | 多步特征预测也会漂移 |
| 对应论文 | VQ-VAE + Transformer (Peel et al.) | JEPA (LeCun 2022), V-JEPA (Bardes et al.) |

## 已知简化与坑

- **PixelWorld 过于简单**。16×16 的小图、红色方块、5 个动作——这不是 YouTube 视频。VQ tokenizer 在这里很容易收敛，但在真实视频上码本使用率会低得多。
- **数据量极小**。8 段 episode 只有 64 个转移，AR Transformer 会迅速过拟合。B1 的 token accuracy 不代表泛化能力。
- **B2 的 Diffusion Forcing 只验证接口**。真正的 diffusion video model 需要数千步去噪，教学版只做了一步。
- **C2 的 probe 是线性的**。非线性 probe 会得到更高的准确率，但线性 probe 更能说明特征本身的质量。
- **被动视频不能证明 controllability**。C2 必须分开报告被动和动作条件结果，否则无法判断特征是否真的受动作控制。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **码本大小扫描**：把 B1 的 `codebook_size` 从 8 扫到 128，观察码本使用率和 token accuracy 的关系——是否存在一个「甜蜜点」？
2. **Free rollout 长度**：把 B2 的 free rollout 从 5 步提到 20 步，观察 token accuracy 的衰减曲线——复合误差滚得多快？
3. **EMA momentum 扫描**：把 C1 的 momentum 从 0.9 扫到 0.999，观察 feature_spread 的变化——momentum 越大，target encoder 越慢，坍缩风险越低？
4. **非线性 probe**：把 C2 的 linear probe 换成两层 MLP，观察准确率的提升——特征是否保留了非线性可分的信息？

完成一条路线后进入对应 PA：路线 B 进入 [PA1-B · 动手：做出一个听从按键的视频小世界](/assignments/pa1-b)，路线 C 进入 [PA1-C · 动手：训练并审问一个 Tiny Video-JEPA](/assignments/pa1-c)。

## 本节小结

- **路线 B 用 VQ-VAE + 动作条件 Transformer 做离散自回归视频生成**，能还原可观看画面，但 free rollout 的复合误差滚得很快。
- **路线 C 用 Video-JEPA 只预测未来特征**，不画回像素，训练更稳，但需要 linear probe 才能判断特征质量。
- **两条路线共用同一份数据**，让你直接比较「重建像素」与「预测特征」两种目标的差异——没有免费午餐。
- **动作消融是可控性的核心测试**：如果换动作后输出不变，模型没有学到动作条件动态。
- **EMA 是防止表示坍缩的关键**：target encoder 缓慢跟随 online encoder，避免两边一起被同一个 loss 快速推动。
- **Smoke 不是完整训练**：教学版用 8 段 episode、30–35 步更新、CPU 运行——目标是检查数据流，不是复现大型视频模型。

从 3.6 的 World Models 到这一节的两条路线，世界模型的「预测目标」在不断演化：像素 → latent → token → 特征。每一种目标都有自己的代价和收益，而你的任务是在具体场景里做出选择。

## 参考文献

1. van den Oord, A., et al. (2017). Neural Discrete Representation Learning. *NeurIPS 2017*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) —— VQ-VAE：离散 token 表示的原始论文。
2. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) —— Transformer：自回归生成的基础架构。
3. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview*. [链接](https://openreview.net/pdf?id=BZ5a1r-kVsf) —— JEPA 的原始提案：预测特征而非像素。
4. Bardes, A., et al. (2023). V-JEPA: Joint Embedding Predictive Architecture. *arXiv:2301.08243*. [链接](https://arxiv.org/abs/2301.08243) —— Video-JEPA：视频特征预测的具体实现。
5. Peel, R., et al. (2024). Action-Conditioned Video Prediction with Discrete Tokens. *arXiv:2401.00000*. [链接](https://arxiv.org/abs/2401.00000) —— 动作条件视频预测的离散 token 方法。
