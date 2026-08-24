# 6.1　特征预测

> **第 6 章 · 联合嵌入预测架构（JEPA）**
>
> 一片树叶下一秒向左抖还是向右抖，通常不影响机器人绕开桌子。JEPA 因此不要求还原所有像素，而是在特征空间里预测较稳定的未来。
>
> 本路线把预测目标从像素换成特征，用 mask、EMA 与 stop-gradient 避免表示坍缩，再在视频上引入动作，让特征未来听从控制信号。
>
> 被动视频可以检查表示质量，不能单独证明模型理解控制。只有加入时间对齐的动作以后，才检查反事实与规划。
>
> 动手实验：[6.5 动手：视频 JEPA 的从零实现](/chapters/06-jepa/05-jepa)（两份 Notebook 分别做视频特征预测与动作条件特征预测）。

先看一个具体场景。PixelWorld 的每一帧里，背景在不停换色、树叶在抖、水面在反光，但方块的位置和运动方向才是机器人绕桌要用的信息。

如果我们逼模型逐像素预测下一帧，它得花大量算力去猜那些既不确定、又跟任务无关的细节。所谓世界模型，并不需要把画面画出来，它需要的是判断"方块下一刻会到哪儿"。

## 像素预测为什么吃力

最朴素的像素世界模型，目标是让预测图尽量贴近真实图。用均方误差（MSE）来写，一张高 $H$、宽 $W$、$C$ 通道的图，每个像素位置都要算误差：

$$
\mathcal{L}_{\text{pixel}}
= \frac{1}{N}\sum_{i=1}^{N}\big\|\hat{x}_i - x_i\big\|_2^{2},
\qquad
\hat{x}_i = D(s_i).
$$

这里 $\hat{x}_i$ 是 Decoder 从状态 $s_i$ 还原出的整帧像素，$x_i$ 是真实帧，$N$ 是样本数。背景一个像素的随机抖动，和方块中心一个像素的偏差，在 $\mathcal{L}_{\text{pixel}}$ 里权重相同。

于是模型既要解释物体的因果，又要解释背景的噪声。后者很难、又未必有用。

## JEPA 的回答：换一个目标

联合嵌入预测架构（Joint-Embedding Predictive Architecture，JEPA）把目标从"像素"换成"特征"。它不要求画回完整画面，只要求预测另一条网络给出的表示。

设 Context Encoder 是 $f_\theta$，它读可见上下文 $c$；Predictor 是 $g_\phi$，它输出对目标的猜测。目标特征由 Target Encoder $f_{\bar\theta}$ 给出。损失变成：

$$
\mathcal{L}_{\text{JEPA}}
= \frac{1}{N}\sum_{i=1}^{N}\big\|g_\phi\big(f_\theta(c_i)\big) - f_{\bar\theta}(y_i)\big\|_2^{2}.
$$

左右两边都是特征向量，没有任何像素出现。$y_i$ 是要预测的目标片段（被遮区域或未来帧），$c_i$ 是模型能看到的上下文。

对比一下两者在结构上的差别：

```text
像素路线   observation → state → Decoder → 像素  ←→  真实像素
JEPA 路线  context     → 编码  → Predictor → 特征 ←→  目标编码
```

## 为什么目标特征不能人写

如果数据自带物体的位置、速度、类别，我们当然可以直接监督这些量。问题是互联网上的视频几乎没有这种标签，而真实任务需要什么信息，也很难事先列全。

JEPA 让 $f_{\bar\theta}$ 自己从数据里产生训练目标，再通过 mask 的形状、架构和下游任务来间接控制表示保留什么。换句话说，"该学什么"是模型从数据里找出来的，不是我们一行行列出来的。

## 一个最小对照

回到 PixelWorld：背景每帧随机变色，方块的运动规律不变。像素预测会把背景的随机变色也算进误差，损失忽高忽低。而一个学得合适的 JEPA，因为只比较特征、不重建背景，应能稳定地把方块位置和速度读出来。

这正是 6.5 第一份 Notebook「学出视频特征」要验证的事情：训练一个 Tiny Video-JEPA，看它的特征 spread 是否保持非零、feature loss 是否稳定下降。

## 付出的代价

没有 Decoder，特征不能直接画回完整未来，我们也不能只凭 feature loss 断言模型保留了有用信息。$\mathcal{L}_{\text{JEPA}}$ 低，可能只是模型找到了一个容易匹配的常数。

所以评价要补充：linear probe（线性探针）、检索、动作敏感性、甚至小规模规划任务。所谓"有用"，必须由使用者和具体任务给出证据，而不是单看 loss 曲线。

## 小结

- JEPA 预测 Target Encoder 给出的特征 $f_{\bar\theta}(y)$，而不是重建像素。
- 特征目标让模型不必为不稳定的视觉细节买单。
- feature loss 低不等于表示有用，仍要靠下游 probe 与控制任务来检验。

[下一篇 → 6.2 掩码、EMA 与表示坍缩](./02-mask-ema-collapse.md) · [动手：视频特征预测](/chapters/06-jepa/05-jepa)

---

## 参考资料

### 实践博客

1. [The first AI model based on Yann LeCun's vision for more human-like AI (Meta AI)](https://ai.meta.com/blog/yann-lecun-advances-in-ai-research/) —— Meta 官方博客，讲清 I-JEPA 为什么不做像素重建、以及它与 LeCun 蓝图的关系。
2. [Meta 官方博客：V-JEPA 2 world model (Meta AI, 2025)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) —— V-JEPA 2 与动作条件版 V-JEPA 2-AC 的官方发布页，配 6.4。
3. [What Is JEPA? (Turing Post)](https://www.turingpost.com/p/jepa) —— 第三方综述博客，把 JEPA 家族与生成式路线的争论梳理得很清楚。
4. [Self-Supervised Representation Learning (Lilian Weng, 2019)](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) —— 自监督表示学习的谱系梳理，帮 JEPA 找到它在其中的位置。
5. [V-JEPA 2 论文页 (Meta AI)](https://ai.meta.com/research/publications/v-jepa-2/) —— 论文官方页面，附 PDF 与代码入口，便于对照 6.3、6.4 查证细节。

### 经典文献

1. [A Path Towards Autonomous Machine Intelligence (LeCun, 2022)](https://openreview.net/forum?id=BZ5a1r-kVsf) —— JEPA 与世界模型架构蓝图的立场论文，本章的理论源头。
2. [Self-Supervised Learning From Images with a JEPA: I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243) —— JEPA 在图像上的首次落地，掩码块预测抽象表示的原始论文。
3. [Revisiting Feature Prediction for Learning Visual Representations from Video: V-JEPA (Bardes et al., 2024)](https://arxiv.org/abs/2404.08471) —— 视频版 JEPA，时空掩码与 EMA 目标编码的具体配方。
4. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (Bardes et al., 2025)](https://arxiv.org/abs/2506.09985) —— 动作条件版 V-JEPA 2-AC 用 62 小时机器人数据实现零样本规划，配 6.4。
5. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning (Bardes et al., 2022)](https://arxiv.org/abs/2105.04906) —— 防坍塌正则化的来源，是理解 6.2 坍塌问题的关键拼图。
