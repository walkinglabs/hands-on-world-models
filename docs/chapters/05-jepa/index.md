# 第 5 章　JEPA：特征空间预测

一片树叶下一秒向左抖还是向右抖，通常不影响机器人绕开桌子。JEPA 因此不要求还原所有像素，而是在特征空间里预测较稳定的未来。

本路线把预测目标从像素换成特征，用 mask、EMA 与 stop-gradient 避免表示坍缩，再在视频上引入动作，让特征未来听从控制信号。

## 本章文章

1. [4.1 预测特征而非像素](./05-01-feature-prediction.md)
2. [4.2 掩码、EMA 与表示坍缩](./05-02-mask-ema-collapse.md)
3. [4.3 视频 JEPA](./05-03-video-jepa.md)
4. [4.4 动作条件 JEPA（Action-JEPA）](./05-04-action-jepa.md)

## 本章动手 notebook

- [C1 动手：视频特征预测（JEPA）](/labs/route-bc)
- [C2 动手：动作条件特征预测（Action-JEPA）](/labs/route-bc)

被动视频可以检查表示质量，不能单独证明模型理解控制。只有加入时间对齐的动作以后，才检查反事实与规划。

## 参考资料

### 实践博客（5 篇）

1. [The first AI model based on Yann LeCun's vision for more human-like AI (Meta AI)](https://ai.meta.com/blog/yann-lecun-advances-in-ai-research/) —— Meta 官方博客，讲清 I-JEPA 为什么不做像素重建、以及它与 LeCun 蓝图的关系。
2. [Meta 官方博客：V-JEPA 2 world model (Meta AI, 2025)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) —— V-JEPA 2 与动作条件版 V-JEPA 2-AC 的官方发布页，配 5.4。
3. [What Is JEPA? (Turing Post)](https://www.turingpost.com/p/jepa) —— 第三方综述博客，把 JEPA 家族与生成式路线的争论梳理得很清楚。
4. [Self-Supervised Representation Learning (Lilian Weng, 2019)](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) —— 自监督表示学习的谱系梳理，帮 JEPA 找到它在其中的位置。
5. [V-JEPA 2 论文页 (Meta AI)](https://ai.meta.com/research/publications/v-jepa-2/) —— 论文官方页面，附 PDF 与代码入口，便于对照 5.3、5.4 查证细节。

### 原始论文（5 篇）

1. [A Path Towards Autonomous Machine Intelligence (LeCun, 2022)](https://openreview.net/forum?id=BZ5a1r-kVsf) —— JEPA 与世界模型架构蓝图的立场论文，本章的理论源头。
2. [Self-Supervised Learning From Images with a JEPA: I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243) —— JEPA 在图像上的首次落地，掩码块预测抽象表示的原始论文。
3. [Revisiting Feature Prediction for Learning Visual Representations from Video: V-JEPA (Bardes et al., 2024)](https://arxiv.org/abs/2404.08471) —— 视频版 JEPA，时空掩码与 EMA 目标编码的具体配方。
4. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (Bardes et al., 2025)](https://arxiv.org/abs/2506.09985) —— 动作条件版 V-JEPA 2-AC 用 62 小时机器人数据实现零样本规划，配 5.4。
5. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning (Bardes et al., 2022)](https://arxiv.org/abs/2105.04906) —— 防坍塌正则化的来源，是理解 5.2 坍塌问题的关键拼图。
