# 第 1 章　预备知识

网格世界只有几个整数。现实中的一次观察可能包含图片、相机、语言和机器人状态，一段经历还要包含动作、奖励与时间顺序。

这一章建立五条路线共用的语言。每个组件只讲到足以判断它的输入、输出和用途；完整实现留到真正使用它的路线文章。正文与公式交错出现：每个关键概念都配上一段数学表达，边讲边列。

## 本章文章

1. [1.1 张量、时间与轨迹](./01-01-tensors-and-trajectories.md)：读懂 $[B,T,C,H,W]$，把动作放在正确的两帧之间。
2. [1.2 图像编码器：CNN 与 ViT](./01-02-cnn-and-vit.md)：比较局部卷积核与 patch token。
3. [1.3 记忆与动态：初见 RNN、Transformer 与 RSSM](./01-03-memory-and-dynamics.md)：从速度线索走到随机隐状态。
4. [1.4 压缩与生成：初见 VAE、VQ-VAE 与扩散](./01-04-compression-and-generation.md)：比较连续 latent、离散 token 和多种未来。
5. [1.5 空间表示：初见 BEV 与占用网格](./01-05-space-representations.md)：认识内参与外参、点云、俯视图和空间占用。
6. [1.6 决策接口：初见价值、策略与规划器](./01-06-value-policy-planner.md)：说明预测怎样被用于选择动作。
7. [1.7 经验回放与第一台模型（预览）](./01-07-data-and-first-model.md)：先看到整条管线，每一步的展开留给第 2 章。
8. [1.8 动手：基础实验](./01-08-basic-experiments.md)：接起看见、记忆、压缩、空间与规划。

## 学完以后怎样选路

先写下模型最需要交出的结果：latent、画面、feature、机器人动作，还是三维占用。第 2–6 章分别围绕这五种结果展开，彼此不是先修关系。

## 参考资料

### 实践博客（5 篇）

1. [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) —— 图解 Transformer 的经典博客，配 1.2 的 ViT 与 1.3 的注意力记忆。
2. [Understanding LSTM Networks (Chris Olah, 2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) —— RNN 与 LSTM 的直觉讲解，配 1.3 的记忆组件。
3. [The Unreasonable Effectiveness of Recurrent Neural Networks (Karpathy, 2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) —— 用大量可视化展示序列模型怎样“记住”过去，配 1.3。
4. [What are Diffusion Models? (Lilian Weng, 2021)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) —— 扩散模型从公式到采样的系统梳理，配 1.4 与第 4 章。
5. [Transformers for Image Recognition at Scale (Google AI, 2020)](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) —— ViT 官方解读博客，讲清 patch 化与规模效应，配 1.2。

### 原始论文（5 篇）

1. [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) —— Transformer 原始论文，1.2、1.3 注意力的来源。
2. [An Image is Worth 16x16 Words: ViT (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929) —— Vision Transformer 原始论文，patch token 的来源。
3. [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735) —— LSTM 原始论文，记忆组件的奠基工作。
4. [Neural Discrete Representation Learning: VQ-VAE (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937) —— 离散 token 表示的奠基论文，第 4 章视频分词器的直接前身。
5. [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) —— DDPM 原始论文，1.4 与第 4 章扩散路线的数学基础。
