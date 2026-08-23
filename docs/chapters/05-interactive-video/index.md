# 第 5 章　交互式视频

普通视频模型只要生成一段像真的视频。视频世界模型还要回答一个更难的问题：从同一段历史出发，按下不同的键，接下来的画面会不会发生相应变化？

本章不把 VQ-VAE、Transformer 和 Diffusion 当作三套互不相干的方法。我们先决定视频应当怎样表示，再决定怎样生成，最后检查它能否连续运行并帮助任务。

```text
连续经历 → 选择视频表示 → 学习带动作的变化 → 连续生成 → 检查控制、记忆与速度
```

## 本章文章

1. [5.1 从视频生成到视频世界模型](./01-video-data.md)：先分清“像视频”和“能模拟动作后果”，再把数据时间接对。
2. [5.2 先决定预测什么](./02-vq-tokenizer.md)：比较像素、连续 latent、离散 token 与语义特征。
3. [5.3 AR、Diffusion 与 Diffusion Forcing](./03-action-transformer.md)：看懂三种生成方式解决了什么，又留下什么问题。
4. [5.4 动作、记忆、长时生成与评价](./04-diffusion-and-evaluation.md)：把模型放进自由 rollout，检查它是否真的可用。
5. [5.5 动手：交互视频实验](./05-interactive-video.md)：VQ-VAE 加动作条件 Transformer。

B1 做出第一台离散视频模型；B2 用相同数据比较动作注入、自由 rollout 和逐帧不同噪声。PA1-B 再把这些检查接成一次完整实验。全章只使用两份 Notebook。

## 学完以后

你应当能拿到一种新方法，先问清四件事：它在哪里表示视频；每次预测什么；动作和历史怎样进入模型；它用什么证据证明长时间生成仍然受控。能回答这四个问题，IRIS、DIAMOND、GameNGen、Oasis、Genie 和后续方法就不再只是一串名字。

## 参考资料

### 实践博客（5 篇）

1. [Genie: Generative Interactive Environments (Google DeepMind, 2024)](https://deepmind.google/blog/genie-generative-interactive-environments/) —— Genie 官方博客：从无动作标注视频学潜在动作，是“动作从哪来”的直观讲解。
2. [Genie 3: A new frontier for world models (Google DeepMind, 2025)](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) —— 实时交互世界模型的最新官方博客，展示长时间一致性现状。
3. [Introducing GAIA-1 (Wayve, 2023)](https://wayve.ai/thinking/introducing-gaia1/) —— Wayve 官方博客，讲清视频 token + 自回归 Transformer 的驾驶世界模型配方。
4. [DIAMOND 项目主页 (Alonso et al.)](https://eloialonso.github.io/diamond/) —— 扩散世界模型的可玩 demo、代码与 Atari 结果，配 5.4。
5. [Genie 2: A large-scale foundation world model (Google DeepMind, 2024)](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) —— 单图生成可玩 3D 世界的官方博客，展示工程取舍。

### 原始论文（5 篇）

1. [Genie: Generative Interactive Environments (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) —— 从无动作标注视频学出潜在动作模型（LAM）的原始论文。
2. [GAIA-1: A Generative World Model for Autonomous Driving (Hu et al., 2023)](https://arxiv.org/abs/2309.17080) —— 视频 token + 自回归 Transformer 的工业级实现，配 5.3 的动作条件生成。
3. [DIAMOND: Diffusion for World Modeling (Alonso et al., 2024)](https://arxiv.org/abs/2405.12399) —— 用扩散模型当世界模型并直接在其中训练策略，配 5.4。
4. [Transformers are Sample-Efficient World Models: IRIS (Micheli et al., 2023)](https://arxiv.org/abs/2209.00588) —— 离散 token + Transformer 在 Atari 100k 上的样本效率证据，与本章 B 路线结构最接近。
5. [Diffusion Models Are Real-Time Game Engines: GameNGen (Valevski et al., 2024)](https://arxiv.org/abs/2408.14837) —— 用扩散模型实时模拟 DOOM，展示噪声增强怎样抑制自回归漂移。
