# 5.1　视频世界模型

> **第 5 章 · 交互式视频**
>
> 普通视频模型只要生成一段像真的视频。视频世界模型还要回答一个更难的问题：从同一段历史出发，按下不同的键，接下来的画面会不会发生相应变化？
>
> 本章不把 VQ-VAE、Transformer 和 Diffusion 当作三套互不相干的方法。我们先决定视频应当怎样表示，再决定怎样生成，最后检查它能否连续运行并帮助任务。
>
> ```text
> 连续经历 → 选择视频表示 → 学习带动作的变化 → 连续生成 → 检查控制、记忆与速度
> ```
>
> 动手实验：[5.6 动手：动作条件视频模型的从零实现](/chapters/05-interactive-video/06-interactive-video)（两份 Notebook 涵盖离散 Token + 动作自回归与扩散去噪比较）。

先看两项任务。

第一项任务给模型一句话：“一辆汽车驶过弯道。”模型生成一段合理视频即可。汽车向左还是向右、速度是多少，只要整体看起来自然，都可能算对。

第二项任务先给出同一段驾驶历史，再分别输入“左转”和“右转”。这时模型不能自由发挥：方向盘不同，车辆位置也必须跟着改变。这才是本章要做的事情。

```text
视频生成：条件 → 一段看起来合理的视频
视频世界模型：历史观察 + 动作 → 这个动作可能造成的未来视频
```

因此，清晰度只是起点。视频世界模型还需要三种能力：画面听从动作，过去发生的事不会很快被忘掉，生成结果可以供人交互或供规划器使用。

## 先把时间接对

一段游戏记录按下面顺序保存：

```text
frame_0, action_0, frame_1, action_1, frame_2
```

`action_0` 是从 `frame_0` 走到 `frame_1` 之间执行的动作。若把它配到后一段变化，模型仍可能得到不高的 loss，因为相邻画面本来就很像；真正换动作时，它才会暴露出不受控制。

一条训练片段至少包含：

```text
frames:  [T+1,C,H,W]
actions: [T,A]
done:    [T]
```

还应记录帧率、控制频率、动作重复次数、传感器与执行器延迟、episode 和 seed。若一个按键会保持四帧，这四帧也属于模型需要学习的因果关系。

## 三层数据，各回答一个问题

| 数据                     | 本章用来回答什么                     | 当前状态                            |
| ------------------------ | ------------------------------------ | ----------------------------------- |
| PixelWorld               | 时间是否对齐，换动作后物体是否换方向 | 项目内可按 seed 生成，必做          |
| DINO-WM PushT 等动作视频 | 连续动作、遮挡和接触下能否预测       | 外部进阶数据，loader 尚未随课程发布 |
| CarRacing 小数据         | 纹理、相机运动和连续驾驶下能否迁移   | 外部选做，collector 尚未发布        |

无动作互联网视频也有用，它能教模型外观和常见运动；但它不能单独证明哪个按键造成了哪个结果。若从无动作视频中发现 latent action，还需要把这个隐含控制量与真实控制接口对齐。

数据要按完整 episode、场景或环境 seed 切分，不能把相邻帧随机分到训练集和测试集。否则测试集可能只是训练片段的下一秒。

## 第一个基线不是神经网络

在训练模型前，先实现两个基线：

1. 复制最后一帧。短时间内像素误差可能很好，但物体不听动作；
2. 不输入动作的下一帧模型。它能利用惯性，却无法通过反事实测试。

固定同一段历史和随机源，分别输入 `left`、`right` 和 `noop`。只有预测位置和方向随动作改变，才能继续讨论“可交互”。

## 小结

- [ ] 视频世界模型不只生成合理画面，还要模拟动作造成的变化。
- [ ] `action_t` 必须与 `frame_t → frame_{t+1}` 对齐。
- [ ] 项目先用可生成小数据验证因果接口，再把同一检查迁移到真实小数据。
- [ ] 复制上一帧、去掉动作和错位动作都是必须保留的基线。

---

## 参考资料

### 实践博客

1. [Genie: Generative Interactive Environments (Google DeepMind, 2024)](https://deepmind.google/blog/genie-generative-interactive-environments/) —— Genie 官方博客：从无动作标注视频学潜在动作，是“动作从哪来”的直观讲解。
2. [Genie 3: A new frontier for world models (Google DeepMind, 2025)](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) —— 实时交互世界模型的最新官方博客，展示长时间一致性现状。
3. [Introducing GAIA-1 (Wayve, 2023)](https://wayve.ai/thinking/introducing-gaia1/) —— Wayve 官方博客，讲清视频 token + 自回归 Transformer 的驾驶世界模型配方。
4. [DIAMOND 项目主页 (Alonso et al.)](https://eloialonso.github.io/diamond/) —— 扩散世界模型的可玩 demo、代码与 Atari 结果，配 5.4。
5. [Genie 2: A large-scale foundation world model (Google DeepMind, 2024)](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) —— 单图生成可玩 3D 世界的官方博客，展示工程取舍。

### 经典文献

1. [Genie: Generative Interactive Environments (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) —— 从无动作标注视频学出潜在动作模型（LAM）的原始论文。
2. [GAIA-1: A Generative World Model for Autonomous Driving (Hu et al., 2023)](https://arxiv.org/abs/2309.17080) —— 视频 token + 自回归 Transformer 的工业级实现，配 5.3 的动作条件生成。
3. [DIAMOND: Diffusion for World Modeling (Alonso et al., 2024)](https://arxiv.org/abs/2405.12399) —— 用扩散模型当世界模型并直接在其中训练策略，配 5.4。
4. [Transformers are Sample-Efficient World Models: IRIS (Micheli et al., 2023)](https://arxiv.org/abs/2209.00588) —— 离散 token + Transformer 在 Atari 100k 上的样本效率证据，与本章 B 路线结构最接近。
5. [Diffusion Models Are Real-Time Game Engines: GameNGen (Valevski et al., 2024)](https://arxiv.org/abs/2408.14837) —— 用扩散模型实时模拟 DOOM，展示噪声增强怎样抑制自回归漂移。
