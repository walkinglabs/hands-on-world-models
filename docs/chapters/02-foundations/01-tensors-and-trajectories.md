# 2.1　张量、时间与轨迹

> **第 2 章 · 预备知识**
>
> 网格世界只有几个整数。现实中的一次观察可能包含图片、相机、语言和机器人状态，一段经历还要包含动作、奖励与时间顺序。
>
> 这一章建立后续五条世界模型路线共用的基础语言。每个组件只讲到足以判断它的输入、输出和用途；完整实现留到真正使用它的路线章节。
>
> 核心实验在 [2.8 动手：组件接口的简洁实现](/chapters/02-foundations/08-basic-experiments) 中串联看见、视觉、记忆、压缩、空间与规划。学完本章后，先明确模型最需要输出的形式（潜在向量、连续画面、特征表示、机器人动作或三维占用），再从第 4–8 章选择对应的设计路线——各路线彼此独立。

网格世界只有几个整数就能讲清楚状态。真实的一次观察却可能是一张彩色图片、一段视频，再加上机器人关节角度。要把这些数据交给模型，我们需要一种统一的装法。

所谓**张量**（tensor），可以先理解成一个装有许多数字的多维盒子。盒子的每一维有固定的长度，这些长度排在一起就构成 shape。

## 从一张图片到一段视频

一张 $16\times16$ 的彩色图片可以写成 shape $[16,16,3]$。三个数字分别表示高、宽和颜色通道，数字总数是 $16\times16\times3=768$。

十六段这样的视频，每段各有 20 帧，合并成一个批量一起计算，shape 就变成：

$$
[B,T,H,W,C]=[16,20,16,16,3].
$$

这里 $B$ 表示一次并排处理多少段样本（batch），$T$ 表示时间步数。把单张图片推广到视频，本质只是在盒子前面再多加两根轴。

## 维度名称比大小更重要

光看 $[32,10,5]$ 这串数字，我们无法判断它到底是什么：可能是 32 段轨迹、每段 10 步、每步 5 个状态量，也可能是完全不同的对象。因此数据卡必须同时写出每一维的语义，而不只是长度。

深度学习框架对图像维度的顺序也有约定。PyTorch 的图像默认是 $[B,C,H,W]$，视频是 $[B,T,C,H,W]$，通道轴靠前；而 NumPy 存储的视频常常是 $[B,T,H,W,C]$，通道轴在最后。两种顺序之间转换需要显式交换维度：

```python
video = video.permute(0, 1, 4, 2, 3)  # [B,T,H,W,C] -> [B,T,C,H,W]
```

顺序错了，模型会把宽当成通道，训练看似能跑，结果却毫无意义。

## T+1 个观察对应 T 个动作

四张连续图片之间只发生了三个动作：

$$
o_0 \xrightarrow{a_0} o_1 \xrightarrow{a_1} o_2 \xrightarrow{a_2} o_3.
$$

观察有 $T+1$ 个，动作只有 $T$ 个，因为动作 $a_t$ 发生在观察 $o_t$ 和 $o_{t+1}$ 之间。一段 episode 的基本 shape 因此是：

```text
observations: [T + 1, ...]
actions:      [T, ...]
rewards:      [T]
dones:        [T]
```

如果把动作整体错开一格对齐，模型仍可能靠画面惯性猜中下一帧，却永远学不会“换了动作，未来会变”。这是动作条件学习里最常见、也最隐蔽的错误。

## Episode 与 transition

一条 **transition** 保存一次完整的变化：

$$
(o_t,\; a_t,\; r_t,\; o_{t+1},\; d_t),
$$

其中 $r_t$ 是这步奖励，$d_t$ 是一个布尔量，表示这一步是否触发了环境重置。许多条连续 transition 串起来，就组成一条 **episode**。

`done=True` 之后，下一条记录不能直接接上一段新起点。否则模型会把“环境重置”当成一种普通动态，认为从终点能凭空跳回起点。正确做法是在 `done` 处切断，分别保存每段 episode。

不同长度的 episode 处理起来麻烦。我们可以从内部采样固定长度的小片段，也可以 padding 后配一张 mask 标出有效位置。本课程优先采用前者，把注意力先放在动态本身。

## 时间信息不能省略

同样移动 5 厘米，用时 $0.1$ 秒和 $1$ 秒代表完全不同的速度。所谓速度，就是位移除以时间：$v=\Delta x/\Delta t$。缺少 $\Delta t$，模型无法把像素位移换算成物理量。

因此动作数据至少要记录：观察时间戳、动作时间戳、控制频率、观察频率和执行延迟。机器人与驾驶数据还可能存在动作重复、跳帧和控制器内部延迟。这些元数据错了，换更大的网络也补救不了。

## 小结

- [ ] shape 要同时标出每一维的长度与含义。
- [ ] $T+1$ 个观察对应 $T$ 个动作，$a_t$ 落在 $o_t$ 与 $o_{t+1}$ 之间。
- [ ] transition 是一次变化，episode 是不跨重置的连续经历。
- [ ] 时间戳与控制频率是动作条件学习的前提，不能省略。

---

## 参考资料

### 实践博客

1. [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) —— 图解 Transformer 的经典博客，配 2.2 的 ViT 与 2.3 的注意力记忆。
2. [Understanding LSTM Networks (Chris Olah, 2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) —— RNN 与 LSTM 的直觉讲解，配 2.3 的记忆组件。
3. [The Unreasonable Effectiveness of Recurrent Neural Networks (Karpathy, 2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) —— 用大量可视化展示序列模型怎样“记住”过去，配 2.3。
4. [What are Diffusion Models? (Lilian Weng, 2021)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) —— 扩散模型从公式到采样的系统梳理，配 2.4 与第 5 章。
5. [Transformers for Image Recognition at Scale (Google AI, 2020)](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) —— ViT 官方解读博客，讲清 patch 化与规模效应，配 2.2。

### 经典文献

1. [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) —— Transformer 原始论文，2.2、2.3 注意力的来源。
2. [An Image is Worth 16x16 Words: ViT (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929) —— Vision Transformer 原始论文，patch token 的来源。
3. [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735) —— LSTM 原始论文，记忆组件的奠基工作。
4. [Neural Discrete Representation Learning: VQ-VAE (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937) —— 离散 token 表示的奠基论文，第 5 章视频分词器的直接前身。
5. [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) —— DDPM 原始论文，2.4 与第 5 章扩散路线的数学基础。
