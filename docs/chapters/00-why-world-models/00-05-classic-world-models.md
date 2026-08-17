# 0.5　经典世界模型

前四篇没有使用"世界模型"的现成结构图。我们先遇到了图片太大、当前观察不完整、动作后果未知和规划会钻模型漏洞等问题。

现在再看 2018 年 Ha 与 Schmidhuber 的《World Models》，图中的三部分就不再只是缩写。论文把整台系统拆成 V、M、C 三块，正好对应我们前面一节节补上的缺口。

## V：把图片压短

论文使用变分自动编码器（Variational Autoencoder，VAE）把游戏画面 $x_t$ 压成低维的 latent 向量 $z_t$：

$$x_t \;\xrightarrow{\text{Encoder}}\; z_t \;\xrightarrow{\text{Decoder}}\; \hat x_t.$$

VAE 的训练目标是让重建 $\hat x_t$ 接近原图 $x_t$，同时约束 $z_t$ 接近一个标准正态先验。具体地，编码器输出分布参数 $(\mu,\sigma)$，再按重参数化采样：

$$z_t = \mu_\phi(x_t) + \sigma_\phi(x_t)\odot \varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,I).$$

这一步解决的是高维观察无法直接建表的问题：一张 $64\times 64\times 3$ 的画面有上万维，而 $z_t$ 可能只有 $32$ 维。压缩也会丢失信息，因此不能只看重建是否漂亮，还要看留下的信息能否支持后续控制。

## M：根据历史和动作预测

记忆模型读取 latent、动作和过去的隐藏状态 $h_t$，预测下一 latent 的分布：

$$\bigl(z_t,\ a_t,\ h_t\bigr) \;\longrightarrow\; P(z_{t+1}\mid \cdot)\ \text{和更新后的}\ h_{t+1}.$$

更新部分形如 RNN：

$$h_{t+1} = g(h_t,\, z_t,\, a_t).$$

论文使用 MDN-RNN（Mixture Density Network + RNN）表达多种未来：输出不是一个 $z_{t+1}$，而是若干高斯分布的加权和：

$$P(z_{t+1}) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}\bigl(z_{t+1};\, \mu_k,\, \sigma_k\bigr).$$

这里 $\pi_k$ 是各分量的权重。RNN 保存历史，混合密度输出描述随机结果——这正是第 0.2 节说的"保留多峰分布"。

## C：使用内部状态行动

控制器（Controller）读取 $z_t$ 与 $h_t$，线性地输出动作：

$$a_t = W_c\,[z_t;\, h_t] + b_c.$$

论文中的一个重要实验，是让控制器在记忆模型生成的"梦境"中训练，再回到真实游戏测试。若模型过于简单，控制器可能利用模型漏洞。论文通过调高 MDN 采样的温度，让内部环境更不确定，减少控制器只适应单一错误未来的机会。

这一步对应第 0.3 节的分工：世界模型 $f$ 产出未来，规划 / 策略使用未来。这里的"Controller"还很简单，真正的规划器要在 PlaNet 和 Dreamer 中才出现。

## 与 PlaNet、Dreamer 的关系

经典 World Models 把视觉表示、记忆动态和控制器拆开。PlaNet 随后引入 **RSSM**（Recurrent State-Space Model）把确定记忆与随机状态分开：用一个确定循环保留长期历史，再用随机变量描述每一步的不确定性。Dreamer 继续使用 RSSM，但在想象轨迹上训练 Actor 与 Critic。

```text
PlaNet：  RSSM + 每一步用 CEM 现场搜索动作
Dreamer： RSSM + 在想象轨迹中做 Actor-Critic 学习
```

两者的共同核心，是把世界模型写成一个可以在其中"做梦"的递推过程：

$$\bigl(h_t,\, z_t\bigr) \xrightarrow{a_t} \bigl(h_{t+1},\, z_{t+1}\bigr),\qquad \hat r_t = R(h_t,z_t,a_t).$$

PlaNet 在这条想象的链上用交叉熵方法（CEM）搜索使累计 $\hat r$ 最大的动作序列；Dreamer 则在链上算 Actor-Critic 的梯度，把好行为直接学进策略网络。

MuZero 又改变了预测目标：不要求重建画面，只学习搜索所需的 reward、policy 和 value 三个头：

$$\bigl(\hat r_t,\, \hat \pi_t,\, \hat v_t\bigr) = F(s_t,\, a_t).$$

## 为什么现在再次成为研究重点

"先预测再行动"并不新。控制系统、Dyna 和棋类搜索长期使用这一思想。过去的困难在于，高维视觉难以压缩，长时预测误差很大，动作数据昂贵，模型也来不及实时运行。

近年的视觉表示、生成模型、Transformer、GPU 与大规模视频数据缓解了其中一部分问题。与此同时，机器人、驾驶和可交互生成要求模型面对动作与现实反馈，使世界模型重新成为连接感知和行动的一条重要路线。

## 小结

- [ ] **V–M–C** 分别承担视觉压缩、动作条件记忆和控制。
- [ ] PlaNet 与 Dreamer 共用 RSSM，但一个现场搜索（CEM），一个训练 Actor。
- [ ] MuZero 不重建像素，只学习规划需要的 reward、policy 与 value。
- [ ] 世界模型不是某一种网络，而是一组关于状态、变化、动作和使用方式的设计选择。

第 1 章开始逐项建立这些设计所需的共同工具。

> 👉 动手实验：[动手：从零实现世界模型](/labs/f0)

## 参考资料

- [World Models](https://arxiv.org/abs/1803.10122)：VAE、MDN-RNN 与 Controller 的经典组合。
- [PlaNet](https://arxiv.org/abs/1811.04551)：在 latent dynamics 中使用在线规划。
- [Dream to Control](https://arxiv.org/abs/1912.01603)：在模型想象中学习行为。
- [MuZero](https://www.nature.com/articles/s41586-020-03051-4)：学习用于规划的 reward、policy 与 value。
