# 0.5　回到经典《World Models》

前四篇没有使用“世界模型”的现成结构图。我们先遇到了图片太大、当前观察不完整、动作后果未知和规划会钻模型漏洞等问题。

现在再看 2018 年 Ha 与 Schmidhuber 的《World Models》，图中的三部分就不再只是缩写。

## V：把图片压短

论文使用变分自动编码器（VAE）把游戏画面压成 latent `z_t`。

```text
图片 x_t → VAE Encoder → z_t
```

这一步解决的是高维观察无法直接建表的问题。压缩也会丢失信息，因此不能只看重建是否漂亮，还要看留下的信息能否支持后续控制。

## M：根据历史和动作预测

记忆模型读取 latent、动作和过去的隐藏状态，预测下一 latent 的分布。

```text
z_t + a_t + h_t → P(z_{t+1}) + h_{t+1}
```

论文使用 MDN-RNN 表达多种未来。RNN 保存历史，混合密度输出描述随机结果。

## C：使用内部状态行动

控制器读取 `z_t` 与 `h_t`，输出动作。论文中的一个重要实验，是让控制器在记忆模型生成的环境中训练，再回到真实游戏测试。

若模型过于简单，控制器可能利用模型漏洞。论文通过调高记忆模型采样的温度，让内部环境更不确定，减少控制器只适应单一错误未来的机会。

## 与 PlaNet、Dreamer 的关系

经典 World Models 把视觉表示、记忆动态和控制器拆开。PlaNet 随后使用 RSSM 表示确定记忆与随机状态，并在行动时用 CEM 搜索。Dreamer 继续使用 RSSM，但在想象轨迹上训练 Actor 与 Critic。

```text
PlaNet：RSSM + 每一步 CEM 搜索
Dreamer：RSSM + 想象中的 Actor-Critic 学习
```

MuZero 又改变了预测目标：不要求重建画面，只学习搜索所需的 reward、policy 和 value。

## 为什么现在再次成为研究重点

“先预测再行动”并不新。控制系统、Dyna 和棋类搜索长期使用这一思想。过去的困难在于，高维视觉难以压缩，长时预测误差很大，动作数据昂贵，模型也来不及实时运行。

近年的视觉表示、生成模型、Transformer、GPU 与大规模视频数据缓解了其中一部分问题。与此同时，机器人、驾驶和可交互生成要求模型面对动作与现实反馈，使世界模型重新成为连接感知和行动的一条重要路线。

## 小结

- [ ] V–M–C 分别承担视觉压缩、动作条件记忆和控制。
- [ ] PlaNet 与 Dreamer 共用 RSSM，但一个现场搜索，一个训练 Actor。
- [ ] MuZero 不重建像素，只学习规划需要的信息。
- [ ] 世界模型不是某一种网络，而是一组关于状态、变化、动作和使用方式的设计选择。

第 1 章开始逐项建立这些设计所需的共同工具。

## 参考资料

- [World Models](https://arxiv.org/abs/1803.10122)：VAE、MDN-RNN 与 Controller 的经典组合。
- [PlaNet](https://arxiv.org/abs/1811.04551)：在 latent dynamics 中使用在线规划。
- [Dream to Control](https://arxiv.org/abs/1912.01603)：在模型想象中学习行为。
- [MuZero](https://www.nature.com/articles/s41586-020-03051-4)：学习用于规划的 reward、policy 与 value。
