# 3.1　把图片压成可预测的状态

PixelWorld 的每张观察是 `16×16×3` 图片。直接预测 768 个像素并非做不到，但规划器真正需要的是方块位置、速度、目标和碰撞等信息。

本路线先把图片编码成较短 latent，再学习 latent 怎样随动作变化。

## Encoder 的接口

```text
observation [B,C,H,W]
→ CNN Encoder
→ embedding [B,D]
```

embedding 是当前图片的特征，不自动包含历史。两个末帧相同、运动方向不同的片段仍然需要记忆模型区分。

## 动态状态不等于图像特征

我们把当前 embedding、上一状态和上一动作交给状态模型：

```text
state_t = update(state_{t-1}, embedding_t, action_{t-1})
```

这个 `state_t` 才是供 reward head、value head、Decoder 和 Actor 使用的动态状态。

## 为什么仍保留 Decoder

Decoder 把 latent 还原成观察，为状态提供密集训练信号。它不是本路线的最终产品：即使重建画面很好，latent 仍可能丢掉奖励、终止或动作造成的细小变化。

因此还要同时预测 reward 与 continue，并用真实环境 return 检查状态是否对控制有用。

## 一步 latent 预测的基线

最简单基线是复制当前 latent，或者用线性层预测下一 latent。若环境大多静止，这些基线可能得到不低分数。

真正的检查是固定历史、替换动作，并连续 rollout。模型若只学到惯性，不会随动作产生正确分叉。

## 本路线暂时不追求什么

我们不要求 latent 解码成高清可观看视频。若最终使用者是人或游戏玩家，第 4 章的互动视频路线更合适。

## 小结

- [ ] 图像 embedding 表示当前画面，动态状态还要结合历史和动作。
- [ ] Decoder 是训练信号，不是决策路线的最终指标。
- [ ] latent 是否有用，由 reward、continue、多步预测和真实控制检查。
