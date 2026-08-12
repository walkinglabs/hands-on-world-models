# 3.3　动作条件 Transformer：一帧一帧生成未来

每帧已经变成 token 网格。接下来把历史 token 与动作排成序列，预测下一帧 token。

## 序列怎样排列

一种简单排列是：

```text
[frame_0 tokens] [action_0] [frame_1 tokens] [action_1] ...
```

模型还要区分空间位置、时间步和 token 类型。位置编码说明“这是第几帧的哪个格子”，type embedding 区分视觉与动作。

## 因果遮罩

预测下一 token 时，只能查看历史与已经生成的 token，不能看到未来答案。训练中的 mask 若错误，会出现数据泄漏：loss 很低，部署时却无法生成。

## 动作怎样进入模型

离散按键可以使用 embedding，连续控制可以经过小 MLP 投影。动作 token 必须位于它所影响的未来帧之前。

有些模型把动作加到每层特征，有些使用 cross-attention。课程先采用最直接的 token 条件，方便做“去掉动作”的消融。

## Teacher forcing 与部署差异

训练时模型通常看到真实历史 token；生成时则不断读入自己刚预测的 token。一步小错误会改变后续输入，形成多步漂移。

因此不能只报告 next-token accuracy。B2 会从同一起点生成 10、30 和 100 步，并测量物体位置、动作方向与失败时间。

## KV Cache 与实时性

自回归生成会重复处理历史。KV Cache 保存过去层的 key 和 value，减少每步计算。实际交互还要统计 tokenizer、Transformer 和 Decoder 的总延迟。

## 小结

- [ ] 历史画面 token、动作 token 与位置编码共同组成序列。
- [ ] 因果遮罩防止未来泄漏。
- [ ] Teacher forcing 的一步成绩不能替代自由生成的多步评价。
