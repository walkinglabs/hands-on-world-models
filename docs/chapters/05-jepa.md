# 第 5 章　JEPA：未来的每个像素都值得预测吗

路线 B 希望把未来画出来。它会遇到一个麻烦：一片树叶向左抖还是向右抖，可能很难预测，却不影响机器人绕开桌子。

如果模型把大部分能力花在不可预测的纹理上，也许会忽略物体、运动和空间这些更稳定的信息。

JEPA 改变的不是“用不用 Transformer”，而是预测目标：不比较生成像素与真实像素，而在表示空间中比较预测特征与目标特征。

## 5.1 这条路线交出什么

被动视频 JEPA 读入可见时空块，预测被遮区域或未来片段的 feature embedding。

```text
可见视频 → Context Encoder → context features
被遮/未来视频 → Target Encoder → target features
context features → Predictor → predicted features
```

训练让 predicted features 接近 target features。

这里没有 Decoder，所以结果不能直接画回完整未来。我们改用下游检查：位置、速度或动作能否从 feature 中读出，表示是否帮助分类、预测或规划。

Action-JEPA 再加入动作：

```text
历史 feature + 候选动作 → 未来 feature
```

只有这一步完成以后，模型才有资格回答“换一个动作会怎样”。被动 V-JEPA 可以是很好的视频表示模型，但它本身不证明可控规划。

这条路线同样只依赖第 0–2 章。ViT、EMA 和 stop-gradient 会在当前问题里重新讲，不要求先学 Dreamer 或互动视频。

## 5.2 从图片 patch 到视频 tubelet

ViT 把图片切成许多 patch。每块展开以后，经线性层变成 token。

视频多了时间维度。我们可以逐帧切 patch，也可以把相邻几帧同一位置合成 tubelet。

```text
图片 patch：H_patch × W_patch
视频 tubelet：T_patch × H_patch × W_patch
```

Tubelet 让一个 token 同时覆盖短期运动。它也会降低时间分辨率：很快的小物体可能在一个 tubelet 内移动很远。

C1 为了保持代码短，先逐帧切 `4×4` patch。理解接口以后，PA 可以把时间也合并进 patch embedding。

## 5.3 Masking 在问什么

若 Encoder 看见完整视频，再让 Predictor 输出同一份特征，任务太容易。

我们遮住一部分区域，只给 context encoder 其余内容。Predictor 必须根据可见线索猜被遮部分。

视频掩码不应只是随机散点。一个大时空块能迫使模型利用运动和物体连续性；短块与长块混合，可以同时训练局部和较长时间关系。

掩码策略其实在定义课程问题：

- 遮空间区域，模型要理解周围结构；
- 遮未来帧，模型要理解动态；
- 遮一个移动物体，模型要保持物体连续性；
- 遮太多，任务可能变得没有足够证据。

## 5.4 为什么需要两个 Encoder

Context encoder 从可见部分提取特征。Target encoder 看见真实目标区域，交出训练目标。

若两个 Encoder 和 Predictor 一起自由更新，系统可能找到一种无用解：无论输入什么，都输出同一个常量向量。预测和目标完全相同，loss 为零，却没有保存任何世界信息。

这叫表示坍缩。

## 5.5 stop-gradient 与 EMA 做了什么

Target encoder 的输出不接收当前 loss 的反向梯度：

```python
target = target_encoder(x).detach()
```

它的参数通过 online encoder 的指数移动平均更新：

```text
target = m × target + (1-m) × online
```

当 `m` 接近 1，target 变化较慢。Predictor 追逐的是一个缓慢移动的目标，而不是和目标一起瞬间滑向常量。

EMA 可以提高稳定性，却不是“有用表示”的证明。所有 feature 仍可能变化很小，或者只保存容易预测却与任务无关的信息。

## 5.6 怎样检查表示没有学空

第一项是 feature spread：把许多样本与位置的 feature 放在一起，计算每个维度的标准差。

接近零值得警惕，但标准差大也不等于信息有用。噪声同样可以有很大方差。

因此还要看：

- 不同输入之间的平均距离；
- feature 各维是否高度重复；
- 最近邻视频是否具有相似物体或运动；
- 一个简单 probe 能否读出已知属性。

这些检查共同回答“表示里还剩下什么”。

## 5.7 Linear Probe 为什么有用

我们冻结 Encoder，只训练一个很弱的线性模型，从 feature 预测方块位置。

若线性 probe 做得好，说明位置信息已经以较容易读取的方式存在 feature 中。若一定要训练一个很深的新网络，可能是新网络重新从复杂表示中学习，而不是原表示已经清楚保存信息。

Probe 仍有边界：

- 能读出位置，不代表能读出碰撞或因果关系；
- 训练集 probe 好，不代表 OOD 视角仍好；
- 一个任务认为无关的信息，换任务以后可能很重要。

C2 会把 probe 与“永远预测平均位置”的基线比较。PA1-C 还要按 episode 切分，在测试集报告结果。

## 5.8 从被动观看走向动作条件

被动视频告诉模型世界通常怎样变化，却可能把人的动作、相机运动和物体自身运动混在一起。

当数据包含真实动作，我们把 action embedding 交给 Predictor：

```text
历史 context + action + target position
→ future feature prediction
```

然后固定同一历史，逐一替换动作。若 predicted feature 完全不变，模型可能忽略动作。

与互动视频一样，动作敏感性有不同强度：

1. feature 数值随动作改变；
2. probe 读出的未来位置随动作正确改变；
3. 多步 feature rollout 保持动作效果；
4. Planner 使用预测后，真实成功率提高。

C2 smoke 只建立前两项接口。PA1-C 的动作版本必须报告第三或第四项。

## 5.9 怎样用 feature 做一个最小规划器

假设目标图片经过 Target encoder 得到 `goal_feature`。对于每个候选动作，Action-JEPA 预测 `future_feature`。

最简单的选择是：

```text
选择与 goal_feature 距离最近的 future_feature
```

这让 feature prediction 接到行动，但仍留下许多问题：

- feature 距离是否真的对应任务进展？
- 一步离目标近，会不会走进死路？
- 不同可能未来怎样表示？
- 模型没有把握时，Planner 是否仍会强行选择？

更完整的方法需要多步动作条件预测、目标或 cost，以及 MPC。这里与路线 A 重新相遇，但世界表示和训练目标已经不同。

## 5.10 被动视频数据与动作数据不能混着下结论

### PixelWorld

同一生成器既能交出视频，也能交出准确动作。它适合逐层检查 feature、probe 与反事实。

### UCF101-mini

它包含真实人类动作视频，可用来检查表示迁移和动作类别 probe。但它没有我们能在每一帧施加的控制输入。

因此，UCF101 上的好结果只能支持“被动视频表示有用”，不能支持“模型懂得机器人控制”。

### 机器人动作数据

Action-JEPA 需要时间对齐的 observation、action、下一 observation，最好还记录控制频率、延迟与机器人自身状态。第 6 章会继续使用这种数据。

## 5.11 JEPA 不等于什么

JEPA 不是一种固定大小的 ViT，也不是“不要像素就一定更聪明”。它是一类 joint-embedding prediction 思路。

它不自动保证：

- feature 保存所有下游任务需要的信息；
- 表示符合真实三维结构；
- 预测具有可靠概率；
- 被动视频模型能够响应控制；
- feature 距离可以直接当作 reward。

这些边界不是缺点清单，而是实验应该继续问的问题。

## 5.12 C1、C2 与 PA1-C

C1 完成：

```text
视频 patch
→ context / target encoder
→ masking
→ stop-gradient + EMA
→ feature spread
```

C2 完成：

```text
linear probe
→ 动作条件 predictor
→ 反事实 feature
→ 一步候选动作选择
```

PA1-C 先在 PixelWorld 完成被动与动作条件对照，再选做 UCF101-mini 的被动迁移。两类数据的结论必须分开写。

## 5.13 读论文时问什么

- [V-JEPA](https://arxiv.org/abs/2404.08471) 如何只用 feature prediction 学视频表示？
- Target encoder、EMA 与掩码策略分别承担什么职责？
- V-JEPA 2 的被动预训练与动作条件阶段为什么需要不同数据？
- 哪些任务确实不需要重建像素，哪些任务会因丢失细节而失败？
- 若 Predictor 能在 feature 空间规划，怎样证明它不是在利用表示漏洞？

下一章转向机器人。VLA 先负责提出动作；世界模型则可以在动作执行以前，预测或检查它的后果。
