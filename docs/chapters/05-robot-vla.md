# 第 5 章　VLA 与机器人：会看会说以后，怎样真正动起来

视觉语言模型可以看见杯子，也能理解“把红杯子放到托盘里”。机器人最后却必须交出速度、位姿或关节命令。

把图像与文字直接映射成动作，就是 VLA 的核心工作。可是，一个动作模型可以做得很快，却不一定在执行以前回答“这一步会不会撞到旁边的盘子”。

本章把两个职责分开学习：

```text
VLA / Policy：根据图像、指令和机器人状态提出动作
Outcome / World Model：根据当前状态和候选动作预测后果
```

它们可以组合，也可以单独存在。直接 VLA 不是世界模型。

## 5.1 机器人数据为什么比图片分类多几条线

一条可用示范至少包含：

```text
image_t
instruction
proprioception_t
action_t
image/state_{t+1}
timestamp 与 control frequency
```

`proprioception` 是机器人对自身的读数，例如关节角、夹爪状态或末端位置。单张图片可能看不清精确关节状态，因此它不是可随意删除的重复信息。

时间仍然重要。相机有曝光和传输延迟，动作指令也要经过控制器才生效。若 `action_t` 实际对应两帧后的变化，直接按相邻帧训练会把延迟误认为物理规律。

项目内 Tabletop 数据把这些变量做得很小：一张桌面图片、红绿两个目标、抓手位置、一个障碍、两条语言指令与二维动作。

## 5.2 先做最简单的行为克隆

行为克隆把专家轨迹当作监督数据：

```text
observation → Policy → predicted action
```

然后最小化预测动作与专家动作的差异。

我们先做 state-only 基线，只读抓手、目标和障碍坐标。若这个模型已经很好，加入视觉模型以后却没有提升，说明图片可能没有提供额外信息，或视觉 Encoder 没有学好。

行为克隆有一个根本限制：训练数据来自专家访问的状态。模型犯一个小错以后，会走到专家很少访问的位置；接下来的误差可能继续累积。这叫分布偏移。

因此动作 MSE 只是训练检查，最终仍要在环境中运行整段任务。

## 5.3 图像、语言和机器人状态怎样接起来

教学版 Tiny VLA 有三条输入：

```text
image → CNN → visual feature
instruction id → Embedding → language feature
proprioception → state feature
三者拼接 → Policy head → action
```

工业 VLA 常使用更大的视觉语言预训练与 token 化动作。RT-2 将机器人动作表示成与文字相同形式的 token；OpenVLA 等工作继续探索开源 VLA。本书不在 24GB 上重新预训练大型视觉语言骨干，只实现 VLA 的输入输出与数据关系。

### 同一画面换指令

一张桌面图同时含红色和绿色目标。固定图片与机器人状态，只把“去红色”换成“去绿色”，动作应改变。

这和前两章的反事实相同：一次只改变一个条件，检查模型是否真的使用它。

## 5.4 为什么一次输出一段动作

逐帧输出动作会受到视觉噪声与推理延迟影响，动作方向可能左右抖动。

Action chunking 一次预测未来 `K` 步：

```text
当前输入 → [a_t, a_{t+1}, ..., a_{t+K-1}]
```

它可以减少推理次数，让局部动作更连贯。代价是环境突然变化时，后半段动作可能已经过时。

实际系统可以只执行 chunk 前几步，再重新观察。Chunk 长度、控制频率和重规划间隔必须一起报告。

## 5.5 动作不只有一个正确答案

绕过障碍可以从左，也可以从右。若用 MSE 回归两种专家动作的平均，结果可能正好指向障碍。

常见办法包括：

- 输出混合分布；
- 离散化动作 token；
- 用 Diffusion 或 Flow 从条件分布采样动作 chunk；
- 先产生多个候选，再由 critic 或 world model 重排。

D1 只使用确定性 chunk，故意保留这个失败。学生在 PA1-D 选做多峰动作头，不额外增加一份 Notebook。

## 5.6 为什么需要在行动以前检查后果

假设 Tiny VLA 提出四个候选方向。我们训练一个一步 outcome model：

```text
当前 state + candidate action
→ next state prediction + collision probability
```

然后计算每个候选动作：

```text
目标距离 + 碰撞惩罚
```

选择得分更高的动作。

这就是最小 lookahead。VLA 负责“可以怎么做”，world model 负责“这样做可能发生什么”，reranker 或 planner 负责比较。

## 5.7 为什么 outcome 数据要包含坏动作

专家示范大多安全。如果 outcome model 只看专家数据，collision 标签可能几乎全是 0。它不能学会识别从未见过的碰撞动作。

D2 额外采样随机候选动作，并故意把一部分障碍放在动作前方。这里出现一个重要的数据分工：

- Policy 数据告诉模型好动作长什么样；
- Outcome 数据告诉模型各种动作会发生什么，包括失败。

真实机器人不能随意撞击收集数据。可以使用模拟器、安全约束、人工演示、离线失败记录或小幅探索，但必须说明数据与现实的差距。

## 5.8 模型检查器也会想错

若 world model 预测某个危险动作“很安全”，reranker 可能比直接 VLA 更糟。

我们要分别检查：

1. VLA 的候选集合里有没有可行动作；
2. Outcome model 的下一状态误差；
3. 碰撞概率是否校准；
4. 重排是否提高真实成功率；
5. 推理延迟是否破坏控制频率。

模块分开以后，失败更容易定位。联合端到端训练可能更强，却要设计额外诊断，避免“想错了”和“做错了”混在一起。

## 5.9 D1、D2 与 PA1-D

D1 从 state-only 行为克隆走到 image + instruction + proprioception，并输出 action chunk。

D2 用另一份动作后果数据训练 next-state/collision model，再比较直接动作与 lookahead reranking。

PA1-D 在 Tabletop 完成闭环成功率与碰撞率，再迁移到 PushT 小数据。若使用 LIBERO 等语言多任务数据，必须保存原始 instruction，而不是只留下 task id。

24GB 目标是小视觉 Encoder、小语言 Encoder 与短 action chunk；当前没有完整训练实测。

## 5.10 读论文时问什么

- [RT-2](https://arxiv.org/abs/2307.15818) 为什么把动作也表示成 token？
- [OpenVLA](https://arxiv.org/abs/2406.09246) 怎样组织视觉语言预训练与机器人数据？
- Action chunk 解决了什么，又牺牲了什么响应能力？
- Diffusion action head 在多解任务中为什么可能优于 MSE？
- 加入 world model checker 后，真实成功率是否提高，还是只增加延迟？

下一章把注意力从机器人动作移到空间本身：跨视角以后，哪些东西仍应该保持在同一位置？
