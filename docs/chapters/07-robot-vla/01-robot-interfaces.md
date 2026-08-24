# 7.1　机器人学习接口

> **第 7 章 · 具身智能与机器人**
>
> 视觉语言模型可以描述桌上的杯子，机器人却必须输出能执行的动作。本章从接口（7.1）走到策略（7.2、7.3），再用一个小型世界模型在执行之前检查候选动作的后果（7.4）；随后分别处理三类本体——操作与触觉（7.5）、腿式与全身（7.6），并把仿真到真机这一关单独讲清（7.7）。
>
> 一条贯穿全章的区分：直接 VLA 的输出是动作，世界模型的输出是动作后果。两者可以串联，但不能互相冒名。动作 MSE 下降不保证闭环成功，仿真通过也不等于真机成功，这两道诚实的缺口会一直保留。
>
> 本章的五份动手页依次是 [7.8 动手：从零实现 VLA 与世界模型检查器](/chapters/07-robot-vla/08-robot-vla)、[7.9 动手：机械臂的仿真与真机迁移](/chapters/07-robot-vla/09-arm-sim2real)、[7.10 动手：灵巧手的视触觉控制](/chapters/07-robot-vla/10-dexhand-visuotactile)、[7.11 动手：全身策略的仿真与真机迁移](/chapters/07-robot-vla/11-whole-body-sim2real)、[7.12 动手：VLA 与动作后果检查](/chapters/07-robot-vla/12-vla-checker)。真机部分的成本与证据要求见 [附录 B](/appendices/data-compute-delivery)。

图片分类里，一张图只配一个类别。机器人示范要复杂得多：每一步都要记下看到了什么、接到什么指令、关节处在什么状态，以及随后执行了什么动作。

我们先用一台桌面小臂把这件事说清楚。所谓机器人数据，就是把同一时刻的图像、语言、自身状态和动作，按照时间一一对齐。

## 一条机器人 episode 长什么样

小臂抓桌上的杯子。摄像机每秒拍下桌面图片，机械臂每一步报告自己的关节角和夹爪开合，人在旁边给一句“拿起红杯”。把这些连同每一步的动作存下来，就得到一条 **episode**。

记一个 episode 有 $T$ 步动作。那么观察有 $T+1$ 帧（多出一个终点观察），动作只有 $T$ 个。一条 episode 的基本字段是：

```text
images:       [T + 1, C, H, W]
proprio:      [T + 1, P]
instruction:  text 或 token
actions:      [T, A]
success/done: [T]
timestamps:   [T + 1] 和 [T]
```

`proprio` 是机器人自身状态，例如关节角、末端位置和夹爪开合。$A$ 维动作可能是关节增量、末端速度或目标位姿，数据卡必须写清坐标系和单位。

观察和动作之所以差一步，是因为动作 $a_t$ 发生在观察 $o_t$ 和 $o_{t+1}$ 之间。把动作整体错开一格对齐，模型仍可能靠画面惯性猜中下一帧，却永远学不会“换了动作，未来会变”。

## 动作空间与控制频率

同一台小臂可以有好几种动作定义，选哪一种直接决定了模型的难度：

| 动作定义               | 维度示例 | 优点                         | 代价                           |
| ---------------------- | -------- | ---------------------------- | ------------------------------ |
| 关节位置增量           | $7$      | 与硬件接口最近，执行稳定     | 与任务目标的关系间接           |
| 末端位姿增量（笛卡尔） | $6 + 1$  | 与任务语义接近，跨本体可迁移 | 需要逆运动学，奇异点附近不稳定 |
| 目标位姿（绝对）       | $6 + 1$  | 不累积漂移                   | 对标定误差敏感                 |
| 关节力矩               | $7$      | 可表达柔顺行为               | 频率要求高，数据采集困难       |

控制频率是另一个必须写进数据卡的量。$50\ \mathrm{Hz}$ 意味着每 $20\ \mathrm{ms}$ 要产出一个动作；若模型单次前向就要 $120\ \mathrm{ms}$，接口在设计阶段就已经不成立了。这个矛盾会在 [7.2](/chapters/07-robot-vla/02-imitation-and-policies) 用动作分块与实时分块正面处理。

## 先只用状态跑一个最小基线

把图像暂时丢掉，只用 `proprio` 加指令训练一个最简单的策略，是检查接口的标准做法。它能单独暴露三类错误：动作归一化写错、时间对齐错位、控制接口的坐标系或单位不一致。这三类错误一旦混进视觉模型，就很难与"视觉不够好"区分开。

这个基线还有一个作用：给后续所有模型一个下限。若加了视觉与语言之后成功率没有超过它，说明新增的模态还没有被真正用上。

## 记录规范：把失败也存下来

只保留成功轨迹的数据集有一个隐蔽的缺陷——它无法回答"哪里容易失败"。课程要求每条 episode 都带上结束原因的分类标签，例如抓空、滑落、碰撞、超时。第 9 章的失败图集完全建立在这个字段上；采集时省掉它，之后无法补。

## 小结

- [ ] 机器人数据同时包含图像、自身状态、指令、动作和时间，$T+1$ 个观察对 $T$ 个动作。
- [ ] 动作 $a_t$ 夹在 $o_t$ 与 $o_{t+1}$ 之间，错开一格对齐会让模型靠画面惯性蒙对，却学不到可控性。
- [ ] 动作定义与控制频率必须写进数据卡；频率预算在设计阶段就约束了模型规模。
- [ ] 只用状态的最小基线是检查接口的第一步，也是后续所有模型的下限。

接口摆好之后，真正的困难才开始：把示范变成能在闭环里执行的策略。下一篇 [7.2 模仿学习与生成策略](/chapters/07-robot-vla/02-imitation-and-policies) 处理复合误差、动作多模态与推理延迟这三件事。

---

## 参考资料

### 实践博客

1. [RT-2: New model translates vision and language into action (DeepMind)](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) —— 官方博客，用大量真机案例展示 VLA 的泛化与失败模式。
2. [ALOHA: A Low-cost Open-source Hardware System (Tony Zhao)](https://tonyzhaozh.github.io/aloha/) —— ALOHA 项目页：低成本双臂遥操作硬件与 ACT 策略的配套说明，配 7.1、7.2。
3. [Mobile ALOHA (Fu et al.)](https://mobile-aloha.github.io/) —— 项目页：移动底盘 + 50 条演示学出全身操作，展示小数据配方的效果。
4. [Diffusion Policy 项目页 (Chi et al.)](https://diffusion-policy.cs.columbia.edu/) —— 扩散策略的项目页，含大量对比可视化，配 7.2 的多模态动作。
5. [OpenVLA: Open-Source Vision-Language-Action Model](https://openvla.github.io/) —— 开源 VLA 项目页：配方、数据与评测全公开，适合动手复现。

### 经典文献

1. [RT-1: Robotics Transformer for Real-World Control at Scale (Brohan et al., 2022)](https://arxiv.org/abs/2212.06817) —— 首个大规模真机数据训出的通用机器人模型，本章数据与 BC 部分的参照。
2. [RT-2: Vision-Language-Action Models (Zitkovich et al., 2023)](https://arxiv.org/abs/2307.15818) —— 把动作写成 token 接进 VLM 的原始论文，配 7.3。
3. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware: ACT (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705) —— 动作分块（action chunking）的来源，ALOHA 硬件与 CVAE 策略，配 7.2。
4. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)](https://arxiv.org/abs/2303.04137) —— 用扩散模型输出多模态动作序列，是“同一上下文多条可行路线”的代表实现。
5. [OpenVLA: An Open-Source Vision-Language-Action Model (Kim et al., 2024)](https://arxiv.org/abs/2406.09246) —— 开源 VLA 的配方与评测，配 7.3 的可复现基线。
