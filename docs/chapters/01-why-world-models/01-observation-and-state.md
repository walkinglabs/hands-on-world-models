# 1.1 观测与状态（Observation and State）

> **本章导读**
>
> **讲什么：** 本章先回答一个根问题：智能体看到一张画面后，为什么还不能据此判断接下来会发生什么。我们将从观测与状态的区别出发，逐步得到世界模型的基本接口，并在一个交互式驾驶环境中完成第一次“观察—预测—选择—验证”。
>
> **为什么从这里开始：** 一张道路照片可以告诉我们汽车在哪里，却不能告诉我们它正在加速还是刹车。只要当前画面没有保留决定未来所需的全部信息，模型就必须记住历史、考虑动作，并表示未来的不确定性；世界模型正是为这个困难而出现的。
>
> **故事线：** `单帧观测不够 → 从历史中形成状态 → 用状态和动作预测未来 → 把高维画面压缩后再预测 → 在驾驶任务中检验预测能否帮助选择动作`

在构建智能体（Agent）或世界模型（World Model）之前，需要先区分两个容易混用的概念：**观测（Observation）** 与 **状态（State）**。

<div align="center">
  <img src="/figures/01-why-world-models/source/01-observation-and-state/drqn-fig3.png" alt="DRQN 的连续帧卷积响应展示单帧难以提供的运动线索怎样在时间上下文中显现。" width="86%">

_图 1.1-1：DRQN 的连续帧卷积响应展示单帧难以提供的运动线索怎样在时间上下文中显现。 出处：Matthew Hausknecht; Peter Stone，[Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)（2015），Figure 3。_

</div>

20 世纪 60 年代，Kálmán 用状态变量、观测方程与递推估计系统化地处理了不可直接观测的动态状态 [[Kalman, 1960]](https://doi.org/10.1115/1.3662552)。Åström 随后研究了状态信息不完备时的马尔可夫控制问题，为 POMDP 的早期形式化奠定了基础 [[Åström, 1965]](https://doi.org/10.1016/0022-247X(65)90154-X)。现代 World Models [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122) 与 Dreamer [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603) 延续了“由观测推断内部状态，再用内部状态预测与决策”的思路；这里的状态由神经网络从高维观测中学习，而不是预先写定。

<div align="center">
  <img src="/figures/01-why-world-models/source/01-observation-and-state/wm-fig19.png" alt="World Models 的记忆示意图展示感知信息如何经短期记忆、复述与巩固进入长期记忆。" width="86%">

_图 1.1-2：World Models 的记忆示意图展示感知信息如何经短期记忆、复述与巩固进入长期记忆。 出处：David Ha; Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 19。_

</div>

本节从简单运动学出发，再过渡到贝叶斯滤波，说明两者的区别，以及如何利用历史观测估计不可直接看见的状态。

## 物理直觉：为什么我们需要状态？

为了让抽象的数学概念落地，让我们先假设一个最简单的高中物理场景：一辆在直线上行驶的汽车。

我们在不同时刻，利用布置在路边的雷达测距仪记录下汽车的位置。在最简单的标量情形下，我们可以用一个实数表示汽车在时刻 $t$ 的位置，记为 $x_t \in \mathbb{R}$。

根据牛顿运动学定律，如果我们要预测汽车在下一个离散时刻 $t+1$ 的位置 $x_{t+1}$，仅仅知道当前时刻的位置 $x_t$ 显然是不够的。我们还必须知道汽车此刻的运动趋势，也就是它的瞬时速度 $v_t$（如果在加速，还需要知道加速度 $a_t$）。

如果我们手中*只有*对位置的记录，即系统对外输出的**观测（Observation）**仅为 $o_t = x_t$，那么对于系统未来演化的预测将不可避免地依赖于更早的历史信息。具体来说，我们只能利用相邻两次的观测来近似估算当前的速度：

$$ v_t \approx \frac{x_t - x_{t-1}}{\Delta t} $$

将该公式代入位移公式，下一时刻的位置预测可以写成：

$$ x_{t+1} = x_t + v_t \Delta t \approx x_t + (x_t - x_{t-1}) = 2x_t - x_{t-1} $$

这个等式说明：若观测中只有位置，预测下一步时至少还要查阅 $t-1$ 时刻的位置。系统越复杂，所需的历史窗口通常越长；直接保存并反复处理整段历史，会让表示长度随时间增长。

为了避免每次都回看整段历史，可以构造**状态（State）**：相对于当前预测或控制任务，它应当概括历史中仍会影响未来的必要信息。对于上述匀速运动的汽车，位置与速度组成了一个合适的状态：

$$ \mathbf{s}_t = \begin{bmatrix} x_t \\ v_t \end{bmatrix} \in \mathbb{R}^2 $$

在速度保持不变、采样间隔固定的假设下，状态转移可以写成固定矩阵：

$$ \mathbf{s}_{t+1} = \begin{bmatrix} x_{t+1} \\ v_{t+1} \end{bmatrix} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_t \\ v_t \end{bmatrix} = \mathbf{A} \mathbf{s}_t $$

对比这两个公式，我们可以得出状态向量最关键的性质：**给定当前的完整状态后，未来的条件概率分布不再依赖更早的历史。** 这就是**马尔可夫性质（Markov Property）**；其名称来自 Markov 对相关随机序列的早期研究 [[Markov, 1906]](https://www.mathnet.ru/eng/im8054)。这里说的是条件独立性，并不意味着随机系统的未来轨迹已经被唯一确定。

## 马尔可夫性质与部分可观测环境

让我们将上述理想且确定性的经典力学系统，推广到充满随机噪声和未知干扰的概率论范畴。在现代机器学习范式中，系统状态不仅受自身内在规律支配，还会受到外界智能体（Agent）注入的控制动作（Action） $A_t$ 的干预。

设大写字母 $S_t$ 为随机变量，表示系统在时刻 $t$ 的真实隐藏状态。如果过程满足马尔可夫性质，那么在给定当前状态 $S_t$ 和采取的动作 $A_t$ 的前提下，系统转移到下一时刻任何可能状态 $S_{t+1}$ 的条件概率分布，完全与 $t$ 时刻之前的任何更古老的状态或动作无关：

$$ \mathbb{P}(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0) = \mathbb{P}(S_{t+1} \mid S_t, A_t) $$

这个条件独立关系是强化学习的重要建模基础。真实系统中，完整状态 $S_t$ 往往不能直接取得，智能体通常只能接收传感器产生的**观测（Observation）** $O_t$。

相机画面可以看作底层状态经过观测模型后的投影：遮挡、视角和传感器噪声都会丢失信息。因此，学习到的内部状态并不是对真实世界的直接拷贝，而是面向预测与决策的历史摘要。

我们将这种只能获得局部、有损信息的框架称为**部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）**。在 POMDP 中，底层状态转移满足马尔可夫性质，而观测由观测概率模型产生：

$$ O_t \sim \mathbb{P}(O \mid S_t) $$

由于 $O_t$ 往往只是状态的有损投影，观测序列通常不能保证具有马尔可夫性质。若只截取一帧赛车画面，模型很难区分赛车正在加速、刹车还是匀速前进；这时需要结合历史线索，形成用于预测和决策的内部状态。不过，如果某类观测已经包含任务所需的充分信息，它也可能近似满足马尔可夫性质。

## 贝叶斯滤波：从历史序列提取状态表示

既然孤立的单一观测 $O_t$ 无法提供充分的信息，最直观的破局之道是将有史以来的所有观测和动作全部累积起来，形成一条严密的**历史轨迹（History）** $H_t$：

$$ H_t = (O_1, A_1, O_2, A_2, \dots, A_{t-1}, O_t) $$

$H_t$ 保留了从起点到当前时刻的信息，因此可以作为状态使用；代价是它的长度会随时间近似线性增长，存储与计算成本也随之增加。

数学上，最优的解决方案是不直接记录无穷的历史序列，而是利用概率论递归地维护一个对当前真实状态 $S_t$ 的概率置信度。我们将给定所有已知历史条件下的真实状态概率分布称为**信念状态（Belief State）**，记为向量或函数 $b_t(S) = \mathbb{P}(S_t = S \mid H_t)$。

通过严密的贝叶斯公式（Bayes' Rule），我们可以推导信念状态的递归更新过程（即贝叶斯滤波，Bayesian Filtering）：

<div align="center">
  <img src="/figures/01-why-world-models/source/01-observation-and-state/planet-fig11.png" alt="PlaNet 的状态诊断曲线显示潜在动力学状态可读出位置、速度与接触等未直接给定的物理量。" width="86%">

_图 1.1-3：PlaNet 的状态诊断曲线显示潜在动力学状态可读出位置、速度与接触等未直接给定的物理量。 出处：Danijar Hafner et al.，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 11。_

</div>

1. **预测步（Prediction Step）：** 在接收到新的动作 $A_{t-1}$ 后，根据系统的状态转移模型 $\mathbb{P}(S_t \mid S_{t-1}, A_{t-1})$ 和上一步的信念状态 $b_{t-1}(S_{t-1})$，结合全概率公式预测当前的状态分布：
   $$ \mathbb{P}(S_t \mid H_{t-1}, A_{t-1}) = \sum_{S_{t-1}} \mathbb{P}(S_t \mid S_{t-1}, A_{t-1}) b_{t-1}(S_{t-1}) $$

2. **更新步（Update Step）：** 在接收到最新的观测 $O_t$ 后，利用观测模型 $\mathbb{P}(O_t \mid S_t)$ 更新并归一化状态信念：
   $$ b_t(S_t) = \mathbb{P}(S_t \mid H_t) = \frac{\mathbb{P}(O_t \mid S_t) \mathbb{P}(S_t \mid H_{t-1}, A_{t-1})}{\mathbb{P}(O_t \mid H_{t-1}, A_{t-1})} \propto \mathbb{P}(O_t \mid S_t) \mathbb{P}(S_t \mid H_{t-1}, A_{t-1}) $$

合并预测步与更新步，可得到递归形式：

$$ b_t(S) \propto \mathbb{P}(O_t \mid S) \sum_{S_{t-1}} \mathbb{P}(S \mid S_{t-1}, A_{t-1}) b_{t-1}(S_{t-1}) $$

<div align="center"><img src="/figures/01-why-world-models/latex/01-observation-and-state/bayes-predict-update.png" alt="先按状态转移汇聚上一信念，再乘观测似然并归一化" width="86%">

_图 1.1-4：预测步先对旧状态求和得到先验信念；更新步再乘最新观测似然并归一化。_

</div>

新的信念状态 $b_t$ 由上一时刻的信念 $b_{t-1}$、动作 $A_{t-1}$ 和最新观测 $O_t$ 递归更新。

在高维问题中，精确计算这套积分往往不可行。因此，现代架构常用 RNN、GRU 等神经网络，把历史压缩为定长隐藏状态 $\mathbf{h}_t$：

$$ \mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, A_{t-1}, O_t) $$

$\mathbf{h}_t$ 可以看作学习得到的有限维历史摘要。它与信念状态有相似的递归结构，但只有在训练目标和模型能力足够时，才可能接近充分统计量；它也不一定是经过校准的概率信念。后续介绍的循环状态空间模型（RSSM）会把这种递归表示与随机潜变量结合起来。

## 观测与状态的张量操作实战

下面用张量操作实现一个有限历史窗口。

对于单帧 2D 图像这类信息不完备的观测，一种常见做法是**帧堆叠（Frame Stacking）**。DQN 就把最近四帧预处理后的图像组合为网络输入 [[Mnih et al., 2015]](https://doi.org/10.1038/nature14236)。连续 $K$ 帧能够提供有限时间窗口内的运动线索；它不能保证恢复完整状态，也不像循环模型那样可以汇总更长的历史。

假设环境传回的观测 $O_t$ 是一帧单通道灰度图，张量形状为 $(H, W)$。通过堆叠过去 $K$ 帧观测，我们将构造一个形状为 $(K, H, W)$ 的三维状态张量。

代码创建一个循环缓冲区，用 PyTorch 实现帧堆叠，为卷积网络提供带有短期运动线索的输入。

```python
import torch

class ObservationBuffer:
    """用于存储有限长度历史观测并构造马尔可夫状态的缓冲区"""
    def __init__(self, k_frames, height, width):
        self.k_frames = k_frames
        self.height = height
        self.width = width
        # 初始化全零张量，形状为 (K, H, W)。
        # K 表示我们堆叠的时间帧数量，通常作为传入 CNN 的 Channel 维度。
        self.buffer = torch.zeros((k_frames, height, width), dtype=torch.float32)

    def add_observation(self, obs):
        """
        接收一个新的观测帧，丢弃超出 K 视野的最旧一帧
        """
        assert obs.shape == (self.height, self.width), "输入观测的形状与初始化不匹配"
        # 利用 PyTorch 的切片操作，将第 1 到 K-1 帧的数据向前“滑动”一位。
        # 这一步抹除了原始张量索引为 0 的最旧数据。
        self.buffer[:-1] = self.buffer[1:].clone()
        # 将最新到达的观测画面存入缓冲区的尾部（最近的时间点）
        self.buffer[-1] = obs

    def get_state(self):
        """
        提取用于智能体决策的近似信念状态。
        返回张量形状为: (K, H, W)
        """
        # 必须返回克隆数据，以切断 PyTorch 的显存引用共享，
        # 防止外部操作意外污染缓冲区内保留的历史记录。
        return self.buffer.clone()

# 设定输入场景为 84x84 (强化学习中 Atari 游戏的经典预处理分辨率)
H, W = 84, 84
# 设定时间窗口，堆叠 4 帧连续图像
K = 4

# 实例化历史缓冲区
buffer = ObservationBuffer(k_frames=K, height=H, width=W)

# 模拟智能体在未知环境中执行 5 个离散时间步的交互循环
for t in range(5):
    # 此处利用正态分布随机张量模拟传感器传回的高维含噪观测数据
    current_obs = torch.randn((H, W))

    # 将最新观测推入时间缓冲区
    buffer.add_observation(current_obs)

    # 提取封装了过去 K 帧历史特征的最新状态张量
    current_state = buffer.get_state()
    print(f"时间步 {t+1}: 输出状态张量维度 {current_state.shape}")
```

帧堆叠把 $K$ 张平面观测组合为 $(K,H,W)$ 张量。二维卷积的首层会同时读取这些通道，因此可以学习短时间窗口内的位移线索。它增加了局部时序信息，但不会自动恢复被遮挡的状态，也不能保证输入重新满足马尔可夫性质。

## 小结

- **观测（Observation）** 是传感器在某一时刻得到的信息；它可能局部、带噪，也不保证对当前任务充分。
- **状态（State）** 是面向预测或控制的历史摘要。给定当前状态与动作后，下一状态的条件分布不再依赖更早历史。
- 在 POMDP 中，贝叶斯滤波递归维护显式信念分布；RNN、RSSM 和帧堆叠则提供不同复杂度的近似历史表示。
