# 1.1 观测与状态（Observation and State）
:label:`sec_observation_and_state`

在开始构建任何智能体（Agent）或世界模型（World Model）之前，我们必须首先厘清两个在日常语言中经常被混用，但在物理学、控制论与人工智能领域有着严格数学界限的概念：**观测（Observation）** 与 **状态（State）**。

早在 20 世纪 60 年代，现代控制理论的奠基人 Rudolf Kálmán 在提出状态空间表示法 `[Kalman, 1960]` 时，便将动态系统的内部真实属性（状态）与其外部可测量的输出（观测）进行了严格的数学分离。随后，在强化学习与随机过程领域，部分可观测马尔可夫决策过程（POMDP） `[Åström, 1965]` 进一步将这一理念发扬光大。直到近年来，基于世界模型架构的深度强化学习算法（例如 `[Ha & Schmidhuber, 2018]` 提出的 World Models 和 `[Hafner et al., 2019]` 引入的 Dreamer 系列），其核心思想无一例外都是：将高维、嘈杂的外部观测数据，压缩并映射为低维、紧凑且能够自主演化的潜在内部状态。

本节我们将从高中物理中的简单运动学起步，逐步平滑过渡到高阶概率论中的贝叶斯滤波理论，以最严密的数学形式揭示这两个概念的本质区别，以及如何通过历史观测去推断系统的隐性真实状态。

## 物理直觉：为什么我们需要状态？
:label:`sec_why_we_need_state`

为了让抽象的数学概念落地，让我们先假设一个最简单的高中物理场景：一辆在直线上行驶的汽车。

我们在不同时刻，利用布置在路边的雷达测距仪记录下汽车的位置。在最简单的标量情形下，我们可以用一个实数表示汽车在时刻 $t$ 的位置，记为 $x_t \in \mathbb{R}$。

根据牛顿运动学定律，如果我们要预测汽车在下一个离散时刻 $t+1$ 的位置 $x_{t+1}$，仅仅知道当前时刻的位置 $x_t$ 显然是不够的。我们还必须知道汽车此刻的运动趋势，也就是它的瞬时速度 $v_t$（如果在加速，还需要知道加速度 $a_t$）。

如果我们手中*只有*对位置的记录，即系统对外输出的**观测（Observation）**仅为 $o_t = x_t$，那么对于系统未来演化的预测将不可避免地依赖于更早的历史信息。具体来说，我们只能利用相邻两次的观测来近似估算当前的速度：

$$ v_t \approx \frac{x_t - x_{t-1}}{\Delta t} $$
:eqlabel:`eq_kinematics_velocity`

将 :eqref:`eq_kinematics_velocity` 代入位移公式，下一时刻的位置预测可以写成：

$$ x_{t+1} = x_t + v_t \Delta t \approx x_t + (x_t - x_{t-1}) = 2x_t - x_{t-1} $$
:eqlabel:`eq_kinematics_obs`

请仔细审视 :eqref:`eq_kinematics_obs` 这个等式：为了决定系统在 $t+1$ 时刻的状态演化，我们不得不查阅 $t-1$ 时刻的旧档案。随着物理系统的非线性或复杂度增加（例如引入加速度甚至加加速度），我们可能需要回溯至 $t-2$、$t-3$ 甚至初始时刻的历史观测。这种对历史记录的长程且深度的依赖，在数学建模中极为笨重，在计算机仿真与强化学习中则是不可接受的计算灾难。

为了打破对无穷历史的依赖，理论家们引入了**状态（State）**的概念。状态被严谨地定义为一个系统中**所有必要历史信息的充分统计量（Sufficient Statistic）**。对于上述匀速运动的汽车而言，如果我们同时掌控了位置与速度，并将这两个标量纵向拼接，组合成一个二维列向量，这就构成了该系统的完备状态 $\mathbf{s}_t$：

$$ \mathbf{s}_t = \begin{bmatrix} x_t \\ v_t \end{bmatrix} \in \mathbb{R}^2 $$
:eqlabel:`eq_state_vector`

当我们拥有了升维后的状态 $\mathbf{s}_t$ 之后，系统的演化规律便得到了一次极其优雅的降维简化——它可以通过与一个固定的状态转移矩阵（State Transition Matrix）相乘来完全确定：

$$ \mathbf{s}_{t+1} = \begin{bmatrix} x_{t+1} \\ v_{t+1} \end{bmatrix} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_t \\ v_t \end{bmatrix} = \mathbf{A} \mathbf{s}_t $$
:eqlabel:`eq_state_transition`

对比 :eqref:`eq_kinematics_obs` 和 :eqref:`eq_state_transition`，我们可以得出状态向量最为关键的物理性质：**只要给定当前的完整状态，系统未来的演化轨迹便完全独立于过去的历史。** 这种切断历史羁绊的“无记忆性”，在数学过程上拥有一个极其显赫的名字——**马尔可夫性质（Markov Property）** `[Markov, 1906]`。

## 马尔可夫性质与部分可观测环境
:label:`sec_markov_property`

让我们将上述理想且确定性的经典力学系统，推广到充满随机噪声和未知干扰的概率论范畴。在现代机器学习范式中，系统状态不仅受自身内在规律支配，还会受到外界智能体（Agent）注入的控制动作（Action） $A_t$ 的干预。

设大写字母 $S_t$ 为随机变量，表示系统在时刻 $t$ 的真实隐藏状态。如果过程满足马尔可夫性质，那么在给定当前状态 $S_t$ 和采取的动作 $A_t$ 的前提下，系统转移到下一时刻任何可能状态 $S_{t+1}$ 的条件概率分布，完全与 $t$ 时刻之前的任何更古老的状态或动作无关：

$$ \mathbb{P}(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0) = \mathbb{P}(S_{t+1} \mid S_t, A_t) $$
:eqlabel:`eq_markov_property`

等式 :eqref:`eq_markov_property` 堪称是整个强化学习大厦的基石。然而，遗憾的是，在真实世界中，我们几乎永远无法直接获取系统内部那台精密仪器的完整状态 $S_t$。智能体所能捕获的，仅仅是传感器通过某层滤镜投射而来的**观测（Observation）** $O_t$。

> 💡 柏拉图在《理想国》卷七中描绘了著名的“洞穴之喻”（Allegory of the Cave）：被锁链束缚在洞穴底部的囚徒无法回头，他们只能直视面前的墙壁，看着火光将背后的三维物体投射在墙上的二维阴影，并误以为这些扁平的影子就是世界的全部真相。
> 
> 在深度强化学习和世界模型的语境中，智能体的处境与洞穴中的囚徒如出一辙。高维且包含一切物理法则的环境真实状态 $S_t$ （真实世界的三维物体）是不可见的，智能体只能接收到降维且常常充斥着环境噪声的观测 $O_t$ （墙壁上的二维影子），例如一枚摄像头捕捉到的一帧静态像素矩阵。智能体乃至世界模型的核心任务，正是从这些影子的时间序列中，反向逆推出多维物体的真实三维结构与运动轨迹。

我们将这种只能获得局部、有损投影信息的框架称为**部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）**。在 POMDP 中，虽然底层状态转移满足马尔可夫性质，但我们所能记录的观测序列 $O_t$ 却是从真实的信念分布中采样的结果，受观测概率模型支配：

$$ O_t \sim \mathbb{P}(O \mid S_t) $$
:eqlabel:`eq_observation_model`

关键在于：由于 $O_t$ 只是状态的有损投影，它**自身绝不具备马尔可夫性质**。如果你仅仅截取一帧静态的赛车游戏画面，你根本无法仅凭这单一图像判断出赛车是在加速、刹车还是匀速前进。为了做出最优决策，智能体必须在自己的“大脑”内部，利用能够获取的所有碎片化线索，重新拼接出一个近似的“内部状态”。

## 贝叶斯滤波：从历史序列提取状态表示
:label:`sec_bayes_filtering`

既然孤立的单一观测 $O_t$ 无法提供充分的信息，最直观的破局之道是将有史以来的所有观测和动作全部累积起来，形成一条严密的**历史轨迹（History）** $H_t$：

$$ H_t = (O_1, A_1, O_2, A_2, \dots, A_{t-1}, O_t) $$
:eqlabel:`eq_history`

显然，$H_t$ 囊括了自时间源头起的所有信息，因此它天然具备马尔可夫性质。然而，随着时间步 $t$ 趋向于无限，历史轨迹的长度将不断增长，这导致状态空间呈指数级维度灾难。

数学上，最优的解决方案是不直接记录无穷的历史序列，而是利用概率论递归地维护一个对当前真实状态 $S_t$ 的概率置信度。我们将给定所有已知历史条件下的真实状态概率分布称为**信念状态（Belief State）**，记为向量或函数 $b_t(S) = \mathbb{P}(S_t = S \mid H_t)$。

通过严密的贝叶斯公式（Bayes' Rule），我们可以推导信念状态的递归更新过程（即贝叶斯滤波，Bayesian Filtering）：

1. **预测步（Prediction Step）：** 在接收到新的动作 $A_{t-1}$ 后，根据系统的状态转移模型 $\mathbb{P}(S_t \mid S_{t-1}, A_{t-1})$ 和上一步的信念状态 $b_{t-1}(S_{t-1})$，结合全概率公式预测当前的状态分布：
   $$ \mathbb{P}(S_t \mid H_{t-1}, A_{t-1}) = \sum_{S_{t-1}} \mathbb{P}(S_t \mid S_{t-1}, A_{t-1}) b_{t-1}(S_{t-1}) $$
   :eqlabel:`eq_bayes_predict`

2. **更新步（Update Step）：** 在接收到最新的观测 $O_t$ 后，利用观测模型 $\mathbb{P}(O_t \mid S_t)$ 更新并归一化状态信念：
   $$ b_t(S_t) = \mathbb{P}(S_t \mid H_t) = \frac{\mathbb{P}(O_t \mid S_t) \mathbb{P}(S_t \mid H_{t-1}, A_{t-1})}{\mathbb{P}(O_t \mid H_{t-1}, A_{t-1})} \propto \mathbb{P}(O_t \mid S_t) \mathbb{P}(S_t \mid H_{t-1}, A_{t-1}) $$
   :eqlabel:`eq_bayes_update`

将 :eqref:`eq_bayes_predict` 代入 :eqref:`eq_bayes_update`，我们得到了一个令人振奋的递归方程：

$$ b_t(S) \propto \mathbb{P}(O_t \mid S) \sum_{S_{t-1}} \mathbb{P}(S \mid S_{t-1}, A_{t-1}) b_{t-1}(S_{t-1}) $$
:eqlabel:`eq_bayes_filter_final`

仔细端详 :eqref:`eq_bayes_filter_final`，我们发现新的信念状态 $b_t$ 完全且仅仅由三项元素决定：上一时刻的信念状态 $b_{t-1}$、采取的动作 $A_{t-1}$ 以及最新到达的观测 $O_t$。

在深度学习尤其是世界模型的设计中，直接计算这套高维积分往往是不可行的。因此，现代架构倾向于使用深度神经网络（例如递归神经网络 RNN，或门控循环单元 GRU）将这一概率推断过程抽象并参数化为一个非线性函数 $f_\theta$。智能体的隐藏状态张量 $\mathbf{h}_t$ 便等价于这一信念状态：

$$ \mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, A_{t-1}, O_t) $$
:eqlabel:`eq_rnn_update`

等式 :eqref:`eq_rnn_update` 与等式 :eqref:`eq_bayes_filter_final` 在数学直觉上是一脉相承的。当前计算图中的状态向量 $\mathbf{h}_t$ 不仅吸纳了最新的环境观测，更关键的是它通过前驱节点 $\mathbf{h}_{t-1}$ 将无穷远古的历史信息进行了高效且定长的特征压缩。这正是后续章节中我们介绍循环状态空间模型（Recurrent State Space Model, RSSM）处理序列数据时的物理本源。

## 观测与状态的张量操作实战
:label:`sec_implementation_obs_state`

理论终须代码检验。现在，让我们从严谨的概率公式降落到具体的张量运算层面。

对于在非马尔可夫观测（诸如单帧 2D 图像）中挣扎的智能体，一种在深度强化学习早期（以 DeepMind 的 DQN 算法 `[Mnih et al., 2015]` 为代表）极其常用且极富直觉的“工程 Hack”，被称为**帧堆叠（Frame Stacking）**。它的理念非常朴素：既然一张画面无法推断速度，那我就将连续 $K$ 张画面像三明治一样叠加在一起，强行人工拼凑出一个近似的状态张量。虽然它不像 RNN 那样能在理论上涵盖无限历史，但由于它直接在通道维度（Channel dimension）保留了时间的显式演化，成为了理解历史与状态映射的最完美跳板。

假设环境传回的观测 $O_t$ 是一帧单通道灰度图，张量形状为 $(H, W)$。通过堆叠过去 $K$ 帧观测，我们将构造一个形状为 $(K, H, W)$ 的三维状态张量。

(**我们将创建一个简易的循环缓冲区，并演示如何使用 PyTorch 实现观测的帧堆叠，从而为卷积网络提供富含马尔可夫信息的近似状态。**)

```{.python .input}
#@tab pytorch
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

在上述代码执行过程中，哪怕我们所能触及的最底层的物理接口仅仅提供一张张扁平的平面观测，通过内存队列中的时间堆叠聚合，我们将单薄的数据升维成了包含时序信息的体素张量。当这束三维数据 $(K, H, W)$ 被喂给二维卷积神经网络（2D CNN）时，首层的卷积核将跨越这 $K$ 个通道同步计算，从而使神经网络有能力自主学习到诸如“像素在连续4帧内的位移矢量”这样高阶的运动学特征。用内存空间的少许冗余，换取了局部马尔可夫性质的复苏。

## 小结

- **观测（Observation）** 往往是局部、带噪且无法自证其完整性的信息碎片，它严重缺乏**马尔可夫性质**，单独依赖观测无法精确预测复杂动态系统的未来。
- **状态（State）** 则是系统中所有历史信息的充分统计表示。它切断了对冗长历史轨迹的依赖，只要给定当前状态，系统的未来演进就被完全决定。
- 面对真实世界的部分可观测限制（POMDP），无论是经典的贝叶斯滤波，还是深度学习中的 RNN 循环层与帧堆叠技巧，其本质数学动机皆高度一致：将庞大、无序的时序观测压缩为一个紧凑且符合马尔可夫演化规律的潜在状态向量。这是我们踏入世界模型（World Models）殿堂必须跨越的第一道理论门槛。

## 练习

1. 请回顾 :eqref:`eq_state_vector` 和 :eqref:`eq_state_transition` 所定义的直线运动。如果该汽车不再做匀速直线运动，而是做**匀加速直线运动**（加速度 $a$ 为一个未知的恒定常数），而我们的传感器依然只能观测到位置 $x_t$。
   - 为了使系统满足马尔可夫性质，你需要如何重新定义状态向量 $\mathbf{s}_t$？它的维度是多少？
   - 请写出针对你所定义的新状态向量，从 $\mathbf{s}_t$ 转移到 $\mathbf{s}_{t+1}$ 的准确的状态转移矩阵 $\mathbf{A}$（假设每一步的时间间隔仍为 $\Delta t$）。
   > *提示：结合高中物理知识，在匀加速运动中，$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2} a (\Delta t)^2$。请思考状态向量中是否需要容纳更多元素来涵盖所有的衍生物理量。*

2. 在本节末尾的 `ObservationBuffer` 实现中，假如我们在 $t=1$ 时刻提取状态，此时缓冲区的最前面有 3 个位置（即历史时刻 $t=0, -1, -2$）尚未被真实的观测画面填充。默认情况下它们是全零矩阵。
   - 在训练基于图像的强化学习智能体时，这种突兀的全零“空洞”特征会对卷积神经网络产生什么样的影响？
   - 能否构思一种无需全零初始化的替代工程策略，来处理初始交互阶段（时间步 $<K$）的状态构造问题？
   > *提示：可以考虑通过对起始帧的重复复制，或者使用自适应序列长度模型进行处理，思考哪种做法更为平滑。*

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
