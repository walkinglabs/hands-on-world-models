# 1.4 在想象中学习与自动驾驶应用

自动驾驶策略需要覆盖正常交通和低频危险场景，但真实道路上的探索受到安全、成本和数据分布限制。世界模型提供了一种补充手段：先从已有数据学习环境如何响应动作，再在模型中比较候选行为的可能后果。

<div align="center">
  <img src="/figures/01-why-world-models/source/04-imagine-driving/mile-fig5.png" alt="MILE 在进入环岛前先观察真实帧，随后只用潜在状态想象未来驾驶状态与动作。" width="86%">

_图 1.4-1：MILE 在进入环岛前先观察真实帧，随后只用潜在状态想象未来驾驶状态与动作。 出处：Anthony Hu et al.，[Model-Based Imitation Learning for Urban Driving](https://arxiv.org/abs/2210.07729)（2022），Figure 5。_

</div>

这种在模型中展开未来并更新策略的方法通常称为“在想象中学习”（Learning in Imagination）。本节先用运动学说明状态转移，再给出一种可微隐空间展开的教学实现，并讨论它与实际自动驾驶世界模型之间的边界。

## 1.4.1 学术脉络：从黑盒环境到世界模型的觉醒

无模型强化学习（Model-Free RL）不显式学习状态转移模型，而是从交互样本直接估计策略或价值。它不要求环境可微，但这不等于模型本身不能利用物理先验，也不等于所有无模型方法都只使用同一种蒙特卡洛估计。

2018年，Ha和Schmidhuber发表了《World Models》论文 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。他们用变分自编码器（VAE）将高维像素压缩为低维特征，再用混合密度循环网络（MDN-RNN）预测下一个潜变量。在 CarRacing 实验中，控制器利用 VAE 与 MDN-RNN 提供的特征在真实游戏环境中训练；论文另在 VizDoom 的 Take Cover 任务中演示了控制器完全在模型生成的潜在环境里训练，再迁回真实游戏。两项实验不能混为同一个“梦境赛车”结果。

随后，Dreamer、DreamerV2 与 DreamerV3 把策略学习移入潜在状态的想象轨迹中 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603); [[Hafner et al., 2021]](https://arxiv.org/abs/2010.02193); [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104)。这些方法在论文所测试的多项基准上提高了数据效率与任务表现，但结论应限定在相应任务和实验设置内。

<div align="center">
  <img src="/figures/01-why-world-models/source/04-imagine-driving/dreamerv2-fig1.png" alt="DreamerV2 的 Atari 汇总结果显示纯粹在世界模型内部学习行为也能达到人类基准线以上的中位表现。" width="86%">

_图 1.4-2：DreamerV2 的 Atari 汇总结果显示纯粹在世界模型内部学习行为也能达到人类基准线以上的中位表现。 出处：Danijar Hafner et al.，[Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)（2021），Figure 1。_

</div>

在自动驾驶领域，真实世界交互的成本与风险推动了两类相关探索。MILE 从离线驾驶数据中学习潜在动力学与驾驶策略，并能在学到的模型中想象未来 [[Hu et al., 2022]](https://arxiv.org/abs/2210.07729)。

<div align="center">
  <img src="/figures/01-why-world-models/source/04-imagine-driving/mile-fig2.png" alt="MILE 对同一路口生成两条八秒未来，展示潜在动力学对交通灯变化保留多模态分支。" width="86%">

_图 1.4-3：MILE 对同一路口生成两条八秒未来，展示潜在动力学对交通灯变化保留多模态分支。 出处：Anthony Hu et al.，[Model-Based Imitation Learning for Urban Driving](https://arxiv.org/abs/2210.07729)（2022），Figure 2。_

</div>

GAIA-1 则根据视频、文本和车辆动作生成未来驾驶场景 [[Anthony Hu et al., 2023]](https://arxiv.org/abs/2309.17080)。MILE 包含策略学习，GAIA-1 主要展示条件视频生成；不能把后者的生成结果直接等同于已经验证的轨迹规划器。

<div align="center">
  <img src="/figures/01-why-world-models/source/04-imagine-driving/gaia1-fig1.png" alt="GAIA-1 的条件生成样例展示视频、文本与车辆动作共同控制未来驾驶场景。" width="86%">

_图 1.4-4：GAIA-1 的条件生成样例展示视频、文本与车辆动作共同控制未来驾驶场景。 出处：Anthony Hu et al.，[GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)（2023），Figure 1。_

</div>

## 1.4.2 状态转移的物理学直觉与数学表达

世界模型要描述状态如何随动作变化。先从一维运动学看清这个接口，再把它推广到学习得到的隐空间。

### 一维空间下的基础动力学

考虑一辆在笔直公路上行驶的汽车，其状态可以通过位置 $p$ 和速度 $v$ 来唯一确定。我们将离散时间步 $t$ 下的状态定义为二维列向量：

$$
s_t = \begin{bmatrix} p_t \\ v_t \end{bmatrix}
$$

假设驾驶员在时间步 $t$ 施加的动作为加速度 $a_t$，且相邻时间步之间的时间间隔为 $\Delta t$。若在一个时间步内加速度保持不变，可以写出下一个状态：

$$
\begin{aligned}
p_{t+1} & = p_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2 \\
v_{t+1} & = v_t + a_t \Delta t
\end{aligned}
$$

由于这是一个线性系统，我们可以将其重写为矩阵乘法的标准形式。令 $A$ 为状态转移矩阵，$B$ 为控制输入矩阵，上述过程可以严谨地表示为：

$$
s_{t+1} = \underbrace{\begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}}_{A} s_t + \underbrace{\begin{bmatrix} \frac{1}{2} \Delta t^2 \\ \Delta t \end{bmatrix}}_{B} a_t
$$

在这些理想假设下，已知当前状态和动作序列，就可以反复应用公式进行推演。现实中的轮胎力、坡度和控制延迟会带来模型误差。

### 推广至高维隐变量空间

现代自动驾驶系统可以取得里程计、GNSS 和车辆控制量，但这些量并不能完整描述周围交通参与者、道路语义与遮挡区域。相机和激光雷达又带来高维观测 $x_t$，因此常把解析车辆模型与学习到的场景表示结合使用。

为了在如此复杂的观测空间中进行未来推演，世界模型引入了特征编码与隐式动力学机制。我们将物理上的转移函数推演为一个由神经网络参数化的非线性隐式状态转移模型 $f_\theta$。整个推演过程被解构为以下几个核心组件：

1. **表征模型 (Representation Model)**：将高维观测 $x_t$ 映射为低维的致密隐状态 $z_t \in \mathbb{R}^d$。
   $$ z_t = \text{Enc}_\theta(x_t) $$

2. **动力学模型 (Transition/Dynamics Model)**：在隐空间中，给定当前隐状态和智能体的动作，预测下一个时间步的隐状态先验分布。
   $$ \hat{z}_{t+1} = f_\theta(z_t, a_t) $$

3. **奖励预测器 (Reward Predictor)**：根据隐状态判断当前状态的好坏（例如是否偏离车道中心、是否发生碰撞等任务目标）。
   $$ \hat{r}_t = \text{Rew}_\theta(z_t) $$

这些组件构成了一个可微的学习型动力学模型。它近似训练数据中的转移规律，并不自动满足真实物理约束。

## 1.4.3 隐空间中的微分与策略优化

若动力学、奖励模型和动作采样路径可微，策略梯度可以沿想象轨迹反向传播。Dreamer 使用了这类路径导数，并结合价值估计训练行为模型；经典 _World Models_ 的控制器则使用 CMA-ES，并没有对 MDN-RNN 做端到端反向传播。因此，可微展开是一种重要做法，但不是所有世界模型的共同要求。

### 想象中的期望回报

假设我们有一个由参数 $\phi$ 控制的策略网络 $\pi_\phi(a_t | z_t)$。在时间步 $t$，智能体无需在真实世界执行任何物理操作，而是利用世界模型在脑海中“展开”一个长度为 $H$ 的想象轨迹 (Imagined Trajectory)：

$$
\tau_{\text{imagine}} = (z_t, a_t, \hat{r}_t, \hat{z}_{t+1}, a_{t+1}, \hat{r}_{t+1}, \dots, \hat{z}_{t+H}, \hat{r}_{t+H})
$$

在这个轨迹中，每一步的状态生成都遵循：
$$ a_{t+k} \sim \pi_\phi(\cdot | \hat{z}_{t+k}), \quad \hat{z}_{t+k+1} = f_\theta(\hat{z}_{t+k}, a_{t+k}) $$

我们定义该想象轨迹的累积折扣奖励 $R$ 为：
$$ R = \sum_{k=0}^{H} \gamma^k \text{Rew}_\theta(\hat{z}_{t+k}) $$

我们的核心目标是寻找最优的策略网络参数 $\phi$，使得期望回报最大化：
$$ \max_\phi \mathbb{E}_{\tau_{\text{imagine}} \sim \pi_\phi, f_\theta} [ R ] $$

### 链式法则与时间反向传播（BPTT）

由于动力学模型 $f_\theta$ 和奖励预测器 $\text{Rew}_\theta$ 都是平滑可导的神经网络，我们可以直接计算期望回报 $R$ 对策略参数 $\phi$ 的导数。为了清晰展示这一推演过程，我们考虑 $H=1$ 的极简情况，即仅仅向前推演一步。

此时的总回报为当前奖励与下一步奖励之和（假设折扣因子 $\gamma=1$）：
$$ R = \text{Rew}_\theta(z_t) + \text{Rew}_\theta(\hat{z}_{t+1}) $$

其中 $\hat{z}_{t+1} = f_\theta(z_t, a_t)$，且动作是通过某种确定性策略（或重参数化技巧下的随机策略）生成的：$a_t = \mu_\phi(z_t)$。

现在，我们严格应用多元微积分中的链式法则（Chain Rule）来计算梯度 $\nabla_\phi R$。由于第一项 $\text{Rew}_\theta(z_t)$ 与当前策略采取的动作无关（初始状态 $z_t$ 已确定且固化），其对 $\phi$ 的导数为零。我们主要关注第二项对策略参数的导数：

$$
\frac{\partial R}{\partial \phi} = \frac{\partial \text{Rew}_\theta(\hat{z}_{t+1})}{\partial \hat{z}_{t+1}} \cdot \frac{\partial \hat{z}_{t+1}}{\partial a_t} \cdot \frac{\partial a_t}{\partial \phi}
$$

<div align="center"><img src="/figures/01-why-world-models/latex/04-imagine-driving/bptt-jacobian-chain.png" alt="奖励梯度沿隐状态、动作和策略参数三段雅可比反向相乘" width="86%">

_图 1.4-5：前向依次产生动作、下一潜状态和奖励；反向传播按相反方向连乘三段局部雅可比，把奖励信号送回策略参数。_

</div>

三项分别表示：

1. $\frac{\partial a_t}{\partial \phi}$：策略网络对网络参数的梯度。
2. $\frac{\partial \hat{z}_{t+1}}{\partial a_t}$：**世界模型的雅可比矩阵 (Jacobian Matrix)**，表示模型预测对动作扰动的局部敏感度。
3. $\frac{\partial \text{Rew}_\theta(\hat{z}_{t+1})}{\partial \hat{z}_{t+1}}$：奖励函数对隐状态的梯度，它指明了在隐空间中朝哪个方向移动会获得更高的奖励，引导状态更新的方向。

当视界扩展到 $H$ 步时，梯度需要沿时间展开反向传播，即 BPTT。较早的动作会影响后续多个预测状态。较长视界也会带来梯度消失或爆炸，模型误差还会让策略利用不真实的预测；实践中常结合短视界、价值估计、正则化和真实数据校正。

## 1.4.4 代码实现：构建隐空间动力学与想象学习

下面实现一个可微动力学展开的最小教学例子。

代码的核心逻辑包含：

1. 定义世界模型的核心子模块：动力学转移网络（代替传统运动学矩阵）和奖励预测网络。
2. 定义策略网络，并在隐空间中向前自回归推演，展开多步的“梦境”轨迹。
3. 计算多步累积奖励，并直接调用自动微分（Autograd）引擎对策略网络进行BPTT反向优化。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义全局常量与超参数，设定张量维度
HIDDEN_DIM = 64
STATE_DIM = 16
ACTION_DIM = 2
SEQ_LEN = 10  # 想象的未来视界 H

class WorldModel(nn.Module):
    def __init__(self):
        super(WorldModel, self).__init__()
        # 简化版动力学模型: 隐式函数 f(z_t, a_t) -> 预测增量
        self.dynamics = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, STATE_DIM)
        )
        # 奖励预测器: Rew(z_t) -> r_t 标量
        self.reward_predictor = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def step(self, z, a):
        """在隐空间中前向推演一步（物理时间的流逝在计算图中的具象化）"""
        x = torch.cat([z, a], dim=-1)
        # 引入残差连接，使其具有偏微分方程离散积分的数值特性
        z_next = z + self.dynamics(x)
        reward = self.reward_predictor(z_next)
        return z_next, reward

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # 策略网络: pi(z_t) -> a_t
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),
            nn.Tanh() # 物理限制：规范动作范围在 [-1, 1] 之间
        )

    def forward(self, z):
        return self.net(z)

def imagine_and_optimize(world_model, policy, initial_state, optimizer):
    """
    在想象中展开虚拟驾驶轨迹并优化策略。
    """
    world_model.eval()
    # 不更新世界模型参数，但保留状态和动作路径上的计算图。
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)
    policy.train()
    optimizer.zero_grad()

    z_t = initial_state
    total_reward = 0.0
    discount = 0.99

    # [在循环中连续展开动力学，构建横跨时间步的前向计算图]
    for t in range(SEQ_LEN):
        # 根据当前隐状态生成控制动作
        a_t = policy(z_t)
        # 将动作输入世界模型，预测下个微小时间间隔后的隐状态及对应奖励
        z_t, r_t = world_model.step(z_t, a_t)
        # 将各时间步折现奖励累加至总期望回报中
        total_reward = total_reward + (discount ** t) * r_t

    # [由于整个演化过程完全由平滑激活的神经网络构成，我们可以直接对期望回报最大化求导]
    # 注意我们需要最大化回报，因此向优化器传递的损失（Loss）是取反的期望
    loss = -total_reward.mean()
    loss.backward()

    # 此时梯度已经严格依据多变量微积分法则，经由时间步反向贯穿到了策略参数层
    optimizer.step()
    return total_reward.mean().item()

# 这里只演示梯度路径。实际使用前应先训练并验证 world_model。
world_model = WorldModel()
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# 假设我们在环境的某个随机隐状态切片开始了本次想象推演
batch_size = 32
initial_state = torch.randn(batch_size, STATE_DIM)

# 执行若干次策略优化迭代
for i in range(5):
    avg_reward = imagine_and_optimize(world_model, policy, initial_state, optimizer)
    print(f"Iteration {i+1}, Imagined Average Reward: {avg_reward:.4f}")
```

## 1.4.5 小结

本节从离散运动学得到“状态与动作决定下一步”的接口，再把它推广到学习型隐空间动力学。对于可微模型，策略可以沿多步展开使用 BPTT；对于不可微或不采用路径导数的方法，也可以通过搜索、价值学习等方式利用模型。自动驾驶中的关键不是单纯增加想象步数，而是让训练数据、模型不确定性和规划范围保持一致。

## 1.4.6 练习

1. 在该公式的雅可比矩阵推导中，如果动作 $a_t$ 的微小扰动会导致隐状态偏向“碰撞”特征区域（假定碰撞区域的预测奖励值极低负数），请说明策略网络将依据怎样的数学符号法则反向调整其参数 $\phi$ 以避免碰撞？
   - 提示：根据链式法则的三项乘积，推导奖励网络输出最终随网络参数 $\phi$ 变化的偏导正负性机制。
2. 为什么在代码实现中，我们需要为动力学网络 `self.dynamics(x)` 强制添加一个残差连接结构（即 `z_next = z + dynamics`）？它在经典物理学和数值积分法上具有怎样的数学隐喻？
   - 提示：尝试考虑位置与速度增量在时间极限下的欧拉法离散微分方程（Euler Method）。
3. 如果世界模型在想象中进入训练数据未覆盖的分布外状态（Out-of-Distribution, OOD），BPTT 得到的策略梯度可能出现什么问题？
