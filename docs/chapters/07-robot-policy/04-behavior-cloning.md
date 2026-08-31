# 行为克隆基础

Pomerleau 的 ALVINN 用人类驾驶样本训练神经网络，把摄像头与激光测距输入映射为道路方向输出，并在改装车辆上进行道路测试 [[Pomerleau, 1989]](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html)。它是早期端到端学习驾驶的实例。所谓**行为克隆**（Behavior Cloning, BC），就是把专家观测—动作对当作监督学习数据；ALVINN 可以作为这一思路的历史案例，但单篇论文不足以承担整个模仿学习范式的起源判断。

<div align="center">

<img src="/figures/07-robot-policy/source/04-behavior-cloning/alvinn-fig3.png" alt="NAVLAB 是 ALVINN 道路测试所用真实车辆，连接监督模仿与实体部署。" width="86%">

_图 7.4-1：NAVLAB 是 ALVINN 道路测试所用真实车辆，连接监督模仿与实体部署。 出处：[ALVINN: An Autonomous Land Vehicle in a Neural Network，Dean A. Pomerleau，1989](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html)。_

</div>

<div align="center">

<img src="/figures/07-robot-policy/source/04-behavior-cloning/alvinn-fig1.png" alt="ALVINN 把道路图像与测距输入直接映射为离散转向输出。" width="86%">

_图 7.4-2：ALVINN 把道路图像与测距输入直接映射为离散转向输出。 出处：[ALVINN: An Autonomous Land Vehicle in a Neural Network，Dean A. Pomerleau，1989](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html)。_

</div>

先看三个驾驶样本：车辆偏左时专家向右修正，位于中心时保持方向，偏右时向左修正。行为克隆把这些“观测—动作”对当作监督学习数据，拟合从状态到动作的函数。它不需要奖励函数，但只能直接学习演示数据覆盖到的状态。

## 数学公式推导与建模

### 标量场景下的损失函数

先考虑标量场景。令状态 $s \in \mathbb{R}$ 表示车辆偏离车道中心线的距离，动作 $a \in \mathbb{R}$ 表示方向盘转角。

假设我们拥有了一位经验丰富的人类驾驶员，我们在其驾驶过程中以固定的采样频率记录了 $N$ 个时间步的数据，从而构建了专家数据集（Dataset）：

$$ \mathcal{D} = \{(s_1, a_1), (s_2, a_2), \dots, (s_N, a_N)\} $$

我们希望构建一个由参数 $\theta$ 所定义和驱动的策略（Policy）函数，记作 $\pi_\theta(s)$。在给定任意输入状态 $s$ 时，策略网络输出动作的预测值 $\hat{a} = \pi_\theta(s)$。

对于数据集中的单个样本 $(s_i,a_i)$，平方误差为：

$$ e_i = (\pi_\theta(s_i) - a_i)^2 $$

### 矢量化与高维空间的泛化扩展

在现实世界的机器人控制中，状态通常包含了海量且复杂的多元信息。例如，机械臂控制可能涉及多个独立关节的欧拉角、角速度，抑或是摄像头捕获的超高维图像像素矩阵。同样，控制指令往往也涉及对多个伺服电机的并发协同控制。

因此，我们需要将上述标量情况严格地扩展到向量和矩阵的线性空间中。令状态 $\mathbf{s} \in \mathbb{R}^d$，表示一个 $d$ 维的状态向量；令动作 $\mathbf{a} \in \mathbb{R}^k$，表示一个 $k$ 维的动作向量。

此时，我们的策略网络 $\pi_\theta$ 是一个将 $d$ 维向量映射到 $k$ 维向量的多变量非线性函数。我们通过经验风险最小化（Empirical Risk Minimization, ERM）来寻找最优的网络参数 $\theta^*$。在整个包含 $N$ 个独立样本的数据集 $\mathcal{D}$ 上，均方误差（Mean Squared Error, MSE）损失函数被定义为所有样本预测误差的平均值。利用向量空间的 $L_2$ 范数，我们可以严谨地将其公式化为：

$$ \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \|\pi_\theta(\mathbf{s}_i) - \mathbf{a}_i\|_2^2 $$

其中，$\|\cdot\|_2$ 严格表示欧几里得范数（Euclidean norm）。我们要优化的全局目标即为：

$$ \theta^* = \mathop{\mathrm{arg\,min}}_{\theta} \mathcal{L}(\theta) $$

这个优化目标就是标准的**监督学习**（Supervised Learning），因此可以直接使用小批量训练、SGD 或 Adam 等常见工具。需要注意，训练损失只衡量专家数据分布上的动作拟合，并不等价于闭环控制性能。

## 协变量偏移问题（Covariate Shift）

监督学习假设训练与测试输入来自相近分布，闭环控制却会让策略自己的动作改变下一步状态。若演示只包含车道中央附近的驾驶，策略一旦偏离，就可能进入没有纠偏样本的区域。这种训练状态分布与部署状态分布的差异称为**协变量偏移**（Covariate Shift）。

<div align="center">

<img src="/figures/07-robot-policy/source/04-behavior-cloning/dagger-fig2.png" alt="DAgger 在交互式收集数据后显著减少赛道失误，直接对应分布偏移的累积后果。" width="86%">

_图 7.4-3：DAgger 在交互式收集数据后显著减少赛道失误，直接对应分布偏移的累积后果。 出处：[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning，Stéphane Ross; Geoffrey Gordon; Drew Bagnell，2011](https://proceedings.mlr.press/v15/ross11a.html)。_

</div>

在严密的学术语言中，这意味着专家演示数据所隐含生成的状态边缘分布 $P_{\text{expert}}(\mathbf{s})$，与模型在实际环境闭环执行时自身诱导生成的状态分布 $P_{\pi_\theta}(\mathbf{s})$ 存在显著且不可忽视的统计学差异。

在训练阶段，模型是在独立同分布（i.i.d.）的强假设下，基于专家状态边缘分布 $P_{\text{expert}}(\mathbf{s})$ 进行误差最小化的：

$$ \mathbb{E}_{\mathbf{s} \sim P_{\text{expert}}} \left[ |\pi_\theta(\mathbf{s}) - \pi_{\text{expert}}(\mathbf{s})|_2^2 \right]
$$

<div align="center">

<img src="/figures/07-robot-policy/latex/04-behavior-cloning/covariate-shift-rollout.png" alt="单步动作偏差使闭环轨迹逐渐离开专家状态分布" width="86%">

_图 7.4-4：训练损失只约束专家访问的状态；一次动作偏差改变后续状态后，策略会进入自身诱导且未充分覆盖的分布。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

部署时，策略 $\pi_\theta$ 的动作会改变下一步状态。某一步的预测误差 $\epsilon$ 可能让 $\mathbf{s}_{t+1}$ 偏离专家轨迹，使模型继续在训练数据较少覆盖的状态上预测，误差因而可能沿闭环累积。

Ross 和 Bagnell 分析了监督式模仿学习中的分布偏移：若学习策略在专家状态分布上的单步错误率为 $\epsilon$，有限时域 $T$ 下的期望代价差在一般情形可达到 $\mathcal{O}(T^2\epsilon)$ [[Ross & Bagnell, 2010]](https://proceedings.mlr.press/v9/ross10a.html)。这里的平方项描述的是代价差界，而不是说每一种任务中的“动作错误数量”都必然按平方增长。

## 代码实现

下面我们将通过工程代码，具体且详尽地实现一个行为克隆的完整计算流图。为了保持示例的纯粹性与学术可重复性，我们将人为构造一个确定性的专家策略（例如一个理想状态下的线性反馈状态调节器），从而合成专家数据集，随后构建并训练一个前馈多层感知机（MLP）来严格克隆该专家的控制流形。

首先生成一个二维线性反馈专家的数据。

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 固定随机种子，便于复现实验输出
torch.manual_seed(42)

# 1. 假设环境状态向量的几何维度为 2 (例如: [相对位置, 相对速度])
# 假设理想的专家控制器是一个预先求解好的静态线性反馈增益矩阵
# 动作也是一个 1 维连续控制向量 (例如: [目标加速度])
state_dim = 2
action_dim = 1
expert_K = torch.tensor([[-0.5, -0.1]]) # 专家的反馈增益矩阵

def expert_policy(states):
    """
    人类专家测度策略：对状态进行线性投影，
    并在输出动作上注入微量的高斯白噪声，以模拟现实世界人类操作的不确定性。
    """
    actions = torch.matmul(states, expert_K.T)
    noise = torch.randn_like(actions) * 0.01
    return actions + noise

# 2. 收集与采样专家状态-动作对
num_samples = 1000
# 随机采样状态，作为专家在其轨迹中经历过的边缘分布数据
expert_states = torch.randn((num_samples, state_dim))
expert_actions = expert_policy(expert_states)

print(f"专家状态张量维度: {expert_states.shape}")
print(f"专家动作张量维度: {expert_actions.shape}")
```

(**接着，我们定义用于模仿专家行为的深度神经网络架构，并构建用于批量梯度优化的数据加载器。**)

```python
# 3. 定义策略网络架构 (前馈多层感知机)
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehavioralCloningPolicy, self).__init__()
        # 使用多层全连接网络拟合复杂的非线性策略流形
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # 执行前向传播运算
        return self.net(x)

# 实例化策略网络
policy_net = BehavioralCloningPolicy(state_dim, action_dim)

# 4. 构建用于随机梯度下降 (SGD) 的微批次数据加载器
batch_size = 64
dataset = TensorDataset(expert_states, expert_actions)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 定义目标损失函数与自适应梯度优化器 (Adam)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
```

(**最后，我们执行标准的监督学习训练微循环，通过反向传播算法最小化动作重构的经验风险。**)

```python
num_epochs = 50
loss_history = []

# 启用网络训练模式
policy_net.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_states, batch_actions in dataloader:
        # 步骤 1: 策略网络前向传播计算预测动作张量
        pred_actions = policy_net(batch_states)

        # 步骤 2: 计算预测张量与专家动作张量之间的 L2 范数均方误差
        loss = loss_fn(pred_actions, batch_actions)

        # 步骤 3: 清空上一步残留的梯度，执行反向传播计算新梯度，并更新网络权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_states.size(0)

    # 记录并计算当前 epoch 在全体数据集上的平均损失
    epoch_loss /= num_samples
    loss_history.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}/{num_epochs}, 经验风险 MSE Loss: {epoch_loss:.6f}")
```

这段代码完成的是开环动作回归。即使训练集 MSE 很低，也仍需把策略放回环境，检查偏离专家轨迹后能否恢复；这正是离线拟合与闭环评测的边界。

## 小结

- **行为克隆（Behavior Cloning, BC）**把专家观测—动作对转化为监督学习问题。
- 开环动作误差与闭环任务成功率是两种不同指标；协变量偏移会把早期小误差带到训练数据未覆盖的状态。

$$
$$
