# 4.6. MuZero：无模型搜索的世界模型
:label:`sec_muzero`

## 4.6.1. 历史脉络与学术背景
:label:`subsec_muzero_history`

在探讨 MuZero 之前，我们有必要简要回顾深度强化学习在规划（Planning）领域的发展脉络。早在 2016 年，AlphaGo `[Silver et al., 2016]` 的横空出世证明了深度神经网络结合蒙特卡洛树搜索（MCTS）可以攻克被视为人工智能领域难以跨越的围棋难题。随后，AlphaGo Zero `[Silver et al., 2017]` 和 AlphaZero `[Silver et al., 2018]` 进一步移除了对人类专家数据的依赖，通过完全的自我博弈（Self-play）在国际象棋、将棋和围棋等多种完美信息博弈中达到了超人类的水平。

然而，AlphaZero 架构存在一个极其苛刻的先决条件：它必须依赖于一个**完美的模拟器（Perfect Simulator）**。在搜索树的每次展开过程中，算法都需要向环境查询：“如果在当前状态 $s$ 执行动作 $a$，下一个状态 $s'$ 是什么？” 在围棋中，这一规则是确定的、已知的；但在现实世界的诸多问题（如机器人控制、自动驾驶甚至 Atari 视频游戏）中，环境的底层状态转移规律通常是未知的、极其复杂的，甚至是具有随机性的。

传统的无模型（Model-free）强化学习算法（例如 DQN 或 PPO）不需要完美的模拟器，它们通过试错直接学习策略（Policy）或价值函数（Value Function），但代价是彻底放弃了前向搜索与规划的能力，导致样本效率极低。另一方面，传统的基于模型（Model-based）的强化学习则尝试从观测数据中学习一个环境的动态模型，通常的做法是致力于**在像素级别重建未来的观测画面**。然而，在高维视觉空间中重建每一个细节不仅计算成本高昂，且往往将网络容量浪费在了与决策无关的背景细节上（例如视频游戏中随风飘动的云彩）。

正是为了打破这一瓶颈，MuZero `[Schrittwieser et al., 2020]` 提出了一种革命性的范式：**仅针对规划所需的核心要素建立世界模型**。它彻底放弃了对环境原始观测的重建，转而在隐空间（Latent Space）中学习一套动态系统，旨在极其精准地预测策略、价值和即时奖励。本节我们将通过严谨的数学语言，逐步拆解这一精妙的世界模型。

## 4.6.2. 隐空间动态系统的数学构建
:label:`subsec_muzero_formulation`

为了理解 MuZero 的数学本质，我们首先从最基础的离散时间动态系统出发。假设在时刻 $t$，环境向智能体提供了一个观测序列 $o_1, \dots, o_t$。我们的目标是基于当前的历史信息决定下一步的动作 $a_{t+1}$。

在完美模拟器中，真实的系统状态 $s \in \mathcal{S}$ 是已知的。但在未知的复杂环境中，我们只能构造一个数学上的隐状态（Hidden State）向量，记为 $s^0, s^1, s^2, \dots$。请注意，这里的上标不代表真实环境的时间，而是代表在脑海中**假想推演（Unrolling）**的步数。

为了驱动这个隐空间，MuZero 定义了三个核心的神经网络函数：

1. **表示函数 (Representation Function) $h$**：
   将其视为现实与隐空间的“桥梁”。它负责将历史观测映射为初始隐状态：
   $$ s^0 = h(o_1, \dots, o_t) $$
   :eqlabel:`eq_muzero_rep`

2. **动态函数 (Dynamics Function) $g$**：
   这是 MuZero 的“心脏”，即它学习到的世界模型。给定当前的隐状态 $s^{k-1}$ 和某个假想动作 $a^k$，动态函数预测出下一步的隐状态 $s^k$ 以及在该步转换中获得的即时奖励 $r^k$：
   $$ r^k, s^k = g(s^{k-1}, a^k) $$
   :eqlabel:`eq_muzero_dyn`

3. **预测函数 (Prediction Function) $f$**：
   在任何一个推演步 $k$，隐状态 $s^k$ 必须能够告诉我们当前局势的评估。预测函数将 $s^k$ 映射为在当前状态下的动作概率分布（策略）$p^k$ 以及标量形式的状态价值评估 $v^k$：
   $$ p^k, v^k = f(s^k) $$
   :eqlabel:`eq_muzero_pred`

通过结合这三个函数，智能体不再需要依赖真实的模拟器即可在神经网络的隐空间中连续展开 $K$ 步，从而进行深度的树搜索规划。

## 4.6.3. 核心机制：值等价原则 (Value Equivalence)
:label:`subsec_muzero_value_eq`

上述公式 :eqref:`eq_muzero_dyn` 中隐藏着一个极其尖锐的数学问题：**在没有任何重建观测损失（Observation Reconstruction Loss）的约束下，网络如何保证隐状态 $s^k$ 不会发生模式崩溃（Mode Collapse），即退化为全零或无意义的噪声矩阵？**

这就引出了 MuZero 全篇最为反直觉、也是最为核心的理论基石——**值等价（Value Equivalence）**。

> **类比理解：值等价（Value Equivalence）与“梦中下棋”**
>
> 传统的基于模型的强化学习试图在脑海中“渲染”出未来每一步的完整棋盘画面（重建观测）。然而，MuZero 的值等价原则就像是一位盲棋大师：他在脑海中并不需要勾勒出每一颗棋子的精确物理反光和材质。相反，他仅仅在神经元中维护着一个抽象的“局势张量”。只要这个张量能够准确推演出“走这步棋会导致我方优势（价值）下降，且被将军的概率（奖励）增加，对手必定会跳马（策略）”，那么这个“局势张量”就与真实的棋盘在数学上是**等价**的。我们不需要知道世界看起来是什么样，只需要知道世界将如何响应我们的目标。

在严格的数学层面上，值等价定义为：两个不同的状态表示空间 $\mathcal{S}_A$ 和 $\mathcal{S}_B$ 是值等价的，当且仅当对于任意相同的动作序列 $(a^1, \dots, a^K)$，它们产生的累积奖励预测和最终状态的价值预测是完全一致的。

因此，MuZero 通过反向传播算法（Backpropagation Through Time, BPTT），利用真实的未来累积奖励和 MCTS 搜索得到的改进策略，**直接作为梯度信号来雕刻（Regularize）动态函数 $g$ 和表示函数 $h$ 的参数**。隐状态 $s$ 没有任何预设的物理意义，它的存在仅仅是为了让预测函数 $f$ 能够准确输出 $p$ 和 $v$，让动态函数 $g$ 能够准确输出 $r$。

## 4.6.4. 隐空间中的蒙特卡洛树搜索 (MCTS)
:label:`subsec_muzero_mcts`

具备了在隐空间前向推演的能力后，我们需要使用 MCTS 来寻找最优策略。在传统的蒙特卡洛树搜索中，树的节点代表真实状态，边代表动作。而在 MuZero 中，树的根节点是当前真实历史的隐状态 $s^0 = h(o_1, \dots, o_t)$，树的所有内部节点都是由动态函数 $g$ 递推生成的假想隐状态 $s^k$。

搜索过程包含多次“模拟（Simulation）”。每次模拟自根节点开始，通过遵循上确界置信区间（Upper Confidence Bound for Trees, UCT）原则选择动作，直至遇到叶子节点。

对于节点 $s$，选择动作 $a$ 的准则为最大化置信区间目标：
$$ a = \mathop{\mathrm{argmax}}_a \left[ Q(s, a) + U(s, a) \right] $$
:eqlabel:`eq_muzero_uct_main`

其中，$Q(s, a)$ 代表在状态 $s$ 采取动作 $a$ 的动作价值估计（基于过往模拟积累的平均收益）。而 $U(s, a)$ 则是探索项，其严格定义为：
$$ U(s, a) = P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \left( c_1 + \log\left( \frac{\sum_b N(s, b) + c_2 + 1}{c_2} \right) \right) $$
:eqlabel:`eq_muzero_exploration`

这里：
- $P(s, a)$ 是网络预测函数 $f$ 给出的先验策略概率。
- $N(s, a)$ 表示在搜索树中该边被访问的次数。
- $\frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$ 构造了随着父节点访问次数增加而增大，但随着该节点自身访问次数增加而衰减的探索因子。
- 对数项 $\log(\cdot)$ 与常数 $c_1, c_2$ （通常 $c_1=1.25, c_2=19652$）用于控制随着搜索深度的极度增加，探索比例的长期缓慢增长。

当模拟到达一个尚未展开的叶子节点时，使用神经网络对该节点进行评估 $p, v = f(s)$，并以此更新整个模拟路径上各个边的 $Q(s, a)$ 和访问计数 $N(s, a)$。经过数千次模拟后，根节点的访问次数分布 $N(s^0, a)$ 将演化为一个极高置信度的改进策略（Improved Policy），记为 $\pi_t$。

## 4.6.5. 损失函数的严格推导与网络展开
:label:`subsec_muzero_loss`

MuZero 的端到端训练是在自我博弈收集到的经验轨迹上进行的。对于真实时间步 $t$，智能体将其收集的观测 $o_1, \dots, o_t$ 送入表示函数，得到 $s^0$。随后，根据历史轨迹中实际记录的真实动作 $a_{t+1}, \dots, a_{t+K}$，我们在隐空间中**展开 (Unroll)** $K$ 步。

在展开的第 $k$ 步（$1 \le k \le K$），网络输出预测元组 $(p_t^k, v_t^k, r_t^k)$。
为了对其进行监督，我们从真实的轨迹中提取目标（Targets）：
- 策略目标 $\pi_{t+k}$：来自时间步 $t+k$ 时 MCTS 的搜索结果（即访问次数分布）。
- 奖励目标 $u_{t+k}$：来自时间步 $t+k$ 时环境给出的真实即时奖励。
- 价值目标 $z_{t+k}$：可以是直到回合结束的真实折现回报，也可以是 $n$ 步自举（Bootstrapping）回报，即 $u_{t+1} + \gamma u_{t+2} + \dots + \gamma^{n-1} u_{t+n} + \gamma^n \nu_{t+n}$，其中 $\nu$ 是在 $t+n$ 时的 MCTS 价值估计。

由此，第 $k$ 步的损失函数由三项组成，外加 L2 正则化惩罚：
$$ L_k = l^p(\pi_{t+k}, p_t^k) + l^v(z_{t+k}, v_t^k) + l^r(u_{t+k}, r_t^k) $$
:eqlabel:`eq_muzero_step_loss`

整个 $K$ 步展开的总损失为时间的累加：
$$ L_t(\theta) = \sum_{k=0}^K \left( L_k \right) + c \| \theta \|^2 $$
:eqlabel:`eq_muzero_total_loss`

### 分类分布支持（Categorical Support）表示
在定义 $l^v$ 和 $l^r$ 时，MuZero 并没有直接使用标量的均方误差（MSE）。这是因为深度神经网络在拟合具有大方差的无界标量时极易发生梯度爆炸。

相反，MuZero 巧妙地将标量值转换为了离散支撑集（Support Set）上的分类概率分布。假设我们需要预测的值的合理区间为 $[-z_{\text{max}}, z_{\text{max}}]$，并且我们采用整数支撑集。
对于任意真实标量 $x \in [i, i+1]$（其中 $i, i+1$ 是相邻的整数支撑点），我们将其概率分配如下：
- 在支撑点 $i$ 上的概率分配为：$i+1 - x$
- 在支撑点 $i+1$ 上的概率分配为：$x - i$
- 其余支撑点概率均为 $0$。

此时，网络只需输出一个维度为 $2 z_{\text{max}} + 1$ 的 Logits 向量，然后通过 Softmax 归一化后计算与目标分类概率分布的交叉熵损失（Cross Entropy Loss）。这种设计极其稳定地约束了反向传播中的梯度幅值。

## 4.6.6. 模型架构与代码实现
:label:`subsec_muzero_code`

下面我们将其转化为严谨的代码。我们将定义整个网络架构中的各个核心组件。

(**首先，我们导入必要的张量运算框架并定义基础常量**)。在此为了简明起见，我们将定义适用于离散动作空间的基于多层感知机（MLP）的玩具网络。在真实场景下，表示网络 $h$ 通常是一个庞大的残差卷积网络（ResNet）。

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):
    """表示函数 h: observation -> hidden_state"""
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
    def forward(self, x):
        # x.shape: (batch_size, obs_dim)
        s = self.net(x)
        # 归一化隐状态，对保持动态网络的稳定至关重要
        # s.shape: (batch_size, hidden_dim)
        return F.normalize(s, p=2, dim=1)
```

(**接下来是核心的动态模型 $g$**)。它不仅负责状态的演进，还要给出奖励的预测。在此，我们将状态和动作拼接后进行前向传播。

```{.python .input}
#@tab pytorch
class DynamicsNetwork(nn.Module):
    """动态函数 g: (hidden_state, action) -> (next_hidden_state, reward_logits)"""
    def __init__(self, hidden_dim, num_actions, support_size):
        super().__init__()
        # 对离散动作进行独热编码后的维度为 num_actions
        self.fc = nn.Linear(hidden_dim + num_actions, 256)
        
        self.state_head = nn.Sequential(
            nn.Linear(256, hidden_dim)
        )
        # 奖励预测使用支持集的交叉熵表示
        self.reward_head = nn.Sequential(
            nn.Linear(256, support_size * 2 + 1)
        )
        
    def forward(self, hidden_state, action_one_hot):
        # 拼接张量，输入维度为 hidden_dim + num_actions
        x = torch.cat([hidden_state, action_one_hot], dim=1)
        x = F.relu(self.fc(x))
        
        next_s = self.state_head(x)
        reward_logits = self.reward_head(x)
        
        return F.normalize(next_s, p=2, dim=1), reward_logits
```

(**预测网络 $f$ 直接基于隐状态给出评估**)。我们需要同时预测策略（行动概率）和价值（胜率或预期回报）。

```{.python .input}
#@tab pytorch
class PredictionNetwork(nn.Module):
    """预测函数 f: hidden_state -> (policy_logits, value_logits)"""
    def __init__(self, hidden_dim, num_actions, support_size):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, support_size * 2 + 1)
        )
        
    def forward(self, hidden_state):
        policy_logits = self.policy_head(hidden_state)
        value_logits = self.value_head(hidden_state)
        return policy_logits, value_logits
```

(**将三者组合，即构成了完整的 MuZero 网络**)。在展开训练阶段，我们需要在时间步上进行前向的循环调用。

```{.python .input}
#@tab pytorch
class MuZeroNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_actions, support_size):
        super().__init__()
        self.num_actions = num_actions
        self.representation = RepresentationNetwork(obs_dim, hidden_dim)
        self.dynamics = DynamicsNetwork(hidden_dim, num_actions, support_size)
        self.prediction = PredictionNetwork(hidden_dim, num_actions, support_size)
        
    def initial_inference(self, obs):
        """对应推演步 k=0"""
        hidden_state = self.representation(obs)
        policy_logits, value_logits = self.prediction(hidden_state)
        return hidden_state, policy_logits, value_logits
        
    def recurrent_inference(self, hidden_state, action):
        """对应推演步 k > 0"""
        # 将动作转化为独热向量
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action_one_hot)
        policy_logits, value_logits = self.prediction(next_hidden_state)
        return next_hidden_state, reward_logits, policy_logits, value_logits
```

在严格的训练循环中，我们通过获取一个含有 $K$ 步真实动作和目标的批次，对 `recurrent_inference` 进行循环调用，积累 $L_k$ 并在反向传播时将梯度一路穿透整个时间序列流至表示网络。正是这种严密的数学耦合与反向传播机制，使得没有任何约束的隐状态被迫学习到了环境真实的动态法则。

## 4.6.7. 小结与练习
:label:`subsec_muzero_summary`

在本节中，我们拆解了 MuZero 如何摆脱对完美模拟器或原始观测像素重建的依赖。通过抽象的**隐空间**和严格的**值等价原则**，MuZero 使用表示、动态和预测三个网络，在神经网络深处闭环完成了蒙特卡洛树搜索的前向推演。这不仅大幅提升了规划模型的泛化能力，更为在未知复杂物理世界中的长程决策提供了一套数学上极其优雅的通用框架。

### 练习

1. 回顾值等价原则（Value Equivalence）。如果我们切断从 $t+k$ 步传递回来的价值和奖励梯度的反向传播路径，仅仅使用策略梯度来训练，隐状态的演化会发生什么问题？
   *提示：思考高维实数向量空间缺乏梯度约束时的自然发散现象。*
2. 在 MuZero 中的目标价值 $z_{t+k}$ 使用的是离散化支撑集的概率分布。请通过推导交叉熵公式，证明当真实价值为一个确定标量 $x=1.5$，且支撑点为 $i=1$ 和 $i=2$ 时，采用均方误差和交叉熵对应的损失函数在梯度方向上的差异。
   *提示：分别写出 $\frac{\partial}{\partial \theta} (x - f_\theta)^2$ 和 $\frac{\partial}{\partial \theta} (-\sum p_i \log \hat{p}_i)$。*
3. 在式 :eqref:`eq_muzero_exploration` 中，为什么分母使用的是 $1 + N(s,a)$ 而不是直接除以 $N(s,a)$？
   *提示：考虑到某个节点被首次访问时的极端情况，如果没有 $1$ 的存在，探索项的极限会如何变化？*

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
