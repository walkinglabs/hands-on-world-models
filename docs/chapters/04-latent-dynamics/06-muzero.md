# 4.6 MuZero：不重建观测的隐空间搜索

## 4.6.1 历史脉络与学术背景

AlphaGo 将策略/价值网络与蒙特卡洛树搜索（MCTS）结合 [[Silver et al., 2016]](https://doi.org/10.1038/nature16961)。AlphaGo Zero [[Silver et al., 2017]](https://doi.org/10.1038/nature24270) 和 AlphaZero [[Silver et al., 2018]](https://doi.org/10.1126/science.aar6404) 进一步减少了对人类棋谱的依赖，并在已知规则的完美信息棋类中通过自我博弈训练。

AlphaZero 的搜索需要已知的状态转移与规则：给定棋盘状态和动作，可以精确得到下一状态。Atari 或真实控制任务通常只提供交互接口，智能体并没有可在搜索树中任意调用的规则模型。

标准 DQN 或 PPO 不在决策时执行前向树搜索；基于模型的方法则学习可用于预测或规划的动力学。许多视觉模型用观测重建训练表示，但重建所有像素并不是模型式强化学习的必要条件，也可能把容量分配给与决策无关的细节。

MuZero 学习用于搜索的隐空间模型，而不要求重建原始观测 [[Schrittwieser et al., 2020]](https://arxiv.org/abs/1911.08265)。其动力学网络预测下一隐状态与奖励，预测网络输出策略和价值；这些量由搜索结果与实际轨迹构造训练目标。因而，更准确的说法是“学习规划所需的量”，而不是声称每个预测都达到没有误差的精确程度。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/06-muzero/muzero-fig2.png" alt="棋类棋盘与 Atari 画面上方对应四组学习曲线，展示 MuZero 同一算法实际覆盖棋类和视觉游戏。" width="86%">

_图 4.6-1：棋类棋盘与 Atari 画面上方对应四组学习曲线，展示 MuZero 同一算法实际覆盖棋类和视觉游戏。 出处：Julian Schrittwieser et al.，[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)（2020），Figure 2。_

</div>

## 4.6.2 隐空间动态系统的数学构建

在时刻 $t$，表示网络根据当前观测或观测历史构造根状态，搜索再据此选择下一动作。

MuZero 构造隐状态 $s^0,s^1,s^2,\dots$。上标 $k$ 表示从搜索根节点开始的网络展开步数，不是环境的绝对时间索引。

为了驱动这个隐空间，MuZero 定义了三个核心的神经网络函数：

1. **表示函数 (Representation Function) $h$**：
   把当前观测表示映射为搜索根节点的隐状态：
   $$ s^0 = h(o_1, \dots, o_t) $$

2. **动态函数 (Dynamics Function) $g$**：
   给定隐状态 $s^{k-1}$ 和假想动作 $a^k$，预测下一隐状态 $s^k$ 与奖励 $r^k$：
   $$ r^k, s^k = g(s^{k-1}, a^k) $$

3. **预测函数 (Prediction Function) $f$**：
   在任何一个推演步 $k$，隐状态 $s^k$ 必须能够告诉我们当前局势的评估。预测函数将 $s^k$ 映射为在当前状态下的动作概率分布（策略）$p^k$ 以及标量形式的状态价值评估 $v^k$：
   $$ p^k, v^k = f(s^k) $$

三个函数使搜索可以在学到的隐空间中展开，而无需在每个节点调用真实环境规则。

## 4.6.3 决策充分的隐状态

MuZero 没有观测重建目标，那么隐状态为何仍能用于规划？关键在于它在每个展开步都接受奖励、价值和策略目标的监督。若不同历史需要不同预测，常数状态就无法同时拟合这些目标。

这种设计可用“决策充分”或**值等价（Value Equivalence）**的直觉理解：表示不必还原观察外观，只需保留搜索目标需要区分的信息。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/06-muzero/vpn-fig5.png" alt="VPN 从同一迷宫状态展开不同长度的潜在计划，展示不重建像素、只预测规划量的直接前史。" width="86%">

_图 4.6-2：VPN 从同一迷宫状态展开不同长度的潜在计划，展示不重建像素、只预测规划量的直接前史。 出处：Junhyuk Oh；Satinder Singh；Honglak Lee，[Value Prediction Network](https://arxiv.org/abs/1707.03497)（2017），Figure 5。_

</div>

> **类比理解：值等价（Value Equivalence）与“梦中下棋”**
>
> 传统的基于模型的强化学习试图在脑海中“渲染”出未来每一步的完整棋盘画面（重建观测）。然而，MuZero 的值等价原则就像是一位盲棋大师：他在脑海中并不需要勾勒出每一颗棋子的精确物理反光和材质。相反，他仅仅在神经元中维护着一个抽象的“局势张量”。只要这个张量能够准确推演出“走这步棋会导致我方优势（价值）下降，且被将军的概率（奖励）增加，对手必定会跳马（策略）”，那么这个“局势张量”就与真实的棋盘在数学上是**等价**的。我们不需要知道世界看起来是什么样，只需要知道世界将如何响应我们的目标。

严格的值等价理论需要指定策略集合与价值函数集合。这里采用更窄的操作性解释：在训练分布和有限展开深度内，若两个表示对搜索使用的奖励、策略与价值预测不可区分，那么没有必要要求它们在像素上相同。

训练时，真实奖励、bootstrap 价值目标和 MCTS 访问分布沿展开网络反向传播到 $g$ 与 $h$。隐状态没有预设的物理语义；它被优化为支持 $r,p,v$ 预测。有限目标仍可能遗漏任务外信息，因此不能据此声称学到了完整环境动力学。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/06-muzero/predictron-fig1.png" alt="Predictron 把多步内部模型回报与可学习的 λ 混合画成一张展开图，为 MuZero 的值等价思想提供历史机制。" width="86%">

_图 4.6-3：Predictron 把多步内部模型回报与可学习的 λ 混合画成一张展开图，为 MuZero 的值等价思想提供历史机制。 出处：David Silver；Hado van Hasselt；Matteo Hessel；Tom Schaul；Arthur Guez；Timothy Harley；Gabriel Dulac-Arnold；David Reichert；Neil Rabinowitz；André Barreto；Thomas Degris，[The Predictron: End-To-End Learning and Planning](https://arxiv.org/abs/1612.08810)（2017），Figure 1。_

</div>

## 4.6.4 隐空间中的蒙特卡洛树搜索（MCTS）

具备了在隐空间前向推演的能力后，我们需要使用 MCTS 来寻找最优策略。在传统的蒙特卡洛树搜索中，树的节点代表真实状态，边代表动作。而在 MuZero 中，树的根节点是当前真实历史的隐状态 $s^0 = h(o_1, \dots, o_t)$，树的所有内部节点都是由动态函数 $g$ 递推生成的假想隐状态 $s^k$。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/06-muzero/alphazero-table2.png" alt="AlphaZero 的十二类棋局开局与自博弈频率显示已知规则搜索如何在棋盘状态上形成策略前身。" width="86%">

_图 4.6-4：AlphaZero 的十二类棋局开局与自博弈频率显示已知规则搜索如何在棋盘状态上形成策略前身。 出处：David Silver et al.，[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)（2017），Table 2。_

</div>

搜索重复进行多次模拟。MuZero 使用带策略先验的 PUCT 风格准则选择动作，直到到达尚未展开的叶节点。

对于节点 $s$，选择动作 $a$ 的准则为最大化置信区间目标：
$$ a = \mathop{\mathrm{argmax}}_a \left[ Q(s, a) + U(s, a) \right] $$

其中 $Q(s,a)$ 是边的平均价值估计，$U(s,a)$ 是由策略先验与访问次数构成的探索项。论文使用的形式为：
$$ U(s, a) = P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \left( c_1 + \log\left( \frac{\sum_b N(s, b) + c_2 + 1}{c_2} \right) \right) $$

<div align="center"><img src="/figures/04-latent-dynamics/latex/06-muzero/puct-visit-pressure.png" alt="PUCT 探索项由策略先验、父节点访问量、当前边访问量和对数调度共同构成" width="86%">

_图 4.6-5：策略先验给出初始偏好；在其他量固定时，父节点搜索越充分，探索压力越大，而某条边自身访问越多，其探索奖励越快衰减。_

</div>

这里：

- $P(s, a)$ 是网络预测函数 $f$ 给出的先验策略概率。
- $N(s, a)$ 表示在搜索树中该边被访问的次数。
- $\frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$ 构造了随着父节点访问次数增加而增大，但随着该节点自身访问次数增加而衰减的探索因子。
- 对数项与常数 $c_1,c_2$ 控制探索系数随父节点访问次数增长的速度；它不是直接按树深度增长。

到达叶节点后，用 $p,v=f(s)$ 评估并扩展，再沿路径回传价值与更新访问计数。根节点访问次数经过归一化形成搜索策略 $\pi_t$。模拟次数取决于任务与计算预算；MuZero 的不同实验设置并不统一为“数千次”。

## 4.6.5 损失函数与网络展开

MuZero 的端到端训练是在自我博弈收集到的经验轨迹上进行的。对于真实时间步 $t$，智能体将其收集的观测 $o_1, \dots, o_t$ 送入表示函数，得到 $s^0$。随后，根据历史轨迹中实际记录的真实动作 $a_{t+1}, \dots, a_{t+K}$，我们在隐空间中**展开 (Unroll)** $K$ 步。

在展开的第 $k$ 步（$1 \le k \le K$），网络输出预测元组 $(p_t^k, v_t^k, r_t^k)$。
为了对其进行监督，我们从真实的轨迹中提取目标（Targets）：

- 策略目标 $\pi_{t+k}$：来自时间步 $t+k$ 时 MCTS 的搜索结果（即访问次数分布）。
- 奖励目标 $u_{t+k}$：来自时间步 $t+k$ 时环境给出的真实即时奖励。
- 价值目标 $z_{t+k}$：可以是直到回合结束的真实折现回报，也可以是 $n$ 步自举（Bootstrapping）回报，即 $u_{t+1} + \gamma u_{t+2} + \dots + \gamma^{n-1} u_{t+n} + \gamma^n \nu_{t+n}$，其中 $\nu$ 是在 $t+n$ 时的 MCTS 价值估计。

由此，第 $k$ 步的损失函数由三项组成，外加 L2 正则化惩罚：
$$ L_k = l^p(\pi_{t+k}, p_t^k) + l^v(z_{t+k}, v_t^k) + l^r(u_{t+k}, r_t^k) $$

整个 $K$ 步展开的总损失为时间的累加：
$$ L_t(\theta) = \sum_{k=0}^K \left( L_k \right) + c \| \theta \|^2 $$

### 分类分布支持（Categorical Support）表示

在定义 $l^v$ 和 $l^r$ 时，MuZero 把标量经过非线性缩放后投影到有限离散支撑，而不是直接用未缩放标量 MSE。这样可以减小不同目标尺度带来的优化差异。

为便于说明，先忽略论文中的可逆标量缩放，直接把 $x$ 投影到整数支撑 $[-z_{\max},z_{\max}]$。完整 MuZero 会先应用带平方根项的标量变换，并在解码时使用逆变换。
对于任意真实标量 $x \in [i, i+1]$（其中 $i, i+1$ 是相邻的整数支撑点），我们将其概率分配如下：

- 在支撑点 $i$ 上的概率分配为：$i+1 - x$
- 在支撑点 $i+1$ 上的概率分配为：$x - i$
- 其余支撑点概率均为 $0$。

网络输出 $2z_{\max}+1$ 个 logits，并用交叉熵拟合目标分布。对输出 logits 的梯度分量有界，但深层网络整体仍可能出现数值或梯度问题。

## 4.6.6 模型架构与代码实现

下面给出只覆盖三个网络接口的 MLP 教学版。它没有实现 MCTS、标量支撑变换、损失缩放或论文中的卷积/残差架构。

先定义适用于离散动作空间的表示网络。

```python
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

动态模型 $g$ 接收隐状态与独热动作，输出下一隐状态和奖励 logits。

```python
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

预测网络 $f$ 输出策略 logits 与价值 logits。

```python
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

最后组合三个接口。完整训练会循环调用 `recurrent_inference` 并累积每个展开步的损失。

```python
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

训练批次包含 $K$ 步动作与目标。对 `recurrent_inference` 展开并累积损失后，梯度会到达表示、动态与预测网络，使隐状态保留有限展开内预测奖励、价值和策略所需的信息。

## 4.6.7 小结

MuZero 用表示、动态和预测三个网络支持隐空间 MCTS，不要求重建原始观测。它学习的是由奖励、价值与搜索策略目标定义的决策相关模型。这样避免了像素重建开销，但搜索成本、模型偏差与训练目标覆盖范围仍然限制其适用性。
