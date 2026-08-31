# 4.7 RSSM 的从零开始实现

前几章介绍了如何把图像压缩为低维表示，再用循环网络描述这些表示随时间的变化。不过，未来往往不止一种：同一个路口既可能左转，也可能直行。若用均方误差训练一个只输出单点预测的模型，它可能把几种未来平均在一起；即使单步误差很小，反复滚动后也可能逐渐偏离真实轨迹。

为同时保留历史摘要并表示当前的不确定性，Hafner 等人在 PlaNet [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551) 中提出了**循环状态空间模型**（Recurrent State Space Model，RSSM）。它把循环网络的确定性记忆与状态空间模型的随机潜变量放在同一递推结构中；在训练方法上，又借用了变分推断的思想。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/07-rssm-scratch/opening-rssm-collect.gif" alt="DMLab Collect 的真实帧与 RSSM 预测连续对照，展示从零实现最终要复现的“看过上下文后继续想象”行为。" width="86%">

_图 4.7-1：DMLab Collect 的真实帧与 RSSM 预测连续对照，展示从零实现最终要复现的“看过上下文后继续想象”行为。 出处：Danijar Hafner；Timothy Lillicrap；Jimmy Ba；Mohammad Norouzi，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Official multi-step prediction GIF: DMLab Collect。_

</div>

本节从一个标量动力学例子出发，再把它翻译成张量形状和 PyTorch 代码。实现刻意省略图像编码器、解码器与奖励头，重点只放在 RSSM 的状态递推、先验和后验。

## 历史背景与动力学建模的困境

确定性的 LSTM 或 GRU 可以很好地保存历史，但若训练目标要求它在多种合理未来中只给出一幅均方误差最小的图像，输出就容易接近这些未来的像素平均。这里的问题不在“确定性网络必然模糊”，而在单点输出、损失函数与多模态数据之间不匹配。

变分循环神经网络 [[Chung et al., 2015]](https://arxiv.org/abs/1506.02216) 在每个时间步引入随机变量，使模型能够表示不同的未来分支。RSSM 进一步显式分开两条路径：确定性状态负责递推历史摘要，随机状态负责表达当前潜变量及其不确定性。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/07-rssm-scratch/vrnn-fig2.png" alt="VRNN 将潜变量均值变化、逐时刻 KL 与输入波形对齐，说明随机状态会在序列事件发生处吸收新信息。" width="86%">

_图 4.7-2：VRNN 将潜变量均值变化、逐时刻 KL 与输入波形对齐，说明随机状态会在序列事件发生处吸收新信息。 出处：Junyoung Chung；Kyle Kastner；Laurent Dinh；Kratarth Goel；Aaron Courville；Yoshua Bengio，[A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/abs/1506.02216)（2015），Figure 2。_

</div>

PlaNet 的 RSSM 在每个时间步同时维护确定性的循环状态与随机状态 [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551)：前者汇总历史，后者表达当前潜变量及其不确定性。DreamerV2 延续这一分解并把随机状态改为离散变量，DreamerV3 继续使用离散 RSSM [[Hafner et al., 2021]](https://arxiv.org/abs/2010.02193); [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104)。

## 从抛物运动到分离状态空间

让我们暂时抛开高维张量和神经网络，回到高中物理中最经典的抛物运动。

假设你在一个大风天抛出一个网球。网球在 $t$ 时刻的位置不仅取决于 $t-1$ 时刻的位置和速度，还取决于你施加的力量（动作），以及当时阵风的扰动。

在一个简化的、一维的离散时间系统中，网球的速度 $v_t$ 可以近似写为：
$$v_t = v_{t-1} + a \cdot \Delta t + \epsilon_t$$

其中，$a$ 是加速度，$\epsilon_t$ 表示未建模的阵风等扰动。等式右边可以分成两部分：

1. **确定性部分**（$v_{t-1} + a \cdot \Delta t$）：根据已有状态和输入计算出的趋势。
2. **随机部分**（$\epsilon_t$）：当前模型没有解释的扰动或不确定性。

推广到非线性系统时，令 $h_t$ 表示对过去信息的有限维摘要，$z_t$ 表示当前随机潜变量。若上一时刻执行动作 $a_{t-1}$，RSSM 的一步计算可以拆成下面三个组件。

### 确定性状态更新方程

确定性状态 $h_t$ 融合上一时刻的记忆 $h_{t-1}$、随机状态 $z_{t-1}$ 和动作 $a_{t-1}$：
$$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$$

这里，$f_\theta$ 通常由 GRU 实现。按照这一计算顺序，$h_t$ 先由上一状态和上一动作得到；当前观测随后进入后验网络，修正对 $z_t$ 的判断。

### 先验动力学（Prior Dynamics）

在没有看到时刻 $t$ 真实发生的画面之前，我们需要基于我们对世界的理解（即 $h_t$），去“想象”当前时刻可能发生什么。这就是**先验分布**。为了数学上的可计算性，我们通常假设它服从多变量高斯分布：
$$p_\theta(z_t \mid h_t) = \mathcal{N}(\mu_\theta(h_t), \Sigma_\theta(h_t))$$

先验动力学网络通过多层感知机（MLP）输出均值 $\mu_\theta$ 和协方差矩阵（通常被限制为对角阵）$\Sigma_\theta$。

### 后验推断（Posterior Inference）

看到时刻 $t$ 的观测 $x_t$ 后，推断网络结合 $h_t$ 与观测特征，得到对当前随机状态的近似后验：
$$q_\phi(z_t \mid h_t, x_t) = \mathcal{N}(\mu_\phi(h_t, x_t), \Sigma_\phi(h_t, x_t))$$

<div align="center">
  <img src="/figures/04-latent-dynamics/source/07-rssm-scratch/dkf-fig2.png" alt="Deep Kalman Filter 的重建、采样与未见数字推断展示深度状态空间模型如何用后验校正再向前生成。" width="86%">

_图 4.7-3：Deep Kalman Filter 的重建、采样与未见数字推断展示深度状态空间模型如何用后验校正再向前生成。 出处：Rahul G. Krishnan；Uri Shalit；David Sontag，[Deep Kalman Filters](https://arxiv.org/abs/1511.05121)（2015），Figure 2。_

</div>

训练时，后验为重构任务提供带有当前观测信息的状态；KL 项则约束先验和后验不要相差过远。不同 RSSM 变体会用不同的梯度截断和 KL 权重，因此后验并不是一个在所有梯度路径上都固定不动的“标签”。

> 💡 **复杂机制的直觉映射：**
> 想象你在黑暗中摸索前行（先验动力学预测可能的位置 $h_t \to z_t$），突然你打开手电筒看清了周围的陈设（观测 $x_t$），此时你在脑海中立刻修正了对自己确切位置的判断（后验推断 $h_t, x_t \to z_t$）。在训练世界模型时，我们要让闭着眼睛摸索的预测，尽可能贴近睁开眼睛看到的真实结果（这正是最小化 KL 散度的物理意义）。

## 张量维度推演

实现前先列出张量形状。设批大小为 $B$，序列长度为 $T$。

1. **确定性状态维度（Deterministic State Dimension）**：$D_h$。对于一般的复杂任务，通常取 200 到 1024。
2. **随机状态维度（Stochastic State Dimension）**：$D_z$。由于它是高斯分布采样的结果，其维度表示潜在因子的数量，通常取 32 到 256。
3. **动作维度（Action Dimension）**：$D_a$。

在单步推进中：

- 提取的历史特征：$h_{t-1} \in \mathbb{R}^{B \times D_h}$
- 历史随机特征：$z_{t-1} \in \mathbb{R}^{B \times D_z}$
- 当前动作：$a_{t-1} \in \mathbb{R}^{B \times D_a}$
- GRU 的输入是将 $z_{t-1}$ 和 $a_{t-1}$ 拼接后经过线性变换得到的向量：$x_{gru} \in \mathbb{R}^{B \times D_{hidden}}$

## 构建 RSSM 核心单元（RSSMCell）

现在，让我们利用 PyTorch 从零构建 `RSSMCell`。这个类负责在单个时间步执行动力学的前向传播和后验推断。

```python
import torch
from torch import nn
from torch.nn import functional as F

class RSSMCell(nn.Module):
    """循环状态空间模型（RSSM）的单步执行单元。"""
    def __init__(self, action_dim, deter_dim=200, stoch_dim=30, hidden_dim=200):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # 1. 确定性状态更新相关网络
        # 作用：处理 (z_{t-1}, a_{t-1}) 作为 GRU 的输入
        self.fc_state_action = nn.Linear(stoch_dim + action_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        # 2. 先验动力学网络 (Prior Dynamics)
        # 作用：基于 h_t 预测先验的 z_t 的均值和方差
        self.fc_prior_hidden = nn.Linear(deter_dim, hidden_dim)
        self.fc_prior_stats = nn.Linear(hidden_dim, 2 * stoch_dim)

        # 3. 后验推断网络 (Posterior Inference)
        # 作用：基于 h_t 和 观测特征 x_t 提取后验的 z_t 的均值和方差
        self.fc_posterior_hidden = nn.Linear(deter_dim + hidden_dim, hidden_dim)
        self.fc_posterior_stats = nn.Linear(hidden_dim, 2 * stoch_dim)
```

下面用两个辅助函数处理高斯分布参数和重参数化采样。这里网络输出的是未经约束的尺度参数，经 Softplus 转为正的标准差，再加上最小值以改善数值稳定性；它并不是“对数方差”。

```python
def extract_stats(stats_tensor, min_std=0.1):
    """
    将网络输出分解为均值和标准差。
    为了防止方差收敛于 0 导致数值崩溃，采用 softplus 并在底层增加安全阈值 min_std。
    """
    mean, unnormalized_std = stats_tensor.chunk(2, dim=-1)
    std = F.softplus(unnormalized_std) + min_std
    return mean, std

def sample_gaussian(mean, std):
    """
    重参数化采样过程。
    严格对应于 z_t = \mu + \sigma \odot \epsilon，其中 \epsilon \sim \mathcal{N}(0, I)
    """
    noise = torch.randn_like(mean)
    return mean + std * noise
```

### 实现单步前向传播（Step）

在前向传播时，RSSM 会区分两种情况：

1. **环境交互/想象阶段（Prior/Imagination）**：没有真实的观测 $x_t$，模型完全依赖先验网络向未来滚动。
2. **状态推断阶段（Observation/Inference）**：有真实的观测 $x_t$，模型利用后验网络校正状态，这主要用于训练阶段和真实环境中的信念状态（Belief State）更新。

我们实现一个通用的 `forward_step` 函数，它可以根据是否提供观测信息，无缝切换先验推断与后验推断。

```python
    # 将以下方法补充至 RSSMCell 类内部
    def forward_step(self, prev_deter, prev_stoch, prev_action, obs_embed=None):
        """
        执行单个时间步的状态推导。
        如果 obs_embed 为 None，则纯粹执行先验想象；否则计算后验分布。
        """
        # 计算确定性状态更新
        za_cat = torch.cat([prev_stoch, prev_action], dim=-1)
        gru_input = F.elu(self.fc_state_action(za_cat))

        # GRU 内部状态更新
        # h_t = GRU(input_t, h_{t-1})
        deter_state = self.cell(gru_input, prev_deter)

        # 计算先验分布
        prior_hidden = F.elu(self.fc_prior_hidden(deter_state))
        prior_stats = self.fc_prior_stats(prior_hidden)
        prior_mean, prior_std = extract_stats(prior_stats)
        prior_stoch = sample_gaussian(prior_mean, prior_std)

        # 计算后验分布（如果有观测）
        if obs_embed is not None:
            # 将当前确定性状态与当前观测嵌入进行拼接
            h_x_cat = torch.cat([deter_state, obs_embed], dim=-1)
            post_hidden = F.elu(self.fc_posterior_hidden(h_x_cat))
            post_stats = self.fc_posterior_stats(post_hidden)
            post_mean, post_std = extract_stats(post_stats)
            post_stoch = sample_gaussian(post_mean, post_std)
        else:
            post_mean, post_std = prior_mean, prior_std
            post_stoch = prior_stoch

        return deter_state, prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std
```

## 序列展开与重构损失

单步的 `RSSMCell` 完成了，但训练世界模型需要对整个时间序列进行展开（Rollout）。在训练阶段，给定过去一个回合（Episode）的完整观测序列 $x_{1:T}$ 和动作序列 $a_{1:T}$，我们需要推断出整个轨迹的状态分布，并计算损失。

这里的损失函数源自变分下界（Variational Lower Bound，通常称为 ELBO）。我们将极大化对数似然转化为最小化以下两项之和：

1. **重构与预测损失**：利用后验状态 $(h_t, z_t)$ 解码当前图像 $x_t$，并预测对应奖励 $r_t$。
2. **动态 KL 散度（Dynamics KL Divergence）**：在每一个时间步 $t$，先验预测的分布 $p_\theta(z_t \mid h_t)$ 与后验计算的分布 $q_\phi(z_t \mid h_t, x_t)$ 应该尽可能接近。这确保了模型在没有观测时依然能做出现实合理的想象。

我们接下来实现整个序列展开的逻辑，并在每个时间步记录先验和后验分布。

```python
class RSSM(nn.Module):
    """处理整个时间序列的顶层 RSSM 模块。"""
    def __init__(self, action_dim, deter_dim=200, stoch_dim=30, hidden_dim=200):
        super().__init__()
        self.cell = RSSMCell(action_dim, deter_dim, stoch_dim, hidden_dim)

    def rollout_observation(self, obs_embeds, actions, init_deter=None, init_stoch=None):
        """
        在给定真实观测序列的情况下展开后验推断（主要用于模型训练）。

        参数:
        obs_embeds: 形状为 (T, B, hidden_dim) 的张量
        actions: 形状为 (T, B, action_dim) 的张量，注意这里的 action 应该是前一步的动作 a_{t-1}
        """
        seq_len, batch_size, _ = obs_embeds.shape

        # 初始化张量容器用于记录每一步的结果
        deter_states = []
        prior_means, prior_stds = [], []
        post_stochs, post_means, post_stds = [], [], []

        # 若未提供初始状态，则全零初始化
        if init_deter is None:
            prev_deter = torch.zeros(batch_size, self.cell.deter_dim, device=obs_embeds.device)
        else:
            prev_deter = init_deter

        if init_stoch is None:
            prev_stoch = torch.zeros(batch_size, self.cell.stoch_dim, device=obs_embeds.device)
        else:
            prev_stoch = init_stoch

        # 沿时间维度展开
        for t in range(seq_len):
            # 调用核心单元执行单步前向推断
            (prev_deter, prior_stoch, prior_mean, prior_std,
             prev_stoch, post_mean, post_std) = self.cell.forward_step(
                 prev_deter, prev_stoch, actions[t], obs_embeds[t]
             )

            # 记录数据
            deter_states.append(prev_deter)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_stochs.append(prev_stoch)
            post_means.append(post_mean)
            post_stds.append(post_std)

        # 将列表堆叠为形状为 (T, B, Dimension) 的张量
        return (
            torch.stack(deter_states),
            torch.stack(post_stochs),
            (torch.stack(prior_means), torch.stack(prior_stds)),
            (torch.stack(post_means), torch.stack(post_stds))
        )
```

### KL 散度与信息瓶颈

<div align="center">
  <img src="/figures/04-latent-dynamics/source/07-rssm-scratch/planet-fig8.png" alt="PlaNet 的消融网格比较标准变分目标与 latent overshooting，展示多步一致性约束对 RSSM 学习的实验作用。" width="86%">

_图 4.7-4：PlaNet 的消融网格比较标准变分目标与 latent overshooting，展示多步一致性约束对 RSSM 学习的实验作用。 出处：Danijar Hafner；Timothy Lillicrap；Ian Fischer；Ruben Villegas；David Ha；Honglak Lee；James Davidson，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 8。_

</div>

在得到了长度为 $T$ 的先验统计量和后验统计量之后，我们需要计算 KL 散度。对于两个多变量高斯分布 $p = \mathcal{N}(\mu_1, \Sigma_1)$ 和 $q = \mathcal{N}(\mu_2, \Sigma_2)$（假设方差为对角矩阵），从 $q$ 到 $p$ 的 KL 散度具有解析解：

$$ D_{KL}(q \parallel p) = \frac{1}{2} \sum_{i=1}^{D_z} \left( \log \frac{\sigma_{1,i}^2}{\sigma_{2,i}^2} + \frac{\sigma_{2,i}^2 + (\mu_{2,i} - \mu_{1,i})^2}{\sigma_{1,i}^2} - 1 \right) $$

<div align="center"><img src="/figures/04-latent-dynamics/latex/07-rssm-scratch/diagonal-gaussian-kl-terms.png" alt="对角高斯 KL 逐维包含方差尺度比、后验扩散和均值错配三类贡献" width="86%">

_图 4.7-5：每个潜变量维度同时比较 q 与 p 的尺度、后验扩散和均值位置；这些贡献相加后才得到整条潜状态的 KL。本文根据上式绘制。_

</div>

不同 Dreamer 版本会把 KL 拆成带不同 stop-gradient 方向的两项，用来分别训练动力学先验和表示后验；基础 RSSM 并不要求只有这一种写法。常见的稳定化手段还包括 KL 权重 $\beta$ 与 free nats。后者把小于阈值的 KL **损失贡献**截在阈值处，使模型不必继续压缩这部分信息；它并不会把分布本身的 KL 强行改成该数值。

```python
def kl_loss(prior_stats, post_stats, free_nats=3.0):
    """
    计算整个序列上的 KL 散度损失。
    使用 free_nats 防止后验崩溃（Posterior Collapse）。
    """
    prior_mean, prior_std = prior_stats
    post_mean, post_std = post_stats

    # 构造分布对象
    prior_dist = torch.distributions.Normal(prior_mean, prior_std)
    post_dist = torch.distributions.Normal(post_mean, post_std)

    # 计算 KL 散度，并在潜变量维度求和
    kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist).sum(dim=-1)

    # Free nats：阈值以下的 KL 不再产生额外梯度
    free_nats_tensor = torch.full_like(kl, free_nats)
    kl_constrained = torch.max(kl, free_nats_tensor)

    # 在时间轴和批次轴上求平均
    return kl_constrained.mean()
```

## 训练循环的高层视点

最后把一次世界模型更新中的数据流串起来：

1. 从重播缓冲区（Replay Buffer）中提取出一批包含视频帧图像和动作的历史轨迹。
2. 将图像 $x_{1:T}$ 送入卷积自编码器的**编码器**，压缩得到观测嵌入 `obs_embeds`。
3. 利用 `RSSM` 的 `rollout_observation` 沿着时间步扫描，获得一系列确定性状态 $h_{1:T}$ 和后验随机状态 $z_{1:T}$。
4. 将 $h_t$ 与 $z_t$ 拼接，送入**解码器**（重构出 $\hat{x}_t$），并送入**奖励预测器**（预测 $\hat{r}_t$）。
5. 计算重构均方误差（MSE），同时使用 `kl_loss` 计算先验与后验的差异。将这些误差反向传播回所有网络。

这套分工让模型既能借助观测推断当前状态，也能在没有未来观测时仅靠先验向前滚动。它因此成为 PlaNet 和 Dreamer 系列的重要基础模块。

## 练习

1. 在 `RSSMCell` 中，为什么 `gru_input` 必须包含 $z_{t-1}$ 而不仅仅是 $a_{t-1}$？尝试用自己的话从马尔可夫性的角度进行分析。
   > **提示**：如果在上一步遇到了一阵无法预期的侧风（由 $z_{t-1}$ 捕获），这阵风带来的影响是否应当被记忆在 $h_t$ 中以指导未来的推断？
2. `extract_stats` 中我们加上了 `min_std=0.1`。如果没有这个常数限制，KL 散度的计算中该公式哪一项最可能发生数值爆炸（NaN）现象？
   > **提示**：观察分母项。
3. 如果训练中 KL 散度很低，而解码器重构的图像仍然很模糊，这可能说明什么？
   > **提示**：这通常被称为“后验崩溃”（Posterior Collapse）。思考当随机状态不再携带额外信息时，模型实质上退化成了什么？
