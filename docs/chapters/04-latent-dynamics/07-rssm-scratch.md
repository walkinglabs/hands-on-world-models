# 潜在动力学（RSSM）的从零开始实现

在前几章中，我们探讨了如何通过自编码器将高维观测（如图像）压缩为低维的潜在表示（Latent Representation），并初步介绍了如何使用循环神经网络（RNN）在时间维度上建模这些状态的演化。然而，真实的物理世界充满了不确定性。一个纯粹确定性的动力学模型（如标准的RNN）在面对多条可能的未来分支时，往往会产生模糊的预测，或者因为微小的误差而在长程预测中彻底崩溃。

为了在潜在空间中既能保持对历史信息的长期记忆，又能稳健地模拟未来的不确定性，Hafner等人在他们的经典工作“Learning Latent Dynamics for Planning from Pixels (PlaNet)” [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551) 中，提出了一种名为**循环状态空间模型**（Recurrent State Space Model, 简称RSSM）的优雅架构。RSSM 并非凭空出现，它是对隐马尔可夫模型（HMM）、卡尔曼滤波（Kalman Filter）以及变分自编码器（VAE）等早期序列生成模型在深度学习时代的集大成者。

在本节中，我们将完全从零开始，使用基本张量操作和标准神经网络层，一步步构建出完整的 RSSM。我们将从最基础的标量动力学方程出发，严格推导其多维张量形式，并最终落实到可以直接运行的代码实现上。

## 历史背景与动力学建模的困境

在深度学习处理时间序列的早期，研究者通常依赖确定性的循环神经网络（如 LSTM 或 GRU）来预测未来的状态。以自动驾驶为例，给定过去的视频帧序列，我们希望预测接下来几秒的画面。如果是确定性模型，当路口前方面临左转或直行的可能性时，网络为了最小化预测误差（均方误差），往往会输出左转和直行的“平均图像”——也就是一团模糊的像素。

为了引入随机性，变分循环神经网络（Variational RNNs） [[Chung et al., 2015]](https://arxiv.org/abs/1506.02216) 被提出，它们在每个时间步引入一个服从高斯分布的随机变量。然而，纯粹依靠随机状态跨越时间步传递信息的模型，在优化时极难保留跨越数百步的长期依赖。

Hafner 敏锐地洞察到了这一矛盾。在 [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551) 中，他提出：**状态的记忆机制应当是确定性的，而对未来的推断应当包含随机性。** 这种将状态强制解耦为“确定性部分”和“随机性部分”的思想，正是 RSSM 在众多世界模型中脱颖而出的核心基础，并在后续的 Dreamer 系列论文 [[Hafner et al., 2020]](https://arxiv.org/abs/2010.02193); [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104) 中被不断发扬光大。

## 从抛物运动到分离状态空间

让我们暂时抛开高维张量和神经网络，回到高中物理中最经典的抛物运动。

假设你在一个大风天抛出一个网球。网球在 $t$ 时刻的位置不仅取决于 $t-1$ 时刻的位置和速度，还取决于你施加的力量（动作），以及当时阵风的扰动。

在一个简化的、一维的离散时间系统中，网球的速度 $v_t$ 可以近似写为：
$$v_t = v_{t-1} + a \cdot \Delta t + \epsilon_t$$

其中，$a$ 是你施加的加速度，$\epsilon_t$ 则是代表阵风影响的随机噪声。我们可以敏锐地发现，等式的右边由两部分组成：
1. **确定性部分**（$v_{t-1} + a \cdot \Delta t$）：它精确记录了系统在此前的状态以及受控输入，这部分是完全可以被物理法则（或网络权重）确定的。
2. **随机性部分**（$\epsilon_t$）：这是系统不可预知的外部扰动，或者说是一种内在的不确定性。

在真实的非线性世界中，我们将上述系统推广。令 $h_t$ 为存储过去所有历史信息的确定性状态，令 $z_t$ 为捕捉当前时刻随机性的随机状态。如果我们在时间步 $t$ 执行了动作 $a_t$，系统的动力学可以拆解为以下三个严谨的组件。

### 确定性状态更新方程

确定性状态 $h_t$ 必须融合上一时刻的记忆 $h_{t-1}$、上一时刻的具体遭遇 $z_{t-1}$ 以及上一时刻的动作 $a_{t-1}$。这是一个纯粹的非线性映射：
$$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$$

这里，$f_\theta$ 通常被实现为一个门控循环单元（GRU）。需要极其注意的是，$h_t$ 的更新**不依赖**当前时刻 $t$ 的任何新观测，它仅仅是对过去的总结。

### 先验动力学（Prior Dynamics）

在没有看到时刻 $t$ 真实发生的画面之前，我们需要基于我们对世界的理解（即 $h_t$），去“想象”当前时刻可能发生什么。这就是**先验分布**。为了数学上的可计算性，我们通常假设它服从多变量高斯分布：
$$p_\theta(z_t \mid h_t) = \mathcal{N}(\mu_\theta(h_t), \Sigma_\theta(h_t))$$

先验动力学网络通过多层感知机（MLP）输出均值 $\mu_\theta$ 和协方差矩阵（通常被限制为对角阵）$\Sigma_\theta$。

### 后验推断（Posterior Inference）

当我们睁开眼睛，切实看到了时刻 $t$ 的画面（观测 $x_t$）后，我们需要更新我们的认知。结合过去的记忆 $h_t$ 和当前的观测特征，我们提取出当前真实的随机状态 $z_t$。这就是**后验分布**：
$$q_\phi(z_t \mid h_t, x_t) = \mathcal{N}(\mu_\phi(h_t, x_t), \Sigma_\phi(h_t, x_t))$$

后验推断由编码器提供支持，其分布必须在训练阶段作为先验分布试图逼近的目标（Target）。

> 💡 **复杂机制的直觉映射：**
> 想象你在黑暗中摸索前行（先验动力学预测可能的位置 $h_t \to z_t$），突然你打开手电筒看清了周围的陈设（观测 $x_t$），此时你在脑海中立刻修正了对自己确切位置的判断（后验推断 $h_t, x_t \to z_t$）。在训练世界模型时，我们要让闭着眼睛摸索的预测，尽可能贴近睁开眼睛看到的真实结果（这正是最小化 KL 散度的物理意义）。

## 严谨的维度推演

在实现代码之前，我们必须对网络中流动的数据张量的形状（Shape）有极其精确的掌控。设批大小为 $B$，序列长度为 $T$。

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

```{.python .input}
#@tab pytorch
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
#@tab tensorflow
# 在 TensorFlow 对应实现中，通常使用 tf.keras.layers.Dense 和 tf.keras.layers.GRUCell。
# 由于核心张量逻辑完全一致，此处主要关注其严格的数学推导在代码中的映射。
```

我们需要几个辅助函数来帮助我们处理高斯分布的参数计算和重参数化技巧（Reparameterization Trick）。为了保证训练的数值稳定性，方差往往不直接被预测，而是预测标准的对数方差（Log-Variance），或者使用 Softplus 激活函数加上一个极小的偏移量。

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab pytorch
    # 将以下方法补充至 RSSMCell 类内部
    def forward_step(self, prev_deter, prev_stoch, prev_action, obs_embed=None):
        """
        执行单个时间步的状态推导。
        如果 obs_embed 为 None，则纯粹执行先验想象；否则计算后验分布。
        """
        # (**计算确定性状态更新**)
        # 对应该公式        za_cat = torch.cat([prev_stoch, prev_action], dim=-1)
        gru_input = F.elu(self.fc_state_action(za_cat))
        
        # GRU 内部状态更新
        # h_t = GRU(input_t, h_{t-1})
        deter_state = self.cell(gru_input, prev_deter)
        
        # (**计算先验分布**)
        # 对应该公式        prior_hidden = F.elu(self.fc_prior_hidden(deter_state))
        prior_stats = self.fc_prior_stats(prior_hidden)
        prior_mean, prior_std = extract_stats(prior_stats)
        prior_stoch = sample_gaussian(prior_mean, prior_std)
        
        # (**计算后验分布（如果有观测）**)
        if obs_embed is not None:
            # 对应该公式            # 将当前确定性状态与当前观测嵌入进行拼接
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
1. **重构损失（Reconstruction Loss）**：利用后验推断出的状态 $(h_t, z_t)$ 必须能够解码还原出当前的图像 $x_t$ 和对应的奖励 $r_t$。
2. **动态 KL 散度（Dynamics KL Divergence）**：在每一个时间步 $t$，先验预测的分布 $p_\theta(z_t \mid h_t)$ 与后验计算的分布 $q_\phi(z_t \mid h_t, x_t)$ 应该尽可能接近。这确保了模型在没有观测时依然能做出现实合理的想象。

我们接下来实现整个序列展开的逻辑，并在每个时间步记录先验和后验分布。

```{.python .input}
#@tab pytorch
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
            # [**调用核心单元执行单步前向推断**]
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

在得到了长度为 $T$ 的先验统计量和后验统计量之后，我们需要计算 KL 散度。对于两个多变量高斯分布 $p = \mathcal{N}(\mu_1, \Sigma_1)$ 和 $q = \mathcal{N}(\mu_2, \Sigma_2)$（假设方差为对角矩阵），从 $q$ 到 $p$ 的 KL 散度具有解析解：

$$ D_{KL}(q \parallel p) = \frac{1}{2} \sum_{i=1}^{D_z} \left( \log \frac{\sigma_{1,i}^2}{\sigma_{2,i}^2} + \frac{\sigma_{2,i}^2 + (\mu_{2,i} - \mu_{1,i})^2}{\sigma_{1,i}^2} - 1 \right) $$
:eqlabel:eq_kl_divergence

值得注意的是，RSSM 中通常会将后验分布的参数截断梯度（Stop Gradient）传递给先验，反之亦然。甚至会对这两部分的学习率进行解耦处理。为了保证训练不会因为某一步太大的误差而崩溃，通常还会在 KL 散度中引入一个超参数 $\beta$，或者使用 Free Nats 机制强制设定一个 KL 散度的最小下界。

```{.python .input}
#@tab pytorch
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
    
    # 应用 Free Nats（最小限度信息约束）
    free_nats_tensor = torch.full_like(kl, free_nats)
    kl_constrained = torch.max(kl, free_nats_tensor)
    
    # 在时间轴和批次轴上求平均
    return kl_constrained.mean()
```

## 训练循环的高层视点

到目前为止，我们已经用极致详尽的代码将极其复杂的 RSSM 分解。为了让你能够一览众山小，我们来简要梳理一下在每一次网络权重更新前，数据到底经历了怎样的流动：

1. 从重播缓冲区（Replay Buffer）中提取出一批包含视频帧图像和动作的历史轨迹。
2. 将图像 $x_{1:T}$ 送入卷积自编码器的**编码器**，压缩得到观测嵌入 `obs_embeds`。
3. 利用 `RSSM` 的 `rollout_observation` 沿着时间步扫描，获得一系列确定性状态 $h_{1:T}$ 和后验随机状态 $z_{1:T}$。
4. 将 $h_t$ 与 $z_t$ 拼接，送入**解码器**（重构出 $\hat{x}_t$），并送入**奖励预测器**（预测 $\hat{r}_t$）。
5. 计算重构均方误差（MSE），同时使用 `kl_loss` 计算先验与后验的差异。将这些误差反向传播回所有网络。

这种极其解耦而又自成一体的结构设计，直接奠定了今天世界模型在解决复杂环境决策任务时的统治地位。

## 练习

1. 在 `RSSMCell` 中，为什么 `gru_input` 必须包含 $z_{t-1}$ 而不仅仅是 $a_{t-1}$？尝试用自己的话从马尔可夫性的角度进行分析。
   > **提示**：如果在上一步遇到了一阵无法预期的侧风（由 $z_{t-1}$ 捕获），这阵风带来的影响是否应当被记忆在 $h_t$ 中以指导未来的推断？
2. `extract_stats` 中我们加上了 `min_std=0.1`。如果没有这个常数限制，KL 散度的计算中该公式哪一项最可能发生数值爆炸（NaN）现象？
   > **提示**：观察分母项。
3. 在现实训练中，如果你发现 KL 散度极低，模型完美地使得先验等同于后验，但此时解码器重构出的图像却极其模糊，这说明发生了什么问题？
   > **提示**：这通常被称为“后验崩溃”（Posterior Collapse）。思考当随机状态不再携带额外信息时，模型实质上退化成了什么？

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
