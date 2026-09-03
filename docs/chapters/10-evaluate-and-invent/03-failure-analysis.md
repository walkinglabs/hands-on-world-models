# 10.3 故障分析与模型局限性

前几章把世界模型拆成编码、动力学、解码和控制环节。评测时也要沿着这条链查错：单步模型可能准确，多步展开却逐渐偏离；观测重建可能清晰，对动作却不敏感；训练损失持续下降，潜变量却越来越少携带输入信息。

多步模型展开可能因分布偏移而失真 [[Talvitie, 2014]](https://ojs.aaai.org/index.php/AAAI/article/view/8852)，在学到的“梦境”中优化控制器也可能利用模型误差 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。此外，一步预测似然与下游控制效果可能并不一致，这就是模型式强化学习中的目标错配 [[Lambert et al., 2020]](https://arxiv.org/abs/2002.04523)。Alemi 等人讨论的则是 VAE 中 ELBO、互信息与潜变量失活等问题 [[Alemi et al., 2018]](https://arxiv.org/abs/1711.00464)，更接近“后验坍塌”，不应直接当作一般联合嵌入方法的“表征坍塌”证据。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/03-failure-analysis/worldmodels-fig18.png" alt="World Models 的控制器学会让梦境中的火球自动消失，直接展示策略如何利用不准确的学习模型。" width="86%">

_图 10.3-1：World Models 的控制器学会让梦境中的火球自动消失，直接展示策略如何利用不准确的学习模型。 出处：David Ha；Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 18。_
</div>

本节从一个一维递推例子开始，区分误差累积、动力学放大和分布偏移；随后检查预测目标与控制目标的错配，以及潜变量失活。重点不是证明所有模型都会以同一种方式失败，而是把可观察的症状变成可复现的诊断实验。

## 自回归展开中的复合误差

对自回归世界模型，第 $t+1$ 步接收的往往是第 $t$ 步的预测，而不是数据集中的真值状态。单步误差因此会改变下一步输入分布。先用一个线性递推例子把这种传播写清楚。

### 从等比数列到误差累积

让我们暂时忘记深度学习、张量和高维隐空间，回到初等代数课堂。假设我们有一个最简单的一维物理系统，其真实的状态转移规律是一个一元一次函数。令 $s_t \in \mathbb{R}$ 表示在时间 $t$ 的系统状态，真实的动力学方程为：

$$s_{t+1} = f(s_t) = \lambda s_t$$

其中 $\lambda$ 是一个常数。这是一个典型的等比数列（Geometric Progression）。如果我们在初始时刻知道了确切的状态 $s_0$，那么第 $T$ 步的状态显然为 $s_T = \lambda^T s_0$。

现在，假设我们使用神经网络训练了一个世界模型来逼近这个真实系统。由于训练数据有限、优化算法不完美或模型容量限制，我们的模型 $\hat{f}$ 不可能完全等于真实函数 $f$。假设在每一步单步预测中，模型总是会产生一个大小为 $\epsilon$ 的恒定误差（即 $\hat{f}(x) = \lambda x + \epsilon$）。

当我们利用这个模型进行自回归预测（Rollout）时，令预测状态为 $\hat{s}_t$，且初始状态完全准确，即 $\hat{s}_0 = s_0$。让我们手动展开前几步的预测：

第一步：
$$\hat{s}_1 = \hat{f}(\hat{s}_0) = \lambda \hat{s}_0 + \epsilon = \lambda s_0 + \epsilon$$

若令 $\epsilon\ge 0$，第一步的绝对误差为 $e_1 = |\hat{s}_1 - s_1| = \epsilon$。这正是我们设定的单步偏差。

第二步：
$$\hat{s}_2 = \hat{f}(\hat{s}_1) = \lambda \hat{s}_1 + \epsilon = \lambda (\lambda s_0 + \epsilon) + \epsilon = \lambda^2 s_0 + \lambda \epsilon + \epsilon$$

第二步的真实状态是 $s_2 = \lambda^2 s_0$。此时第二步的绝对误差满足：
$$e_2 = |\hat{s}_2 - s_2| = |\epsilon(\lambda + 1)| \le \epsilon (1 + |\lambda|)$$

以此类推，到了第 $T$ 步，预测状态为：
$$\hat{s}_T = \lambda^T s_0 + \epsilon \sum_{i=0}^{T-1} \lambda^i$$

第 $T$ 步的累积误差因此有上界：
$$e_T = |\hat{s}_T - s_T| \le \epsilon \sum_{i=0}^{T-1} |\lambda|^i$$

当 $|\lambda|\neq 1$ 时，上式等于 $\epsilon(|\lambda|^T-1)/(|\lambda|-1)$；当 $|\lambda|=1$ 时，上界是 $T\epsilon$。

这个上界说明：当 $|\lambda|>1$ 时，持续的单步偏差可能被几何放大；当 $|\lambda|=1$ 时，上界线性增长；当 $|\lambda|<1$ 时，上界趋于常数。它并不声称实际误差必然达到上界，因为不同时间步的误差方向可能互相抵消。

### 协变量偏移（Covariate Shift）与利普希茨连续性

在实际的世界模型中，状态 $s_t$ 通常是高维空间（如 $\mathbb{R}^d$）中的向量，并且真实的动力学函数 $f: \mathbb{R}^d \times \mathcal{A} \rightarrow \mathbb{R}^d$ 是高度非线性的，并且还受到动作 $a_t$ 的影响。我们如何将一维标量的结论推广到多维非线性系统？

为了严谨地分析这一问题，我们需要引入微积分中的利普希茨连续性（Lipschitz Continuity）。对于真实转移函数 $f(\cdot)$，假设在给定动作序列的情况下，其关于状态空间满足 $L$-Lipschitz 连续，即存在常数 $L>0$，使得对于任意两个状态 $x, y \in \mathbb{R}^d$：

$$\| f(x) - f(y) \|_2 \le L \| x - y \|_2$$

这里的常数 $L$ 就相当于前文一维等比数列中的 $|\lambda|$。它在物理上反映了系统动力学的“敏感度”：如果系统处于混沌边缘（如“蝴蝶效应”），微小的状态变化会导致截然不同的未来，此时 $L \gg 1$。

同时，我们假设世界模型 $\hat{f}$ 在整个状态空间上的单步预测误差有界，即对于任意状态 $s$，都有：

$$\| \hat{f}(s) - f(s) \|_2 \le \epsilon$$

当我们进行 $T$ 步推演时，设 $\hat{s}_0 = s_0$。对于任意步 $t$，推演误差 $e_t = \| \hat{s}_t - s_t \|_2$。我们利用三角不等式（Triangle Inequality），这是一种将复杂偏差拆解为已知变量相加的几何学利器。

我们考察第 $t+1$ 步的误差：
$$e_{t+1} = \| \hat{s}_{t+1} - s_{t+1} \|_2 = \| \hat{f}(\hat{s}_t) - f(s_t) \|_2$$

我们在公式中间巧妙地加上并减去一个虚拟项 $f(\hat{s}_t)$，即“如果我们从有误差的当前预测状态 $\hat{s}_t$ 出发，应用**真实**的动力学函数，会得到什么结果”。

$$e_{t+1} = \| \hat{f}(\hat{s}_t) - f(\hat{s}_t) + f(\hat{s}_t) - f(s_t) \|_2$$

应用三角不等式 $\|A + B\| \le \|A\| + \|B\|$：

$$e_{t+1} \le \underbrace{\| \hat{f}(\hat{s}_t) - f(\hat{s}_t) \|_2}_{\text{单步模型误差}} + \underbrace{\| f(\hat{s}_t) - f(s_t) \|_2}_{\text{动力学放大效应}}$$

观察该公式的两部分。第一部分正是我们在 $\hat{s}_t$ 处的模型预测误差，根据假设该公式，它不超过 $\epsilon$。第二部分则是真实函数 $f$ 对输入偏差的放大，根据 Lipschitz 假设该公式，它满足 $\| f(\hat{s}_t) - f(s_t) \|_2 \le L \| \hat{s}_t - s_t \|_2 = L e_t$。

因此，我们得到了误差的递推不等式：
$$e_{t+1} \le \epsilon + L e_t$$

<div align="center">
<img src="/figures/10-evaluate-and-invent/latex/03-failure-analysis/lipschitz-error-contribution.png" alt="每一步注入的单步模型误差在后续推演中被不同次数的 Lipschitz 常数放大，最终形成几何级数" width="86%">

_图 10.3-2：越早注入的单步误差经历越多次 L 放大；把各步贡献相加，便得到推演误差的几何级数上界。_
</div>

由于 $e_0 = 0$，我们可以再次使用初等数列求和的技巧，将其展开至第 $T$ 步：

$$e_T \le \epsilon \sum_{i=0}^{T-1} L^i$$

这给出了高维非线性系统中的最坏情况上界：$L>1$ 时可能几何放大，$L=1$ 时至多线性累积，$L<1$ 时保持有界。接下来更关键的问题是：用于推导的统一单步误差界 $\epsilon$，在模型离开训练分布后往往并不成立。

请注意单步模型误差项 $\| \hat{f}(\hat{s}_t) - f(\hat{s}_t) \|_2$。在训练阶段，世界模型是通过最小化由训练集（真实经验回放池）采样出的真实状态 $s_t$ 的单步预测误差来优化的，即我们极力让 $\| \hat{f}(s_t) - f(s_t) \|_2$ 变小。然而，在推演（Rollout）阶段，模型在时间步 $t$ 接收到的输入是它自己上一阶段的预测输出 $\hat{s}_t$。随着时间推移，$\hat{s}_t$ 会逐渐偏离真实的数据分布。这就意味着，模型被强迫在其**从未见过的、分布外的数据（Out-of-Distribution, OOD）** 上进行预测。一旦发生这种协变量偏移，由于深度神经网络在分布外的高方差特性，单步误差界限 $\epsilon$ 在实际中根本无法保持恒定，它本身也会随着步数急剧增大。这就是为何在长视野预测中，世界模型生成的画面或状态最终会崩溃成无意义的噪声或静态图像。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/03-failure-analysis/scheduled-fig1.png" alt="Scheduled Sampling 对照训练时使用真实前项与推断时使用模型输出的路径，揭示自回归误差为何把后续输入推离训练分布。" width="86%">

_图 10.3-3：Scheduled Sampling 对照训练时使用真实前项与推断时使用模型输出的路径，揭示自回归误差为何把后续输入推离训练分布。 出处：Samy Bengio；Oriol Vinyals；Navdeep Jaitly；Noam Shazeer，[Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)（2015），Figure 1。_
</div>

## 目标错配：预测与控制的鸿沟

世界模型很少仅仅为了“生成视频”而存在，它们最终的归宿往往是作为大脑，用于在隐空间中规划动作（Planning）并执行控制任务。这里隐藏着另一个深刻的局限性：**模型训练的目标函数，与其被使用的目标函数之间存在错配。**

### 最大似然估计与控制最优性的割裂

在标准的机器学习范式中，我们通常使用均方误差（MSE）或负对数似然（Negative Log-Likelihood, NLL）来训练状态转移模型。对于一个多元高斯分布的世界模型，我们最小化：

$$\mathcal{L}_{\text{model}} = \mathbb{E}_{\mathcal{D}} \left[ \frac{1}{2} \| s_{t+1} - \hat{s}_{t+1} \|^2_2 \right]$$

从经典物理力学的角度来看，MSE 衡量的是两点之间欧几里得距离的平方。这意味着模型在状态空间中的每一个维度都被视为同等重要，它在平等地拟合环境中的每一个像素或每一丝微小的震动。

然而，在强化学习或控制理论中，我们的目标是最大化长期累积奖励（Cumulative Reward）：

$$J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]$$

两种目标并不自动一致。以驾驶为例，挡风玻璃上的雨滴会贡献许多像素误差，前车刹车灯只占很小区域，却可能对动作选择更重要。有限容量的模型若只优化平均像素误差，可能优先改善大面积纹理，而没有可靠保存影响控制的稀有信号。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/03-failure-analysis/lambert-fig1.png" alt="目标错配图把动力学模型的似然训练目标与控制器的奖励目标并列，明确标出两者之间没有自动一致性。" width="86%">

_图 10.3-4：目标错配图把动力学模型的似然训练目标与控制器的奖励目标并列，明确标出两者之间没有自动一致性。 出处：Nathan Lambert et al.，[Objective Mismatch in Model-based Reinforcement Learning](https://arxiv.org/abs/2002.04523)（2020），Figure 1。_
</div>

模拟引理一类结果会把策略价值误差上界与奖励误差、转移分布误差以及有效时域联系起来。具体常数取决于采用的全变差定义、奖励界和两个模型是否共享奖励函数，但共同信息是：当 $\gamma$ 接近 1 时，局部模型误差可能在长时域中被显著放大。这个最坏情况界同样不会告诉我们哪些观测维度对奖励最重要，所以仍需要任务加权的诊断和闭环实验。

## 变分下界与后验坍塌

除了推演误差和目标错配外，潜变量模型还可能出现**后验坍塌（Posterior Collapse）**：近似后验不再依赖输入，潜变量携带的信息显著减少。它是表征失效的一种具体机制，但不应与联合嵌入方法中所有形式的“表征坍塌”混为一谈。

<div align="center">
<img src="/figures/10-evaluate-and-invent/source/03-failure-analysis/alemi-fig1.png" alt="Broken ELBO 的率失真相图把零率自动解码区域与携带信息的编码区域分开，显示潜变量失活在 ELBO 几何中的位置。" width="86%">

_图 10.3-5：Broken ELBO 的率失真相图把零率自动解码区域与携带信息的编码区域分开，显示潜变量失活在 ELBO 几何中的位置。 出处：Alexander A. Alemi et al.，[Fixing a Broken ELBO](https://arxiv.org/abs/1711.00464)（2018），Figure 1。_
</div>

为了彻底理解这一点，我们必须不可避免地直面变分自编码器（VAE）的数学内核。在世界模型中，我们将高维观测 $x_t$ 压缩为低维隐状态 $z_t$，并通过优化证据下界（Evidence Lower Bound, ELBO）来训练：

$$\log p(x_{1:T}) \ge \sum_{t=1}^T \underbrace{\mathbb{E}_{q}[\log p(x_t|z_t)]}_{\text{重构项}} - \underbrace{D_{KL}(q(z_t | x_t) \| p(z_t | z_{t-1}, a_{t-1}))}_{\text{正则化/动力学项}}$$

让我们仔细观察该公式。等式的右侧由两部分组成，并且它们之间存在一个天然的对抗张力：

1. **重构项（Reconstruction Term）** 迫使后验分布 $q(z_t|x_t)$（编码器）必须保留足够的信息以还原原始图像 $x_t$。
2. **KL 散度项（KL Divergence Term）** 充当正则化器，它强迫编码器的输出分布 $q(z_t|x_t)$ 去尽可能贴近先验动力学模型的预测 $p(z_t|z_{t-1}, a_{t-1})$。

这里隐藏着一个深渊。

::: tip 唯一的类比：慵懒的学生与苛刻的导师
想象隐空间表征 $z_t$ 是一个正在准备期末考试的“学生”。重构项是一份非常严格的试卷，要求学生必须记住大量的细节（比如图像中的每个像素）。而 KL 散度项则像是一位极度追求死板一致性的“导师”，他要求学生今天的知识状态（后验）必须完全符合他昨天的预测（先验）。

如果这张试卷太难（或者解码器已经能从其他通路完成任务），学生可能会选择放弃答题。只要让 $q(z_t|x_t)$ 不再依赖 $x_t$ 并贴近先验，就能把 KL 项压低。此时系统接近后验坍塌：编码器输出对输入不敏感，潜变量不再提供解码器原本应使用的信息。
:::

当自回归解码器足够强时，它可能主要依赖自身历史而忽略 $z_t$。这会让优化更倾向于将后验贴近先验。严重的后验坍塌会削弱隐状态对观测和任务变量的表征，进而损害依赖它的预测或规划；影响程度仍取决于模型是否还有其他信息通路。

## 实验与代码实现：诊断世界模型

下面实现一个不参与训练的 `WorldModelDiagnoser`，用来记录多步误差曲线，并统计后验均值中有多少维度会随输入变化。这些量是诊断线索，不是单独的故障判决。

我们将使用 PyTorch 进行实现。请注意，接下来的代码段展示了如何在评估阶段收集并剥析隐变量的统计特性，以此来检测 KL 散度坍缩，同时比较单步预测 MSE 与多步推演 MSE 之间的非线性分化。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# (定义基础的诊断工具类)
class WorldModelDiagnoser:
    def __init__(self, encoder, transition_model, decoder, device='cpu'):
        """
        初始化诊断器，需要传入世界模型的三个核心组件：
        encoder: q(z_t | x_t)
        transition_model: p(z_t | z_{t-1}, a_{t-1})
        decoder: p(x_t | z_t)
        """
        self.encoder = encoder
        self.transition_model = transition_model
        self.decoder = decoder
        self.device = device

        # 确保所有组件处于评估模式，关闭 Dropout 和 BatchNorm 的变动
        self.encoder.eval()
        self.transition_model.eval()
        self.decoder.eval()

    @torch.no_grad()
    def evaluate_compounding_error(self, initial_obs, actions, true_trajectory):
        """
        记录开环误差随预测步长的变化。
        initial_obs: 初始观测图像，形状 (1, C, H, W)
        actions: 给定的动作序列，形状 (T, action_dim)
        true_trajectory: 真实的观测轨迹，用于比对，形状 (T, C, H, W)
        返回每一步的 MSE 误差。
        """
        T = actions.size(0)
        mses = []

        # 1. 初始化第一步隐状态
        # 为了严谨，这里我们获取后验分布的均值作为确定性的状态表示
        z_t = self.encoder(initial_obs.to(self.device)).mean

        for t in range(T):
            # 2. 动力学模型前向推演一步 (先验预测)
            # z_t = f(z_{t-1}, a_{t-1})
            a_t = actions[t].unsqueeze(0).to(self.device)
            z_t = self.transition_model(z_t, a_t)

            # 3. 解码出预测的图像帧
            pred_obs = self.decoder(z_t)

            # 4. 计算与真实轨迹的像素级 MSE
            true_obs = true_trajectory[t].unsqueeze(0).to(self.device)
            mse = F.mse_loss(pred_obs, true_obs).item()
            mses.append(mse)

        return mses

    @torch.no_grad()
    def detect_representation_collapse(self, obs_batch):
        """
        检查后验均值是否出现大量失活维度。
        计算后验分布 q(z|x) 的方差统计量。如果所有维度的方差都极度接近先验分布(例如 N(0,I)的方差1)，
        或者均值几乎无视输入特征的变化，则表明出现了坍塌。
        obs_batch: 批次观测，形状 (B, C, H, W)
        """
        obs_batch = obs_batch.to(self.device)

        # 获取后验分布的均值和对数方差
        posterior = self.encoder(obs_batch)
        mu = posterior.mean
        log_var = posterior.logvar

        # 计算均值的批次内方差，即 "Active Units" 的检测标准 (Burda et al., 2015)
        # 衡量该隐变量维度是否对不同的输入做出了响应
        var_of_mu = torch.var(mu, dim=0)

        # 计算平均对数方差，看其是否趋近于0 (即方差趋近于1的常数)
        avg_log_var = torch.mean(log_var, dim=0)

        # 如果 var_of_mu 非常小 (例如 < 0.01)，说明该维度的均值完全不随数据变化而改变，属于失活 (Dead) 状态。
        active_units = torch.sum(var_of_mu > 0.01).item()
        total_units = mu.size(1)

        print(f"活跃隐变量维度 (Active Units): {active_units} / {total_units}")
        print(f"均值跨批次的平均方差: {torch.mean(var_of_mu):.4f}")
        print(f"隐变量自身的平均指数方差: {torch.exp(avg_log_var).mean():.4f}")

        return active_units
```

`evaluate_compounding_error` 模拟开环推演：除初始观测外，模型不再接收真值状态。得到的 `mses` 应画成“误差—预测步长”曲线，并和单步教师强制结果对照。曲线可能上升、饱和或波动；前面的推导只给出条件性的上界，不预言它一定是抛物线或指数曲线。

`detect_representation_collapse` 使用**活跃单元（Active Units）**统计：计算后验均值 $\mu$ 在不同输入样本间的方差 `var_of_mu`。若某一维几乎不随输入变化，它会被标为失活。但阈值依赖表征尺度和数据分布，因此还应同时查看每样本 KL、重构或任务性能，以及改变输入后表征是否响应。

## 小结

- 在自回归推演中，误差可能累积；只有当局部动力学具有持续放大性且误差不抵消时，最坏情况上界才呈几何增长。
- **目标错配**揭示了以最小化像素级 MSE 等指标为目标的预测训练，与最终最大化强化学习回报（Reward）之间存在理论鸿沟。模型可能会耗尽参数去拟合与决策无关的背景噪音。
- **后验坍塌**会让近似后验对输入不敏感。活跃单元、KL 分布和下游可预测性应结合起来看，单一阈值只能作为报警器。
