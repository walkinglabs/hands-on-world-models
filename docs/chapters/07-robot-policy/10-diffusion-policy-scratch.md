# 7.10 扩散策略的从零开始实现

在过去的几十年里，机器人学习社区一直在探索如何让智能体从人类演示中学习复杂的交互行为。传统的行为克隆（Behavior Cloning, BC）通常将策略学习建模为从状态到动作的确定性映射，或者简单的单峰高斯分布。然而，当面临多模态的人类演示时——例如，遇到障碍物时既可以从左侧绕过，也可以从右侧绕过——试图用均方误差（MSE）回归单一动作的模型往往会输出这两个合理动作的平均值，即控制机器人直接撞向正前方的障碍物。这就是机器人学习中著名的“布里丹之驴（Buridan's ass）”困境。

为了解决这一问题，研究者们曾引入混合密度网络（MDN）或基于能量的模型（EBM），但它们在训练稳定性或采样效率上依然存在瓶颈。直到最近，Chi 等人提出了扩散策略（Diffusion Policy）`[Chi et al., 2023]`，创造性地将动作生成建模为条件去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）`[Ho et al., 2020]`。扩散策略不仅优雅地解决了多模态动作分布的问题，还在多个复杂的机器人操作任务中展现出了惊人的稳定性与鲁棒性。

在本节中，我们将不依赖于抽象的高阶数学推导，而是从最基础的概率与线性变换出发，逐步推演并从零开始实现一个完整的扩散策略模型。

## 7.10.1 动作块与条件生成

在连续控制中，如果我们只预测当前时刻的单一动作 $a_t$，模型往往容易受到瞬时噪声的干扰而产生执行轨迹的抖动。扩散策略继承并发挥了“动作块（Action Chunking）”的思想 `[Zhao et al., 2023]`。

具体而言，在时刻 $t$，我们收集过去 $T_o$ 步的连续观测数据，形成观测序列 $\mathbf{O} = [o_{t-T_o+1}, \dots, o_t]$。我们的目标不再是仅仅预测下一步动作，而是预测未来 $T_a$ 步的完整动作序列，即**动作块** $\mathbf{A} = [a_t, a_{t+1}, \dots, a_{t+T_a-1}]$。
在实际推断时，机器人会执行这组生成序列的前几步动作，然后在下一个决策周期重新观测并规划。这种时间尺度上的重叠与平均，极大地平滑了控制轨迹。

在扩散策略的视角下，我们将多维动作序列 $\mathbf{A}$ 视作一幅一维的“图像”或信号序列，通过扩散模型来生成它。而历史观测序列 $\mathbf{O}$ 则作为指导动作生成过程的“条件（Condition）”。

## 7.10.2 前向加噪过程：从标量到矩阵

> [!NOTE]
> 我们可以将扩散过程想象为往一杯清水中滴入一滴墨水。最初，墨水的分布是高度集中且结构化的（即演示数据中精准的确定性动作）。随着时间的推移，水分子的随机热运动不断撞击墨水颗粒，结构逐渐瓦解，最终整杯水变成了均匀的灰色（即完全的纯高斯噪声）。我们用数学来精确描述这个热运动破坏结构的过程，并试图利用神经网络学习如何“时光倒流”，把均匀的灰水重新聚集回那一滴清晰的墨水。这是全篇唯一一次类比，接下来我们将完全依靠严格的代数推导。

为了便于理解，我们先抽离出动作序列中某一个特定时间点上的某一维度的标量动作，记为 $x$。最初的真实且精准的动作记为 $x_0$。
在第 $k$ 步加噪时（扩散步数 $k \in [1, K]$），我们以极小的比例 $\beta_k \in (0, 1)$ 向数据中注入标准正态分布的噪声 $\epsilon \sim \mathcal{N}(0, 1)$。同时，为了保持数据的方差不至于在多次加噪后发散爆炸，我们需要对上一时刻的值 $x_{k-1}$ 进行按比例的轻微缩放。标量形式的递推更新公式如下：

$$x_k = \sqrt{1 - \beta_k} x_{k-1} + \sqrt{\beta_k} \epsilon$$

其中，$\beta_k$ 被称为方差表（Variance Schedule），它通常被设定为随着 $k$ 的增加而缓慢变大的常数序列。

然而，在实际训练中，我们需要经历数十甚至数百步加噪。如果每一步都依赖上一步的迭代计算来获取 $x_k$，计算将极其低效。由于高斯分布具有优良的可加性（两个独立高斯随机变量的线性组合依然是高斯分布），我们可以直接推导出从 $x_0$ 一步跨越到 $x_k$ 的直接表达式。

令 $\alpha_k = 1 - \beta_k$，并定义其从第 $1$ 步到第 $k$ 步的累积乘积为 $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$。经过连续代入与方差合并，我们可以得到非常简洁的单步跳跃公式：

$$x_k = \sqrt{\bar{\alpha}_k} x_0 + \sqrt{1 - \bar{\alpha}_k} \epsilon$$

现在，我们将这个标量公式自然地推广到高维矩阵形式。设 $\mathbf{A}_0 \in \mathbb{R}^{T_a \times D_a}$ 为完整的原始动作块矩阵，其中 $T_a$ 是动作序列长度，$D_a$ 是动作空间的维度。注入的随机噪声 $\boldsymbol{\epsilon}$ 是与 $\mathbf{A}_0$ 形状完全相同的纯高斯噪声矩阵。那么，在第 $k$ 步被破坏后的加噪动作块 $\mathbf{A}_k$ 可以表示为：

$$\mathbf{A}_k = \sqrt{\bar{\alpha}_k} \mathbf{A}_0 + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}$$

## 7.10.3 逆向去噪过程与目标函数

训练扩散策略的核心在于学习一个深度神经网络 $\boldsymbol{\epsilon}_\theta$。该网络需要能够在给定当前带噪动作块 $\mathbf{A}_k$、当前的扩散时间步 $k$ 以及历史观测条件 $\mathbf{O}$ 的情况下，准确预测出在第 $k$ 步时究竟注入了怎样形状的真实噪声 $\boldsymbol{\epsilon}$。

优化目标非常直接，即网络预测的噪声与真实注入噪声之间的均方误差（MSE）：

$$\mathcal{L}_{DP} = \mathbb{E}_{\mathbf{A}_0, \boldsymbol{\epsilon}, k, \mathbf{O}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{A}_k, k, \mathbf{O}) \right\|_2^2 \right]$$

一旦这个网络训练完成，在实际推断（控制机器人）时，我们就可以从纯随机的高斯噪声 $\mathbf{A}_K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发。利用预测出的噪声 $\boldsymbol{\epsilon}_\theta$，我们可以依据 DDPM 的逆向采样推导公式，逐步减去预测出的噪声，恢复出干净的动作序列：

$$\mathbf{A}_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{A}_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \boldsymbol{\epsilon}_\theta(\mathbf{A}_k, k, \mathbf{O}) \right) + \sigma_k \mathbf{z}$$

其中 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是为了维持逆向随机过程的分布形状而加入的补偿项（当最后一步 $k=1$ 迈向 $k=0$ 时 $\mathbf{z}=0$），而 $\sigma_k$ 的方差系数通常取为 $\sqrt{\beta_k}$ 或者更精确的后验方差。

## 7.10.4 核心组件的代码实现

在这一部分，我们将基于 PyTorch 从零实现扩散策略的核心逻辑。首先，我们定义调度器（Scheduler），负责管理所有的常数并执行前向加噪。

(**定义DDPM的方差表与前向辅助常数**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DDPMScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # 构造线性增长的方差表 beta
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        # 计算 alpha 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算前向加噪所需的项，避免在训练循环中重复计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, original_samples, noise, timesteps):
        """实现基于方程式(7.10.4)的前向矩阵加噪过程"""
        # 根据 timestep 提取对应的系数，并将其形状对齐到输入张量
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)
        
        # 返回加噪后的动作块 A_k
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
```

接下来，我们需要构建条件去噪网络 $\boldsymbol{\epsilon}_\theta$。在图像生成中，U-Net 架构是标准的解决方案。而在扩散策略中，由于动作块通常是一个相对较短的一维序列，我们采用一维的条件卷积网络（1D Conditional CNN）。
为了将标量扩散步数 $k$ 提供给网络，我们需要使用正弦位置编码将其映射为高维特征。为了将历史观测序列 $\mathbf{O}$ 作为条件注入，我们采用特征线性调制（Feature-wise Linear Modulation, FiLM）技术对卷积层输出的均值和方差进行缩放与平移。

(**构建带有时间嵌入与条件特征注入的一维卷积块**)

```{.python .input}
#@tab pytorch
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        # 使用 1D 卷积处理时序连续的动作序列数据
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.Mish()
        
        # 将条件特征向量映射到线性调制的缩放 (scale) 和平移 (shift) 参数
        self.cond_encoder = nn.Linear(cond_dim, out_channels * 2)
        
    def forward(self, x, cond):
        out = self.conv(x)
        out = self.norm(out)
        
        # FiLM 操作: 基于条件 cond 生成特定的缩放与偏移
        # cond 的形状为 (Batch, cond_dim)
        cond_emb = self.cond_encoder(cond)
        scale, shift = cond_emb.chunk(2, dim=-1)
        # 将维度对齐到 (Batch, out_channels, 1) 以便在动作序列长度维度上进行广播
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        # 执行调制，使用 scale + 1.0 作为残差偏置策略
        out = out * (scale + 1.0) + shift
        return self.activation(out)
```

具备了基础的特征调制层之后，我们可以将其组装成一个完整的网络结构。真实的论文实现中通常会使用带有完整编码器-解码器和跳跃连接（Skip-Connection）的深层一维 U-Net。在此处，为了保证逻辑的清晰并突出参数传递方式，我们抽象实现一个多层堆叠的条件卷积预测网络。

(**组装完整的条件去噪预测网络**)

```{.python .input}
#@tab pytorch
class ConditionalNoisePredictor(nn.Module):
    def __init__(self, action_dim, obs_dim, time_emb_dim=32):
        super().__init__()
        # 将标量时间步转换为高维嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 假设我们在网络外部已经将历史观测展平或通过CNN提取为长度为 obs_dim 的一维向量
        # 最终的全局条件向量维度 = 时间嵌入维度 + 观测特征维度
        cond_dim = time_emb_dim + obs_dim
        
        # 定义一维的去噪网络结构
        self.net = nn.ModuleList([
            ConditionalConv1dBlock(action_dim, 64, cond_dim),
            ConditionalConv1dBlock(64, 64, cond_dim),
            nn.Conv1d(64, action_dim, kernel_size=1)
        ])
        
    def forward(self, action_noisy, timestep, obs_cond):
        # 1. 提取连续的时间步嵌入
        t_emb = self.time_mlp(timestep)
        
        # 2. 在特征维度上拼接得到全局条件表示
        global_cond = torch.cat([t_emb, obs_cond], dim=-1)
        
        # 3. 调整 action 张量的形状以适应 1D 卷积的标准输入: (Batch, Sequence, Dim) -> (Batch, Dim, Sequence)
        x = action_noisy.permute(0, 2, 1)
        
        # 4. 依次通过条件调制网络层
        x = self.net[0](x, global_cond)
        x = self.net[1](x, global_cond)
        x = self.net[2](x) # 最后一层仅作维度对齐映射，不再需要注入条件
        
        # 输出前将形状恢复到与动作块对齐: (Batch, Dim, Sequence) -> (Batch, Sequence, Dim)
        return x.permute(0, 2, 1)
```

## 7.10.5 训练与推断循环

现在，我们将环境调度器和噪声预测网络整合在一起，展示扩散策略完整的梯度下降训练步骤与自动回归式的推断采样步骤。

(**实现扩散策略的训练损失计算与逆向动作生成**)

```{.python .input}
#@tab pytorch
def train_step(model, scheduler, action_batch, obs_cond_batch, optimizer):
    model.train()
    optimizer.zero_grad()
    
    batch_size = action_batch.shape[0]
    device = action_batch.device
    
    # 1. 采样与原始动作形状完全一致的随机正态噪声
    noise = torch.randn_like(action_batch)
    
    # 2. 均匀采样批次中每个样本独立的扩散步数 k
    timesteps = torch.randint(
        0, scheduler.num_train_timesteps, 
        (batch_size,), device=device
    ).long()
    
    # 3. 执行前向过程：一步到位获取各个时刻带噪的动作矩阵 A_k
    noisy_actions = scheduler.add_noise(action_batch, noise, timesteps)
    
    # 4. 前向传播：传入带噪数据、时间步与观测条件，预测注入的噪声
    noise_pred = model(noisy_actions, timesteps, obs_cond_batch)
    
    # 5. 计算 MSE 损失并执行反向传播
    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def generate_actions(model, scheduler, obs_cond, action_shape):
    model.eval()
    device = obs_cond.device
    batch_size = obs_cond.shape[0]
    
    # 从没有任何结构信息的纯高斯噪声开始采样 A_K
    action = torch.randn((batch_size, *action_shape), device=device)
    
    # 逐步去噪，逆时间方向：从 num_train_timesteps-1 严格递减到 0
    for k in reversed(range(0, scheduler.num_train_timesteps)):
        timesteps = torch.full((batch_size,), k, device=device, dtype=torch.long)
        
        # 通过网络预测在第 k 步存在于动作序列中的噪声
        noise_pred = model(action, timesteps, obs_cond)
        
        # 提取代数推导中的已知标量常数
        alpha = scheduler.alphas[k].to(device)
        alpha_cumprod = scheduler.alphas_cumprod[k].to(device)
        
        # 严格应用逆向采样公式 (方程 7.10.6) 求取均值部分
        mean = (1.0 / torch.sqrt(alpha)) * (
            action - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod)) * noise_pred
        )
        
        # 恢复随机游走的方差：当且仅当 k > 0 时添加额外的高斯补偿
        if k > 0:
            noise = torch.randn_like(action)
            sigma = torch.sqrt(1.0 - alpha) # 这里采取了简化的方差估计形式
            action = mean + sigma * noise
        else:
            action = mean
            
    return action
```

扩散策略将原本艰难的直接动作生成转化为了一个逐步迭代精化的平缓过程。通过这种基于去噪的条件生成框架，网络无需被迫在互斥的人类行为间做强硬的取舍（从而避免求均值），而是可以精准地拟合和再现高度复杂的、多峰分布的自然操作轨迹。

## 小结

* 扩散策略将机器人的动作序列生成建模为逐步去除高维空间中高斯噪声的随机动力学过程。
* 通过动作块（Action Chunking）技术，模型能够一次性预测未来的多步完整轨迹，从而大幅提升控制过程的时间一致性与执行平滑度。
* 传统的均方误差回归无法处理多模态分布（即布里丹之驴困境），而以 DDPM 为代表的扩散概率模型能够通过随机微分和条件生成，完美覆盖并重建复杂的演示数据分布。
* 一维条件卷积网络通过参数化的特征线性调制（FiLM）层融合历史观测，灵活控制着动作序列在不同去噪时间步下的演化方向。
