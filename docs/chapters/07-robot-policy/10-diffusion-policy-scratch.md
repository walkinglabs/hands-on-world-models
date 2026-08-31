# 扩散策略（Diffusion Policy）的从零开始实现

把杯子在空中翻转到指定姿态，需要连续调整接触点、腕部方向和夹爪运动。Diffusion Policy 论文在真实机械臂上测试了这类 6DoF 杯子翻转任务；扩散策略不直接输出单个条件均值，而是学习观测条件下的动作序列分布。

<div align="center">

<img src="/figures/07-robot-policy/source/10-diffusion-policy-scratch/dp-fig9.png" alt="真实机械臂完成 6DoF 杯子翻转，显示动作生成必须处理接触与姿态变化。" width="86%">

_图 7.10-1：真实机械臂完成 6DoF 杯子翻转，显示动作生成必须处理接触与姿态变化。 出处：[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，Cheng Chi et al.，2023](https://arxiv.org/abs/2303.04137)。_

</div>

Chi 等人把机器人动作序列建模为观测条件下的去噪扩散过程 [[Chi et al., 2023]](https://arxiv.org/abs/2303.04137)。论文还展示了倒酱和涂抹等接触任务。方法沿用 DDPM 的前向加噪与反向去噪框架 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)，但生成对象从图像换成连续动作轨迹；结论应限定在所测试的操作基准内。

<div align="center">

<img src="/figures/07-robot-policy/source/10-diffusion-policy-scratch/dp-fig10.png" alt="倒酱与涂抹展示同一策略族在两种真实接触任务中的多模态动作。" width="86%">

_图 7.10-2：倒酱与涂抹展示同一策略族在两种真实接触任务中的多模态动作。 出处：[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，Cheng Chi et al.，2023](https://arxiv.org/abs/2303.04137)。_

</div>

第 7.5 节介绍了方法全貌；本节集中实现前向加噪、时间嵌入和一次训练更新，并明确哪些部署环节尚未包含。

## 从高中统计学起步：状态与动作的概率映射

先从标量动作理解为什么要学习条件分布。

假设机器人的动作是一个标量 $a \in \mathbb{R}$，观察状态是 $o$。传统回归试图学习一个确定性函数 $f(o) = a$。但如果对于相同的 $o$，存在两个等概率的正确动作 $a_1=1$ 和 $a_2=-1$，函数 $f$ 只能输出其期望值 $0$。

为了表示这种分布，我们引入“加噪”与“去噪”过程。给专家动作 $a_0$ 加入高斯噪声 $\epsilon \sim \mathcal{N}(0,1)$，可得到含噪变量。按照合适的方差表重复 $T$ 步后，$a_T$ 的分布会接近标准正态分布；有限步数下的接近程度由 $\bar\alpha_T$ 决定。

推理时从标准正态分布采样 $a_T$，再逐步估计并去除噪声。不同初始样本可能落到不同动作模态，因此模型能够表示多模态分布，但不保证每个模态都被正确学到。

每个反向步骤都利用当前含噪动作、扩散时间步和观测条件估计噪声，再按采样器更新动作。这个概率更新本身不保证物理可行；还需要训练数据覆盖、动作限幅和执行期安全检查。

## 扩散过程的严密数学推导

现在，我们将上述直觉转化为严格的数学语言。定义动作序列为 $a_0$。我们设计一个包含 $T$ 步的马尔可夫前向加噪过程（Forward Process）。

<div align="center">

<img src="/figures/07-robot-policy/source/10-diffusion-policy-scratch/ddpm-fig6.png" alt="DDPM 的渐进生成序列直观展示从噪声到结构的逐步恢复。" width="86%">

_图 7.10-3：DDPM 的渐进生成序列直观展示从噪声到结构的逐步恢复。 出处：[Denoising Diffusion Probabilistic Models，Jonathan Ho; Ajay Jain; Pieter Abbeel，2020](https://arxiv.org/abs/2006.11239)。_

</div>

### 前向加噪过程

在任意时间步 $t \in [1, T]$，我们在前一步 $a_{t-1}$ 的基础上添加少量的高斯噪声。给定一族预先设定的方差表（Variance Schedule） $\beta_1, \beta_2, \dots, \beta_T \in (0, 1)$，前向转移概率定义为：

$$
q(a_t | a_{t-1}) = \mathcal{N}(a_t; \sqrt{1-\beta_t} a_{t-1}, \beta_t \mathbf{I})
$$

为了能够在训练时直接跳跃到任意时间步 $t$，而不需要一步步迭代计算，我们利用高斯分布的叠加性质（重参数化技巧）。令 $\alpha_t = 1 - \beta_t$，并定义 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。我们可以直接得到给定初始动作 $a_0$ 时，时刻 $t$ 的边缘概率分布：

$$
q(a_t | a_0) = \mathcal{N}(a_t; \sqrt{\bar{\alpha}_t} a_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

这意味着，时刻 $t$ 的含噪动作 $a_t$ 可以被确定性地表示为：

$$
a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
$$

其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 是真实注入的噪声。当 $T$ 足够大且 $\bar{\alpha}_T \to 0$ 时，$q(a_T | a_0)$ 趋近于标准正态分布 $\mathcal{N}(0, \mathbf{I})$。

### 逆向去噪过程与损失函数

生成动作的目标是从 $a_T \sim \mathcal{N}(0, \mathbf{I})$ 出发，逐步采样逆向过程的转移概率 $p(a_{t-1} | a_t)$。由于真正的逆向分布 $q(a_{t-1} | a_t, a_0)$ 在给定 $a_0$ 时是一个高斯分布，我们可以用一个参数化的神经网络 $p_\theta(a_{t-1} | a_t, o)$ 来近似它。在扩散策略中，网络以上下文观察 $o$ 为条件。

根据变分下界（Variational Lower Bound, VLB）理论及 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239) 的推导，与其直接预测均值，不如让神经网络去预测前向过程中加入的噪声 $\epsilon$。我们定义神经网络为 $\epsilon_\theta(a_t, t, o)$，其目标是最小化预测噪声与真实噪声之间的均方误差：

$$
L(\theta) = \mathbb{E}_{t \sim \mathcal{U}(1,T), a_0, \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(a_t, t, o) \|^2 \right]
$$

因此，常用训练目标可以写成噪声预测的均方误差。它对应 DDPM 变分目标的一种加权简化形式。

## 动作组块与时间维度

<div align="center">

<img src="/figures/07-robot-policy/source/10-diffusion-policy-scratch/dp-fig3.png" alt="不同策略表示的行为轨迹对比说明扩散策略可表达多峰动作分布。" width="86%">

_图 7.10-4：不同策略表示的行为轨迹对比说明扩散策略可表达多峰动作分布。 出处：[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，Cheng Chi et al.，2023](https://arxiv.org/abs/2303.04137)。_

</div>

扩散策略通常一次生成未来 $H$ 步动作，并以滚动时域方式执行其中一部分后重新观测。动作块有利于建模局部时间结构，但平滑性和延迟容忍度仍取决于训练数据与执行策略。

假设每个单步动作的维度为 $D_a$，那么网络的预测目标就是一个形状为 $(H, D_a)$ 的张量。

在去噪网络 $\epsilon_\theta$ 内部，这一动作序列可以被视作一个一维的时间序列，因此我们通常使用一维卷积神经网络（1D CNN）或者 Transformer 来处理它。在下面的代码实现中，我们将采用经典的 1D CNN 结构。

<div align="center">

<img src="/figures/07-robot-policy/latex/10-diffusion-policy-scratch/two-time-axes.png" alt="扩散步与动作块内物理未来步是两条不同时间轴" width="86%">

_图 7.10-5：扩散步 t 表示噪声层级；每个 t 上都有完整 H×D_a 动作块，其中 h 才是机器人未来的物理时间索引。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

## 从零开始的代码实现

我们将分别实现噪声调度器、去噪神经网络以及整体的训练流程。首先，我们需要构建前向加噪过程的调度器。

(**实现 DDPM 的方差调度器**)

```python
import torch
from torch import nn
import math

class DDPMScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=1e-4, beta_end=2e-2):
        self.num_train_timesteps = num_train_timesteps

        # 线性方差表 (Linear Variance Schedule)
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """
        基于该公式实现前向加噪
        original_samples: 形状为 (B, H, D_a) 的初始动作 a_0
        noise: 从 N(0, I) 采样的同形状噪声
        timesteps: 形状为 (B,) 的整数时间步
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1) # 调整形状以支持广播

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
```

接下来，我们需要实现去噪网络。在这里，我们实现一个精简版的带有正弦位置编码的一维卷积残差网络。条件观察 $o$ 将作为额外的特征通道注入到网络中。

(**实现条件一维卷积去噪网络**)

```python
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

class Conditional1DCNN(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 将含噪动作和观察条件在通道维度拼接
        self.conv_in = nn.Conv1d(action_dim + obs_dim, hidden_dim, kernel_size=3, padding=1)

        # 简化的残差块
        self.block1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.block2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.conv_out = nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, noisy_action, timestep, obs):
        """
        noisy_action: (B, H, action_dim)
        timestep: (B,)
        obs: (B, obs_dim) -> 我们将其扩展到整个时间序列以便拼接
        """
        B, H, _ = noisy_action.shape

        # 提取时间嵌入
        t_emb = self.time_mlp(timestep) # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(-1)     # (B, hidden_dim, 1)

        # 扩展观察条件以匹配动作序列的时间步长
        # (B, obs_dim) -> (B, obs_dim, H)
        obs_expanded = obs.unsqueeze(-1).expand(-1, -1, H)

        # 转换动作张量的形状以适应 Conv1d: (B, action_dim, H)
        x = noisy_action.transpose(1, 2)

        # 拼接并投影
        x = torch.cat([x, obs_expanded], dim=1)
        x = self.conv_in(x)

        # 加入时间嵌入与残差计算
        res = x
        x = self.act(self.block1(x + t_emb))
        x = self.block2(x) + res

        # 输出预测的噪声
        out = self.conv_out(x)
        return out.transpose(1, 2) # 恢复为 (B, H, action_dim)
```

有了调度器和网络，我们可以清晰地构建出该公式描述的训练流程。

(**实现训练循环**)

```python
def train_diffusion_policy():
    # 超参数定义
    B, H, action_dim, obs_dim = 32, 16, 2, 64
    num_timesteps = 100

    # 模拟专家数据 (B, H, action_dim) 和 图像/状态观察特征 (B, obs_dim)
    expert_actions = torch.randn(B, H, action_dim)
    observations = torch.randn(B, obs_dim)

    # 实例化组件
    model = Conditional1DCNN(action_dim, obs_dim)
    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟一次训练迭代
    optimizer.zero_grad()

    # 1. 随机采样纯高斯噪声，形状必须与动作序列完全一致
    noise = torch.randn_like(expert_actions)

    # 2. 为每个 Batch 独立地采样一个随机时间步 t
    timesteps = torch.randint(0, num_timesteps, (B,), device=expert_actions.device).long()

    # 3. 按照方差表计算前向加噪结果 a_t
    noisy_actions = scheduler.add_noise(expert_actions, noise, timesteps)

    # 4. 神经网络以上下文观察 o 和时间步 t 为条件，预测注入的噪声
    noise_pred = model(noisy_actions, timesteps, observations)

    # 5. 计算 MSE 损失并更新参数
    loss = nn.functional.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()

    return loss.item()

print(f"Training Loss: {train_diffusion_policy():.4f}")
```

## 实现边界与部署检查

上面的程序只完成一次训练更新，还不是可执行的机器人策略。完整推理至少要从高斯噪声初始化动作块，并实现与训练参数化一致的反向采样器。若采用 DDPM、DDIM 或其他求解器，更新公式和随机项也会不同。

接到真实机器人之前，还需要逐项检查：观测与动作是否使用训练时相同的归一化统计量；动作单位和坐标系是否一致；采样耗时能否满足控制周期；每次重规划执行动作块中的多少步；位置、速度、加速度与碰撞约束如何落实；相机延迟和丢帧如何处理。离线噪声损失下降不能替代闭环回放、仿真压力测试和小速度实机验证。

## 练习

1. **推导变分下界**：尝试手动推导从逆向概率分布匹配到该公式所述的均方误差形式。
   _提示_：回顾重参数化技巧，并关注 KL 散度在两个高斯分布之间的计算方式。
2. **加速推理采样**：在标准的去噪循环中，必须严格按照时间步 $T \to 0$ 逐一迭代。如果要实现跳步推理（如 DDIM），调度器的 `step` 函数应该如何修改？
   _提示_：重新审视 $a_{t-1}$ 是如何由 $a_t$ 和预测出的 $\epsilon_\theta$ 确定性重构出来的。
3. **控制频率检查**：如果部署时机械臂的控制周期明显抖动，这会怎样改变动作块中相邻动作的实际时间间隔？
   _提示_：思考时间维度的张量在执行物理动作时，一旦预设的时间间隔被打破，动作之间的连续性（如速度和加速度）会发生什么变化。
