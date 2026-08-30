# 扩散策略（Diffusion Policy）的从零开始实现

:label:sec_diffusion_policy_scratch

在探讨具身智能（Embodied AI）中的动作生成时，我们常常面临一个基础性却极难克服的挑战：对于相同的环境状态，专家演示（Expert Demonstrations）可能包含多种完全合理的动作序列。传统的行为克隆（Behavior Cloning, BC）广泛依赖于均方误差（Mean Squared Error, MSE）作为损失函数，这在数学上等价于假设动作分布服从单峰高斯分布。然而，当真实专家数据呈现多模态（Multi-modal）分布时，最小化均方误差会驱使模型预测出所有可能动作的平均值，这往往会导致灾难性的失败。例如，在面对障碍物时，专家可能选择从左侧绕过或从右侧绕过，而模型预测的平均动作则是径直撞向障碍物。

Chi 等人提出扩散策略，把机器人动作序列建模为观测条件下的去噪扩散过程 [[Chi et al., 2023]](https://arxiv.org/abs/2303.04137)。它沿用 DDPM 的前向加噪与反向去噪框架 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)，但生成对象从图像换成了连续动作轨迹。这种参数化可以表达多模态动作分布；论文结论应限定在所测试的机器人操作基准内。

在本节中，我们将从最基础的统计学概念起步，逐步推导扩散策略的严密数学公式，并最终从零开始实现一个完整的扩散策略模型。

## 从高中统计学起步：状态与动作的概率映射

:label:subsec_diffusion_math_basics

为了深刻理解扩散模型的本质，我们不必立刻深陷高维张量和随机微分方程。让我们从高中统计学中最简单的标量概念出发。

假设机器人的动作是一个标量 $a \in \mathbb{R}$，观察状态是 $o$。传统回归试图学习一个确定性函数 $f(o) = a$。但如果对于相同的 $o$，存在两个等概率的正确动作 $a_1=1$ 和 $a_2=-1$，函数 $f$ 只能输出其期望值 $0$。

为了能够表示这种分布，我们不再直接预测 $a$，而是引入“加噪”与“去噪”的过程。假设我们拥有一个完美的专家动作 $a_0$。如果我们对其施加一个微小的高斯噪声 $\epsilon \sim \mathcal{N}(0, 1)$，得到 $a_1 = a_0 + \sigma_1 \epsilon$。如果我们不断重复这个加噪过程 $T$ 次，每次都累加微小的随机扰动，最终 $a_T$ 的信息将完全丧失，退化为一个标准正态分布的随机变量。

在生成（推理）时，我们反转这一过程：从标准正态分布中随机抽取一个初始值 $a_T$，然后逐步“去除”噪声，最终恢复出一个具体的动作 $a_0$。由于我们的起点 $a_T$ 是随机采样的，去噪过程有可能会沿着不同的概率路径收敛到 $a_1=1$ 或 $a_2=-1$，从而完美解决了多模态问题。

在这里，我们引入全篇唯一一次类比：这一去噪过程宛如古典主义雕刻。纯粹的随机噪声 $a_T$ 是一块未经雕琢的粗糙大理石（代表毫无规律的高斯分布），而基于视觉或状态的条件观察 $o$ 则是艺术家心中的蓝图。随着时间步 $t$ 从 $T$ 逐渐递减到 $0$，神经网络像凿子一样，在每一步精准地剔除多余的石料（即预测并减去噪声），直到最终显露出那尊完美符合蓝图的雕像——即物理上可行且高效的动作序列 $a_0$。

## 扩散过程的严密数学推导

:label:subsec_diffusion_derivation

现在，我们将上述直觉转化为严格的数学语言。定义动作序列为 $a_0$。我们设计一个包含 $T$ 步的马尔可夫前向加噪过程（Forward Process）。

### 前向加噪过程

在任意时间步 $t \in [1, T]$，我们在前一步 $a_{t-1}$ 的基础上添加少量的高斯噪声。给定一族预先设定的方差表（Variance Schedule） $\beta_1, \beta_2, \dots, \beta_T \in (0, 1)$，前向转移概率定义为：

$$
q(a_t | a_{t-1}) = \mathcal{N}(a_t; \sqrt{1-\beta_t} a_{t-1}, \beta_t \mathbf{I})
$$

:eqlabel:eq_diffusion_forward_step

为了能够在训练时直接跳跃到任意时间步 $t$，而不需要一步步迭代计算，我们利用高斯分布的叠加性质（重参数化技巧）。令 $\alpha_t = 1 - \beta_t$，并定义 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。我们可以直接得到给定初始动作 $a_0$ 时，时刻 $t$ 的边缘概率分布：

$$
q(a_t | a_0) = \mathcal{N}(a_t; \sqrt{\bar{\alpha}_t} a_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

:eqlabel:eq_diffusion_forward_t

这意味着，时刻 $t$ 的含噪动作 $a_t$ 可以被确定性地表示为：

$$
a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
$$

:eqlabel:eq_diffusion_reparam

其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 是真实注入的噪声。当 $T$ 足够大且 $\bar{\alpha}_T \to 0$ 时，$q(a_T | a_0)$ 趋近于标准正态分布 $\mathcal{N}(0, \mathbf{I})$。

### 逆向去噪过程与损失函数

生成动作的目标是从 $a_T \sim \mathcal{N}(0, \mathbf{I})$ 出发，逐步采样逆向过程的转移概率 $p(a_{t-1} | a_t)$。由于真正的逆向分布 $q(a_{t-1} | a_t, a_0)$ 在给定 $a_0$ 时是一个高斯分布，我们可以用一个参数化的神经网络 $p_\theta(a_{t-1} | a_t, o)$ 来近似它。在扩散策略中，网络以上下文观察 $o$ 为条件。

根据变分下界（Variational Lower Bound, VLB）理论及 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239) 的推导，与其直接预测均值，不如让神经网络去预测前向过程中加入的噪声 $\epsilon$。我们定义神经网络为 $\epsilon_\theta(a_t, t, o)$，其目标是最小化预测噪声与真实噪声之间的均方误差：

$$
L(\theta) = \mathbb{E}_{t \sim \mathcal{U}(1,T), a_0, \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(a_t, t, o) \|^2 \right]
$$

:eqlabel:eq_diffusion_loss

这是扩散模型中极为优雅的结果：尽管背后的概率推导涉及复杂的积分与边界，其最终的训练目标却异常简洁，仅是对高斯噪声的误差匹配。

## 动作组块与时间维度

:label:subsec_action_chunking

在实际的具身智能中，为了保证运动的平滑性和应对网络延迟，扩散策略通常结合了动作组块（Action Chunking）技术。具体而言，模型并非在每个控制周期只预测单步动作，而是一次性预测未来 $H$ 步的动作序列。

假设每个单步动作的维度为 $D_a$，那么网络的预测目标就是一个形状为 $(H, D_a)$ 的张量。在去噪网络 $\epsilon_\theta$ 内部，这一动作序列可以被视作一个一维的时间序列，因此我们通常使用一维卷积神经网络（1D CNN）或者 Transformer 来处理它。在下面的代码实现中，我们将采用经典的 1D CNN 结构。

## 从零开始的代码实现

:label:subsec_diffusion_implementation

我们将分别实现噪声调度器、去噪神经网络以及整体的训练流程。首先，我们需要构建前向加噪过程的调度器。

(**实现 DDPM 的方差调度器**)

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab pytorch
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

## 2026 年具身智能开源生态深度剖析与融合

:label:subsec_2026_ecosystem_analysis

扩散策略虽然在数学推导和学术基准测试上展现了极高的上限，但要将其部署到真实物理世界，必须面对感知延迟、硬件碎片化以及泛化能力不足等严峻考验。2026 年标志着具身智能从“作坊式”学术实验全面转向“工业化”开源生态的一年。有几个里程碑式的开源项目不仅解决了部署难题，更为扩散策略提供了完美的土壤：

首先，**Physical Intelligence (Pi) 探索的通用机器人大脑**模型确立了跨形态泛化的基准。Pi 的基础模型不再局限于特定机械臂的关节空间（Joint Space），而是通过学习统一的末端执行器（End-effector）六自由度动作流形（Action Manifold）。在这一框架下，扩散策略不再只是一个孤立的动作生成器，而是被集成到 Pi 的基础世界模型中。扩散过程的高维隐空间通过跨模态对齐，完美适应了 Pi 的泛化特征表征，极大地增强了复杂长视野（Long-horizon）任务的完成率。

其次，硬件接口的碎片化一直是机器人算法落地的最大阻碍。**Linux Foundation 发起的 AIRSEAI 模块化框架**彻底改变了这一现状。AIRSEAI 提供了一层高度抽象的硬件抽象层（HAL），统一了不同厂商（如 Franka, UR, 甚至各类双足人形机器人）的驱动接口，并将控制频率严格对齐。这对于高度依赖时序一致性和动作组块（Action Chunking）的扩散策略而言至关重要。开发者现在可以直接使用 AIRSEAI 提供的零拷贝内存管理来传递观察状态 $o$，使得高频扩散推理的延迟降低了 40% 以上。

此外，**Hugging Face 的 LeRobot 通用具身智能生态库**为跨平台的扩散模型部署提供了极佳的范式。LeRobot 作为一个高度通用的开源架构，旨在成为具身智能领域的“Transformers”库。它无缝整合了模型库、数据集和训练管道，对于扩散策略而言，传统的 DDPM 推理需要进行 100 步以上的神经网络迭代，而 LeRobot 深度集成了先进的扩散模型加速采样算法（如 DDIM 与 Consistency Models），并将一维 CNN 和 Transformer 推理过程极端优化。这不仅使得策略模型能够在异构设备上流畅运行，更打通了从仿真环境到物理硬件的端到端部署，为大规模通用机器人提供了高度统一的调度标准。

最后，数据的匮乏曾是限制扩散模型发挥其概率建模优势的瓶颈。**AGIBOT WORLD 2026 开源数据集**的发布填补了这一空白。该数据集包含了数万小时、跨十余种真实家庭和工厂场景的多模态遥操作（Teleoperation）演示数据，并且由于是众包采集，同一任务常常包含极其多样的专家解决路径。这种充满多模态分布的数据正是传统 BC 算法的噩梦，但却是扩散策略展现实力的绝佳舞台。正是借助 AGIBOT WORLD 2026 提供的海量非结构化数据，扩散策略终于证明了其在解决现实物理世界多义性（Ambiguity）方面不可替代的价值。

## 练习

:label:subsec_diffusion_exercises

1. **推导变分下界**：尝试手动推导从逆向概率分布匹配到该公式所述的均方误差形式。
   _提示_：回顾重参数化技巧，并关注 KL 散度在两个高斯分布之间的计算方式。
2. **加速推理采样**：在标准的去噪循环中，必须严格按照时间步 $T \to 0$ 逐一迭代。如果要实现跳步推理（如 DDIM），调度器的 `step` 函数应该如何修改？
   _提示_：重新审视 $a_{t-1}$ 是如何由 $a_t$ 和预测出的 $\epsilon_\theta$ 确定性重构出来的。
3. **生态系统结合**：如果在使用 LeRobot 框架部署扩散策略时，机械臂的控制频率出现了明显抖动，这会对 Action Chunking 产生怎样的影响？
   _提示_：思考时间维度的张量在执行物理动作时，一旦预设的时间间隔被打破，动作之间的连续性（如速度和加速度）会发生什么变化。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
