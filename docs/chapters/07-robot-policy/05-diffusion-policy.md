# 7.5 扩散策略理论与连续动作建模 (Diffusion Policy)

在上一节中，我们见证了传统行为克隆（BC）在面对误差累积与协变量偏移时的局限。然而，传统策略在物理世界中面临的另一个更为严峻的挑战是——**人类专家动作的多模态（Multimodality）与高维连续性**。

当机械臂面对工作台上的一个物体时，人类专家既可以从左侧推，也可以从右侧推；当机器人端着装满水的杯子移动时，关节的加速度与加加速度（Jerk）必须保持极高的连续性，否则微小的顿挫就会使水花剧烈洒出。

2023 年，由哥伦比亚大学 Cheng Chi、斯坦福大学 Shuran Song 等人提出的 **Diffusion Policy（扩散策略）** 彻底重塑了机器人动作生成的范式。它将生成式扩散模型（Diffusion Models）引入机器人连续视觉运动控制，使机器人能够从随机噪声中直接“雕刻”出高精度、多模态、时间平滑的动作轨迹。

<div align="center">

<img src="/figures/07-robot-policy/source/05-diffusion-policy/dp-fig7.png" alt="真实 Push-T 轨迹显示扩散策略如何保留多条合理动作路径。" width="86%">

_图 7.5-1：真实 Push-T 轨迹显示扩散策略如何保留多条合理动作路径。 出处：[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，Cheng Chi et al.，2023](https://arxiv.org/abs/2303.04137)。_

</div>

---

## 7.5.1 物理与数学基石：从确定性回归到连续概率动力学流

要理解扩散策略的突破，我们首先需要审视传统概率建模方法在表达复杂物理动作时的局限。

### 1. 经典多模态建模的困境：混合高斯模型（GMM）
在扩散模型问世之前，机器人学者通常使用**混合高斯模型（Gaussian Mixture Models, GMM）**来表达多模态分布：

$$p(\mathbf{a} \mid \mathbf{s}) = \sum_{k=1}^K \pi_k(\mathbf{s}) \mathcal{N}\left(\mathbf{a}; \boldsymbol{\mu}_k(\mathbf{s}), \boldsymbol{\Sigma}_k(\mathbf{s})\right)$$

然而，GMM 在高维物理控制中存在致命的缺陷：
1. **模式坍塌与数量固化**：超参数 $K$（高斯分量个数）必须人为预先设定。如果设得太小，无法覆盖复杂的长程操作；设得太大，极大似然估计极易遭遇奇异性数值崩溃（某一分量方差趋于 0）；
2. **高维协方差拟合困难**：当控制自由度较高时，高维全协方差矩阵的参数量呈平方级爆炸，导致训练极度不稳定。

### 2. 扩散策略的物理视角：能量流形上的朗之万去噪动力学
扩散策略将动作生成视为物理学中的**朗之万动力学（Langevin Dynamics）**过程：
- 策略不再直接预测动作的具体数值，而是预测整个动作概率分布流形上的**对数概率密度梯度（称为得分函数 Score Function，$\nabla_{\mathbf{a}} \log p(\mathbf{a} \mid \mathbf{o})$）**；
- 机器人首先从任意的纯白噪声中随机采样一个初始点，随后沿着得分函数指示的梯度流，一步一步“滑向”概率密度最高的专家动作流形。

<div align="center">

<img src="/figures/07-robot-policy/source/05-diffusion-policy/ddpm-fig2.png" alt="DDPM 图模型并列前向加噪与逆向生成链，给出动作扩散的概率基础。" width="86%">

_图 7.5-2：DDPM 图模型并列前向加噪与逆向生成链，给出动作扩散的概率基础。 出处：[Denoising Diffusion Probabilistic Models，Jonathan Ho et al.，2020](https://arxiv.org/abs/2006.11239)。_

</div>

---

## 7.5.2 核心数学推导一：DDPM 逆向采样单步递推

在推理执行时，机器人以观测 $\mathbf{o}$ 为条件，从完全随机的标准高斯噪声 $\mathbf{a}_K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，执行 $K$ 步离散逆向去噪。

<div align="center">

<img src="/figures/07-robot-policy/latex/05-diffusion-policy/reverse-denoise-one-step.png" alt="一次条件逆扩散更新的去噪、缩放与随机采样顺序" width="86%">

_图 7.5-3：一次条件逆扩散更新的去噪、缩放与随机采样顺序。_

</div>

### 1. 逆向均值与方差修正方程
在第 $k$ 个去噪步（从 $k = K$ 逐步递减至 $k = 1$），网络 $\boldsymbol{\epsilon}_\theta(\mathbf{a}_k, k, \mathbf{o})$ 预测出当前包含的噪声量。去除噪声后的前一步动作 $\mathbf{a}_{k-1}$ 满足解析递推公式：

$$\mathbf{a}_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{a}_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} \boldsymbol{\epsilon}_\theta(\mathbf{a}_k, k, \mathbf{o}) \right) + \sigma_k \mathbf{z}$$

其中：
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（当 $k > 1$ 时注入高斯随机扰动以维持探索，当 $k = 1$ 时 $\sigma_1 = 0$ 确定性收敛）；
- 扰动标准差 $\sigma_k = \sqrt{\tilde{\beta}_k} = \sqrt{\frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k} \beta_k}$。

**手算代入算例**：
设在某一步 $k$，方差系数 $\beta_k = 0.05, \alpha_k = 0.95, \bar{\alpha}_k = 0.64, \bar{\alpha}_{k-1} = 0.6737$。假定当前含噪动作 $a_k = 0.50$，网络预测的噪声为 $\epsilon_\theta = 0.20$。

1. 计算缩放分母：$\sqrt{\alpha_k} = \sqrt{0.95} \approx 0.9747$；
2. 计算噪声扣除权重：$\frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} = \frac{0.05}{\sqrt{0.36}} = \frac{0.05}{0.60} \approx 0.0833$；
3. 计算去噪后的预测均值：
   $$\mu_{k-1} = \frac{1}{0.9747} \left( 0.50 - 0.0833 \times 0.20 \right) = \frac{0.50 - 0.0167}{0.9747} = \frac{0.4833}{0.9747} \approx 0.4958$$

通过初等代数的几步加减乘除，模型极其精准地从噪声中滤除了一层杂质，使动作向真实专家流形迈进了一步！

<details>
<summary><b>深入推导：得分匹配（Score Matching）理论与随机微分方程（SDE）逆向时间流（点击展开查看完整推导）</b></summary>

根据 Song & Ermon (NeurIPS 2019) 的连续时间随机微分方程理论，前向加噪过程对应伊藤随机微分方程：
$$d\mathbf{a} = f(\mathbf{a}, t) dt + g(t) d\mathbf{w}$$
根据 Anderson (1982) 定理，逆向时间 SDE 具有精确的封闭形式：
$$d\mathbf{a} = \left[ f(\mathbf{a}, t) - g(t)^2 \nabla_{\mathbf{a}} \log p_t(\mathbf{a} \mid \mathbf{o}) \right] dt + g(t) d\bar{\mathbf{w}}$$
网络预测的噪声 $\boldsymbol{\epsilon}_\theta$ 与流形得分函数直接满足严格代数反比关系：
$$\nabla_{\mathbf{a}} \log p_t(\mathbf{a} \mid \mathbf{o}) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{a}, t, \mathbf{o})}{\sqrt{1 - \bar{\alpha}_t}}$$
这证明了 DDPM 的逆向去噪过程在数学上完全等价于沿着能量流形梯度的确定性积分与随机朗之万扰动。
</details>

---

## 7.5.3 核心数学推导二：滚动时域执行（Receding Horizon Planning）

在工业机器人与自动驾驶中，直接单步执行预测动作很容易因通信延迟或高频抖动导致运动不平稳。

<div align="center">

<img src="/figures/07-robot-policy/source/05-diffusion-policy/dp-fig2.png" alt="Diffusion Policy 以观测为条件，对整段含噪动作序列反复去噪并滚动执行。" width="86%">

_图 7.5-4：Diffusion Policy 以观测为条件，对整段含噪动作序列反复去噪并滚动执行。 出处：[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，Cheng Chi et al.，2023](https://arxiv.org/abs/2303.04137)。_

</div>

Diffusion Policy 借鉴了经典控制理论中的**模型预测控制（Model Predictive Control, MPC）**与**滚动时域（Receding Horizon）**机制：
1. **长程预测时域（Prediction Horizon, $T_p = 16$）**：去噪网络在每一轮推理中，一次性生成未来 16 个时间步的完整动作轨迹 $\mathbf{A} = [\mathbf{a}_t, \mathbf{a}_{t+1}, \dots, \mathbf{a}_{t+15}]$；
2. **短程执行时域（Execution Horizon, $T_a = 8$）**：物理机械臂仅连续执行前 8 个时间步的动作；
3. **滚动重新规划**：在执行完第 8 步时，相机获取最新的工作台图像观测 $\mathbf{o}_{t+8}$，重新启动扩散去噪，生成全新的 16 步轨迹并继续执行前 8 步。

这种“预测看远、执行留余量、滚动刷新”的设计，既保证了单次动作块内部物理速度的高阶平滑性，又使机器人具备了应对突发物理扰动的实时抗干扰能力。

---

## 7.5.4 纯底层 PyTorch 代码实现：条件扩散策略完整逆向采样引擎

下面我们使用纯底层 PyTorch 算子实现完整的条件扩散去噪网络、逆向采样引擎与滚动执行验证。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """
    正弦时间步位置编码层
    将标量时间步 t 映射为高维连续特征向量
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Conditional1DCNN(nn.Module):
    """
    基于 1D 卷积残差结构的条件去噪网络
    """
    def __init__(self, action_dim: int = 2, obs_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        # 时间步 MLP 编码器
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 融合卷积输入层 (通道维度拼接动作与观测条件)
        self.conv_in = nn.Conv1d(action_dim + obs_dim, hidden_dim, kernel_size=3, padding=1)

        # 残差卷积块
        self.block1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.block2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # 输出投影层 (还原回动作维度)
        self.conv_out = nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(
        self, noisy_action: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor
    ) -> torch.Tensor:
        """
        前向去噪预测
        :param noisy_action: (B, H, action_dim) 含噪动作块
        :param timestep: (B,) 时间步
        :param obs: (B, obs_dim) 观测特征
        :return: (B, H, action_dim) 预测的噪声张量
        """
        B, H, _ = noisy_action.shape

        # 1. 编码时间步特征: (B, hidden_dim, 1)
        t_emb = self.time_mlp(timestep).unsqueeze(-1)

        # 2. 沿时间维度展开观测条件: (B, obs_dim, H)
        obs_expanded = obs.unsqueeze(-1).expand(-1, -1, H)

        # 3. 调整形状适配 Conv1d: (B, action_dim, H)
        x = noisy_action.transpose(1, 2)

        # 4. 通道拼接进入网络
        x = torch.cat([x, obs_expanded], dim=1)
        x = self.conv_in(x)

        # 5. 残差卷积与时间特征注入
        res = x
        x = self.act(self.block1(x + t_emb))
        x = self.block2(x) + res

        # 6. 输出预测噪声并还原形状: (B, H, action_dim)
        out = self.conv_out(x)
        return out.transpose(1, 2)

class DiffusionPolicySampler:
    """
    Diffusion Policy 条件逆向去噪采样引擎
    """
    def __init__(self, model: nn.Module, num_timesteps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.model = model
        self.num_timesteps = num_timesteps

        # 预先计算扩散常数表
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 采样方差系数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        以观测为条件，从纯白噪声逐步逆向去噪为动作轨迹
        :param obs: (B, obs_dim)
        :param shape: (B, horizon, action_dim)
        :return: (B, horizon, action_dim)
        """
        device = obs.device
        # 1. 从标准正态分布采样初始混沌噪声 a_T
        a_t = torch.randn(shape, device=device)

        # 2. 从 T-1 逆序倒推到 0
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # 预测当前噪声
            noise_pred = self.model(a_t, t_tensor, obs)

            # 提取常数标量
            alpha_t = self.alphas[t].to(device)
            alpha_bar_t = self.alphas_cumprod[t].to(device)
            beta_t = self.betas[t].to(device)

            # 计算逆向去噪均值
            mean = (1.0 / (alpha_t ** 0.5)) * (
                a_t - (beta_t / ((1.0 - alpha_bar_t) ** 0.5)) * noise_pred
            )

            # 注入随机扰动 (最后一步 t=0 时不注入噪声)
            if t > 0:
                var = self.posterior_variance[t].to(device)
                z = torch.randn_like(a_t)
                a_t = mean + (var ** 0.5) * z
            else:
                a_t = mean

        return a_t

# ===================================================================
# 单元测试：从零模拟逆向去噪采样全流程
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    horizon = 16
    action_dim = 2
    obs_dim = 64
    num_timesteps = 20

    # 实例化去噪网络与采样器
    denoise_net = Conditional1DCNN(action_dim=action_dim, obs_dim=obs_dim, hidden_dim=64)
    denoise_net.eval()

    sampler = DiffusionPolicySampler(model=denoise_net, num_timesteps=num_timesteps)

    dummy_obs = torch.randn(batch_size, obs_dim)
    action_shape = (batch_size, horizon, action_dim)

    sampled_actions = sampler.sample(dummy_obs, action_shape)

    print(f"[Sampler Test] 逆向采样步数: {num_timesteps}")
    print(f"[Sampler Test] 生成动作块形状: {sampled_actions.shape}")
    print(f"[Sampler Test] 生成动作均值与方差: {sampled_actions.mean().item():.4f}, {sampled_actions.var().item():.4f}")

    assert sampled_actions.shape == action_shape, "采样动作张量形状不符！"
    assert not torch.isnan(sampled_actions).any(), "采样出现 NaN！"
    print("✓ 扩散策略条件逆向去噪采样引擎单测全部通过！")
```

---

## 7.5.5 本节小结

回顾本节内容，我们建立了扩散策略在连续动作控制中的完整理论基础：
1. **多模态物理本质**：通过学习能量流形上的得分函数，扩散模型从根本上克服了传统回归在多分支轨迹下的平均化缺陷；
2. **逆向朗之万去噪**：从纯白噪声出发，利用解析递推公式在 $K$ 步内逐步剔除扰动，恢复平滑物理动作；
3. **滚动时域执行**：结合预测时域 $T_p$ 与执行时域 $T_a$，在保证动作长程平滑的同时赋予机器人毫秒级的抗干扰动态重规划能力。
