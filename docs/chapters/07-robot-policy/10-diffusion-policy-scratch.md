# 7.10 从零实现扩散策略 (Diffusion Policy from Scratch)

在前面的章节中，我们探讨了动作分块（ACT）与扩散策略（Diffusion Policy）的高层设计思想。与传统的确定性回归策略不同，扩散策略将机器人复杂的多模态动作建模为一个“从随机噪声中逐步剔除杂质、还原清晰物理轨迹”的逆向生成过程。

然而，对于初学者而言，扩散模型往往被笼罩在诸如“随机微分方程”、“得分匹配”、“变分下界”等复杂的数学符号之中。

在本节中，我们将回归经典力学与初等代数的直观几何视角，地毯式拆解去噪扩散概率模型（DDPM）的核心递推方程，并使用纯底层 PyTorch 算子从零手写一个完整的 1D CNN 动作扩散策略引擎。

<div align="center">

<img src="/figures/07-robot-policy/source/10-diffusion-policy-scratch/chi-fig2.png" alt="Diffusion Policy 用去噪网络把高斯噪声逐步迭代为动作轨迹。" width="86%">

_图 7.10-1：Diffusion Policy 用去噪网络把高斯噪声逐步迭代为动作轨迹。 出处：[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，Cheng Chi et al.，2023](https://arxiv.org/abs/2303.04137)。_

</div>

---

## 7.10.1 物理与数学基石：非平衡热力学扩散与多模态动作建模

要理解扩散策略的本质，我们首先必须回到经典非平衡统计力学与概率建模的起点。

### 1. 经典物理力学中的扩散现象与时间反演
在经典热力学中，如果我们把一滴蓝色的墨水轻轻滴入一碗静止的清水中，水分子永不停歇的微观无规则碰撞（布朗运动）会促使墨水分子自发向周围扩散。随着时间的推移，原本轮廓分明、高浓度的液滴逐渐弥散为均匀分布的微观混沌状态。

根据热力学第二定律，在自然孤立系统中，这一前向扩散过程是单向不可逆的（熵增）。

然而，数学家与计算机科学家提出了一个极富创造力的构想：**如果我们能够在每一个微小的时间间隔内，精确记录下水分子的微观扰动速度，我们是否可以训练一个神经网络逆转时间箭头，从一碗浑浊均匀的随机噪声中，一步一步把墨水分子‘重新凝聚’为清晰有序的初始液滴？**

这一逆转时间箭头的数学模型，正是现代**生成式去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）**。

<div align="center">

<img src="/figures/07-robot-policy/latex/10-diffusion-policy-scratch/ddpm-forward-reverse-chain.png" alt="前向加噪过程与逆向去噪生成链" width="86%">

_图 7.10-2：前向加噪过程逐步注入微小高斯扰动；逆向去噪生成链由神经网络预测噪声，逐步恢复清晰动作序列。本文绘制；TikZ/LaTeX 编译。_

</div>

### 2. 为什么机器人模仿学习必须依赖扩散模型？
在传统的行为克隆（BC）中，网络通常使用均方误差损失（MSE Loss）来最小化预测动作与专家动作之间的欧氏距离。

但在复杂的物理操作中，专家的决策往往是**多模态（Multimodal）**的：
- 例如前方有一个障碍物，专家既可以选择“从左侧绕过去”，也可以选择“从右侧绕过去”。

如果强制使用单峰的确定性网络计算 MSE 损失，网络为了同时讨好左边和右边的标签，最终只能输出两者的算术平均值——**直接笔直撞向正中央的障碍物**！

扩散策略通过将动作生成建模为一个多步连续去噪的随机动力学过程，天然支持复杂的任意多峰概率分布，从而彻底攻克了传统模仿学习在多分支任务下的“平均化失灵”缺陷。

---

## 7.10.2 核心数学推导一：前向加噪过程与方差调度封闭解

在前向加噪过程中，我们从真实的专家动作序列 $\mathbf{a}_0$ 出发，在 $T$ 个离散的时间步内，连续向其注入微小的高斯随机白噪声。

<div align="center">

<img src="/figures/07-robot-policy/source/10-diffusion-policy-scratch/ho-fig2.png" alt="DDPM 的前向加噪与逆向去噪马尔可夫链示意图。" width="86%">

_图 7.10-3：DDPM 的前向加噪与逆向去噪马尔可夫链示意图。 出处：[Denoising Diffusion Probabilistic Models，Jonathan Ho et al.，2020](https://arxiv.org/abs/2006.11239)。_

</div>

### 1. 单步加噪与方差衰减
设在第 $t$ 步注入的噪声方差比例为 $\beta_t \in (0, 1)$（例如 $\beta_t$ 从第 1 步的 $0.0001$ 线性递增到第 $T$ 步的 $0.02$）。
单步加噪的物理递推关系为：

$$\mathbf{a}_t = \sqrt{1 - \beta_t} \mathbf{a}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}, \quad \text{其中 } \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

> **初等代数直觉**：
> 观察前面的两个系数：$(\sqrt{1 - \beta_t})^2 + (\sqrt{\beta_t})^2 = (1 - \beta_t) + \beta_t = 1$。
> 就像平面几何中直角三角形两条直角边的平方和等于斜边平方一样，系数平方和恒等于 1 保证了数据在加噪过程中的**总能量（方差）始终维持在单位尺度**，不会无限膨胀发散。

### 2. 跨越百步的单步封闭解（Closed-form Formula）
如果我们想计算第 $t = 50$ 步加噪后的动作 $\mathbf{a}_{50}$，我们是否需要像串糖葫芦一样把前 49 步依次循环计算一遍？

不需要！定义中间缩放因子 $\alpha_t = 1 - \beta_t$，以及自第 1 步到第 $t$ 步的**累乘系数** $\bar{\alpha}_t$：

$$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i = \alpha_1 \times \alpha_2 \times \dots \times \alpha_t$$

利用独立正态分布相加的方差叠加性质，我们可以将整个 $t$ 步加噪过程浓缩为一个极其优美的**单步瞬时计算公式**：

$$\mathbf{a}_t = \sqrt{\bar{\alpha}_t} \mathbf{a}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

> **公式符号逐一拆解**：
> - $\mathbf{a}_0$：原始完全纯净的专家动作轨迹；
> - $\bar{\alpha}_t \in (0, 1)$：当前时间步保留的纯净信号能量占比（随着 $t$ 增大，$\bar{\alpha}_t$ 单调衰减趋近于 0）；
> - $\boldsymbol{\epsilon}$：直接从标准正态分布中采样出的单次等效总噪声；
> - $\mathbf{a}_t$：在时间步 $t$ 得到的含噪动作。

<div align="center">

<img src="/figures/07-robot-policy/latex/10-diffusion-policy-scratch/noise-residual-triangle.png" alt="含噪动作是原始动作与标准高斯噪声的正交矢量合成" width="86%">

_图 7.10-4：含噪动作是原始动作与标准高斯噪声的正交矢量合成；系数平方和恒等于 1。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

**手算代入算例**：
设机械臂某关节初始动作标量为 $a_0 = 0.50$，采样的标准高斯噪声为 $\epsilon = -0.20$。假定在第 $t = 40$ 步时，系统预先计算好的方差累乘系数为 $\bar{\alpha}_t = 0.64$。

1. 计算纯净信号权重系数：
   $$\sqrt{\bar{\alpha}_t} = \sqrt{0.64} = 0.80$$
2. 计算噪声成分权重系数：
   $$\sqrt{1 - \bar{\alpha}_t} = \sqrt{1 - 0.64} = \sqrt{0.36} = 0.60$$
3. 计算该时刻加噪后的动作数值：
   $$a_t = 0.80 \times 0.50 + 0.60 \times (-0.20) = 0.40 - 0.12 = 0.28$$

这个手算结果生动地展现了：任何时间步 $t$ 的含噪动作，都可以直接用初等代数一步算清，无需任何循环迭代！

<details>
<summary><b>深入推导：前向高斯马尔可夫链独立随机变量线性叠加方差累乘性质的数学归纳证明（点击展开查看完整推导）</b></summary>

对前向递推式展开两步：
$$\mathbf{a}_t = \sqrt{\alpha_t}\mathbf{a}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}$$
$$\mathbf{a}_{t-1} = \sqrt{\alpha_{t-1}}\mathbf{a}_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}$$
将 $\mathbf{a}_{t-1}$ 代入 $\mathbf{a}_t$：
$$\mathbf{a}_t = \sqrt{\alpha_t \alpha_{t-1}}\mathbf{a}_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})}\boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}$$
由于 $\boldsymbol{\epsilon}_{t-2}, \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 为两个互相独立的标准高斯分布，根据正态分布线性组合定理：
$$X \sim \mathcal{N}(\mathbf{0}, \sigma_1^2 \mathbf{I}), Y \sim \mathcal{N}(\mathbf{0}, \sigma_2^2 \mathbf{I}) \implies X + Y \sim \mathcal{N}\left(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I}\right)$$
合成后的随机扰动方差为：
$$\sigma_{\text{merged}}^2 = \left(\sqrt{\alpha_t(1 - \alpha_{t-1})}\right)^2 + \left(\sqrt{1 - \alpha_t}\right)^2 = \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t \alpha_{t-1}$$
因此合成噪声可等价表示为单个标准正态变量 $\sqrt{1 - \alpha_t \alpha_{t-1}}\bar{\boldsymbol{\epsilon}}$。
依数学归纳法递推 $t$ 步，即可完全证明：
$$\mathbf{a}_t = \sqrt{\bar{\alpha}_t}\mathbf{a}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$$
</details>

---

## 7.10.3 核心数学推导二：逆向去噪学习与训练目标简化

在逆向去噪阶段，我们的目标是训练一个神经网络 $\boldsymbol{\epsilon}_\theta$。

我们向网络输入三个信息：
1. 当前受到污染的含噪动作 $\mathbf{a}_t$；
2. 当前所处的噪声时间步 $t$（告诉网络当前污染有多严重）；
3. 机器人当前眼前的相机图像与状态观测 $\mathbf{o}$。

网络的目标极其明确：**精准猜出当时到底向 $\mathbf{a}_0$ 里面掺入了多少噪声 $\boldsymbol{\epsilon}$！**

系统的均方误差训练损失函数为：

$$\mathcal{L}_{\text{Diffusion}}(\theta) = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{o}) - \boldsymbol{\epsilon}\|_2^2\right]$$

一旦网络学会了准确预测噪声 $\boldsymbol{\epsilon}_\theta$，在推理执行时，我们就可以从一段纯白噪声 $\mathbf{a}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，反复利用去噪公式一步步减去预测出的噪声杂质，最终雕刻出极其平滑精准的专家级物理动作。

<details>
<summary><b>深入推导：贝叶斯后验均值表达式与变分下界（ELBO）简化证明（点击展开查看完整推导）</b></summary>

在变分自编码器（VAE）与 DDPM 理论中，真实的逆向转移概率 $q(\mathbf{a}_{t-1} \mid \mathbf{a}_t, \mathbf{a}_0)$ 服从条件高斯分布：
$$q(\mathbf{a}_{t-1} \mid \mathbf{a}_t, \mathbf{a}_0) = \mathcal{N}\left(\mathbf{a}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{a}_t, \mathbf{a}_0), \tilde{\beta}_t \mathbf{I}\right)$$
由贝叶斯公式与高斯分布乘积展开可得后验分布均值为：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{a}_t, \mathbf{a}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\mathbf{a}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{a}_t$$
将前向封闭解 $\mathbf{a}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{a}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon})$ 代入上式消去 $\mathbf{a}_0$：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{a}_t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{a}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}\right)$$
当使用参数化网络 $\boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t)$ 替代真实噪声 $\boldsymbol{\epsilon}$ 时，两个高斯分布之间的 KL 散度严格等价于预测噪声与真实噪声之间的加权均方误差。Jonathan Ho 等人证明，将加权系数简化为常数 1 后的未加权 MSE 损失，在视觉与具身控制中具有最优异的样本生成质量。
</details>

---

## 7.10.4 动作分块（Action Chunking）与两条独立时间轴的解耦

在实现扩散策略时，初学者最容易混淆的是**两条截然不同的“时间轴”**。

<div align="center">

<img src="/figures/07-robot-policy/latex/10-diffusion-policy-scratch/two-time-axes.png" alt="扩散步与动作块内物理未来步是两条不同时间轴" width="86%">

_图 7.10-5：扩散步 t 表示噪声层级；每个 t 上都有完整 H×D_a 动作块，其中 h 才是机器人未来的物理时间索引。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

1. **扩散去噪时间轴（Diffusion Timestep $t \in [1, T]$）**：
   - 描述的是**噪声污染与净化的程度**。$t = T$ 代表纯随机噪声混沌态，$t = 0$ 代表完全去噪后的纯净动作；
2. **物理未来时间轴（Horizon Horizon $h \in [1, H]$）**：
   - 描述的是**机器人在真实世界中即将执行的未来物理轨迹**。例如预测未来 $H = 16$ 步的连续位移：$[\mathbf{a}_{\tau}, \mathbf{a}_{\tau+1}, \dots, \mathbf{a}_{\tau+H-1}]$。

去噪网络在一次推理中，是以整条长度为 $H$ 的动作多维张量为单位进行集体去噪，从而保证了动作块内部物理加速度与速度的绝对连贯平滑。

---

## 7.10.5 纯底层 PyTorch 代码实现：从零构建扩散策略引擎

下面我们使用纯底层 PyTorch 算子实现完整的 DDPM 方差调度器、基于正弦位置编码的一维卷积去噪网络（Conditional 1D-CNN）以及训练循环。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DDPMScheduler:
    """
    DDPM 扩散方差调度器
    实现了前向加噪过程的方差累乘计算与单步瞬时加噪
    """
    def __init__(self, num_train_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_train_timesteps = num_train_timesteps

        # 1. 构造线性增长的方差序列 beta_t
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        # 2. 计算 alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas
        # 3. 计算累乘系数 alpha_bar_t = prod(alpha_1 ... alpha_t)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        前向单步封闭解加噪函数
        a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1 - alpha_bar_t) * noise
        :param original_samples: (B, H, action_dim) 原始动作序列 a_0
        :param noise: (B, H, action_dim) 采样的标准高斯噪声
        :param timesteps: (B,) 随机采样的扩散时间步
        :return: (B, H, action_dim) 加噪后的动作序列 a_t
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        # 提取 sqrt(alpha_bar_t)，并调整形状以支持广播
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1)

        # 提取 sqrt(1 - alpha_bar_t)
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)

        # 封闭解线性组合
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

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

# ===================================================================
# 单元测试与单步训练迭代验证
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    horizon = 16
    action_dim = 2
    obs_dim = 64
    num_timesteps = 100

    # 1. 实例化调度器与去噪网络
    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    model = Conditional1DCNN(action_dim=action_dim, obs_dim=obs_dim, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 2. 模拟专家动作块与多模态观测特征
    expert_actions = torch.randn(batch_size, horizon, action_dim)
    observations = torch.randn(batch_size, obs_dim)

    # 3. 执行单步训练计算
    optimizer.zero_grad()

    # 采样标准高斯白噪声
    true_noise = torch.randn_like(expert_actions)
    # 随机采样扩散步
    timesteps = torch.randint(0, num_timesteps, (batch_size,)).long()

    # 按照封闭解生成含噪动作 a_t
    noisy_actions = scheduler.add_noise(expert_actions, true_noise, timesteps)

    # 模型前向预测噪声
    predicted_noise = model(noisy_actions, timesteps, observations)

    # 计算 MSE 损失并反向传播
    loss = F.mse_loss(predicted_noise, true_noise)
    loss.backward()
    optimizer.step()

    print(f"[Diffusion Policy Test] 专家动作块形状: {expert_actions.shape}")
    print(f"[Diffusion Policy Test] 含噪动作块形状: {noisy_actions.shape}")
    print(f"[Diffusion Policy Test] 预测噪声张量形状: {predicted_noise.shape}")
    print(f"[Diffusion Policy Test] 单步加噪 MSE 训练损失: {loss.item():.4f}")

    assert predicted_noise.shape == (batch_size, horizon, action_dim), "输出张量形状不符！"
    assert not torch.isnan(loss), "训练损失出现 NaN！"
    print("✓ 扩散策略方差调度器与 1D-CNN 去噪网络单测全部通过！")
```

---

## 7.10.6 本节小结

回顾本节内容，我们完成了一套从热力学扩散理论到工程代码实现的完整闭环：
1. **多模态动作的物理本质**：扩散策略通过在连续去噪过程中拟合概率得分流，彻底解决了传统均方误差在多分支决策下的“算术平均相撞”难题；
2. **方差调度的代数美感**：利用高斯分布的叠加性质，百步加噪过程被提炼为初等代数的单步封闭解 $\mathbf{a}_t = \sqrt{\bar{\alpha}_t}\mathbf{a}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$；
3. **双时间轴的解耦设计**：清晰区分了去噪步 $t$ 与物理未来轨迹 $h$，以动作为整体块协同去噪，保障了机器人控制指令的时间连续性。
