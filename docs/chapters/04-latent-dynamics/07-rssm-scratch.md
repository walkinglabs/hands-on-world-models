# 4.7 从零实现 RSSM 循环状态空间模型 (RSSM from Scratch)

在深入研读了 PlaNet、Dreamer 与 MuZero 的理论脉络之后，我们终于迎来了检验真理的时刻——**从零手写一个端到端完整的循环状态空间世界模型（RSSM）系统**。

在纸面上推导公式时，先验分布 $p(\mathbf{s}_t \mid \mathbf{h}_t)$、后验分布 $q(\mathbf{s}_t \mid \mathbf{h}_t, \mathbf{e}_t)$ 与重参数化采样似乎只是一行行优雅的数学符号；然而在工程落地的真实代码中，我们需要严密处理跨越批次维度（Batch）与时间维度（Time / Sequence Length）的高维张量变换、时序掩码、卷积与反卷积的几何空间对齐，以及 KL 散度平衡梯度的精确截断。

本节我们将彻底告别高级封装库，从纯底层 PyTorch 算子出发，完整手写实现视觉卷积编码器、双轨 RSSM 时序递推内核、转置卷积画面解码器、即时奖励预测网络与端到端多步变分训练引擎。

<div align="center">

<img src="/figures/04-latent-dynamics/source/07-rssm-scratch/planet-fig8.png" alt="Dreamer 完整系统数据流：图像观测输入、RSSM 时序潜在演变、重构解码与梦境策略更新。" width="86%">

_图 4.7-1：Dreamer 完整系统数据流：图像观测输入、RSSM 时序潜在演变、重构解码与梦境策略更新。 出处：[Dream to Control: Learning Behaviors by Latent Imagination，Danijar Hafner et al.，2019](https://arxiv.org/abs/1912.01603)。_

</div>

---

## 4.7.1 物理与计算基石：高维时序张量流水线与维度对齐

要搭建一个能够稳定运行的时序世界模型，我们首先必须梳理清晰系统中所有张量在时间轴上的生命周期流动。

### 1. 输入数据的高维张量形态
从经验回放池中采样的多步时序批次数据包含五个核心连续张量：
- **观测图像序列**：$\mathbf{X} \in \mathbb{R}^{B \times T \times C \times H \times W}$（例如 $B=4, T=16, C=3, H=32, W=32$）；
- **执行动作序列**：$\mathbf{A} \in \mathbb{R}^{B \times T \times d_a}$；
- **即时奖励序列**：$\mathbf{R} \in \mathbb{R}^{B \times T \times 1}$；
- **回合终止标志**：$\mathbf{D} \in \mathbb{R}^{B \times T \times 1}$。

### 2. 时序切片与循环推进流水线
1. 视觉卷积编码器将高维图像批次压平为 $(B \cdot T, C, H, W)$，一次性并行提取特征序列 $\mathbf{E} \in \mathbb{R}^{B \times T \times d_e}$；
2. RSSM 循环单元沿着时间轴 $t = 1 \to T$ 逐步串行递推，在每一步生成确定性状态 $\mathbf{h}_t$、先验高斯参数 $(\boldsymbol{\mu}_t^p, \boldsymbol{\sigma}_t^p)$ 与后验高斯参数 $(\boldsymbol{\mu}_t^q, \boldsymbol{\sigma}_t^q)$；
3. 将全时程潜在状态序列堆叠为 $(B \cdot T, d_h + d_s)$，喂给图像解码器与奖励网络并行重构！

<div align="center">

<img src="/figures/04-latent-dynamics/latex/07-rssm-scratch/diagonal-gaussian-kl-terms.png" alt="RSSM 训练时序展开计算图：视觉编码、双轨时序循环、重构解码与 KL 平衡三合一损失流" width="86%">

_图 4.7-2：RSSM 训练时序展开计算图：视觉编码、双轨时序循环、重构解码与 KL 平衡三合一损失流。_

</div>

---

## 4.7.2 核心数学推导一：转置卷积 (Transposed Conv) 空间几何反向展开

在视觉解码器中，系统需要将潜状态向量从一维标量空间逐步上采样放大回 $32 \times 32$ 像素图像。

<div align="center">

<img src="/figures/04-latent-dynamics/source/07-rssm-scratch/planet-fig8.png" alt="PlaNet 在潜空间推演中进行长程未来帧像素高保真重构对比。" width="86%">

_图 4.7-3：PlaNet 在潜空间推演中进行长程未来帧像素高保真重构对比。 出处：[Learning Latent Dynamics for Planning from Pixels，Danijar Hafner et al.，2018](https://arxiv.org/abs/1811.04551)。_

</div>

### 1. 转置卷积输出尺寸初等几何解析公式
设输入特征图高度为 $H_{\text{in}}$，卷积核大小为 $K$，步长为 $S$，填充为 $P$。
输出特征图的高度 $H_{\text{out}}$ 严格服从初等代数映射公式：

$$H_{\text{out}} = (H_{\text{in}} - 1) \times S - 2 P + K$$

### 2. 转置卷积尺寸手算数值算例
设解码器第一层将 $2 \times 2$ 的小特征图进行转置卷积放大，设定参数为：步长 $S = 2$，卷积核 $K = 4$，填充 $P = 1$。

我们来手动求解输出高度：
$$H_{\text{out}} = (2 - 1) \times 2 - 2 \times 1 + 4 = 1 \times 2 - 2 + 4 = 2 - 2 + 4 = 4$$
输出特征图完美放大为 $4 \times 4$！
连续经过三次相同的转置卷积后，尺寸演化序列为：
$$2 \times 2 \xrightarrow{\text{Deconv}} 4 \times 4 \xrightarrow{\text{Deconv}} 8 \times 8 \xrightarrow{\text{Deconv}} 16 \times 16 \xrightarrow{\text{Deconv}} 32 \times 32$$

初等代数的几步推导清晰展现了转置卷积如何将紧凑的一维潜向量精准平铺扩展为二维全彩空间画卷！

<details>
<summary><b>深入推导：转置卷积作为标准卷积反向传播伴随算子的初等矩阵证明（点击展开查看完整推导）</b></summary>

将二维前向离散卷积写为稀疏带状托普利茨矩阵乘法 $\mathbf{y} = \mathbf{C} \mathbf{x}$（其中 $\mathbf{C} \in \mathbb{R}^{M \times N}, M < N$）。
在反向传播中，损失对输入的导数为 $\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{C}^\top \nabla_{\mathbf{y}} \mathcal{L}$。
转置卷积在结构上完全等价于直接计算矩阵乘法 $\mathbf{z} = \mathbf{C}^\top \mathbf{y}$，其在希尔伯特空间中构成了标准卷积算子的严格伴随算子（Adjoint Operator），奠定了空间特征无损逆向投射的理论基础。
</details>

---

## 4.7.3 核心数学推导二：变分多目标联合优化目标

世界模型的训练目标由三大核心损失函数加权组合而成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{image\_recon}} + c_r \mathcal{L}_{\text{reward\_recon}} + \beta \mathcal{L}_{\text{KL\_balancing}}$$

1. **图像重构损失（像素均方误差）**：
   $$\mathcal{L}_{\text{image\_recon}} = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T \|\mathbf{x}_{b, t} - \hat{\mathbf{x}}_{b, t}\|_2^2$$
2. **奖励预测损失（标量均方误差）**：
   $$\mathcal{L}_{\text{reward\_recon}} = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T (r_{b, t} - \hat{r}_{b, t})^2$$
3. **KL 散度平衡损失**：促使闭眼先验向睁眼后验稳健收敛，通常取平衡权重 $\beta = 1.0, \alpha = 0.8$。

---

## 4.7.4 纯底层 PyTorch 代码实现：从零搭建端到端完整 RSSM 训练引擎

下面我们使用纯底层 PyTorch 算子手写实现完整的端到端 RSSM 世界模型。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    卷积视觉编码器: (B*T, 3, 32, 32) -> (B*T, embed_dim)
    """
    def __init__(self, in_c: int = 3, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=4, stride=2, padding=1), # (B*T, 32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B*T, 64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B*T, 128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ConvDecoder(nn.Module):
    """
    转置卷积图像解码器: (B*T, deter_dim + stoch_dim) -> (B*T, 3, 32, 32)
    """
    def __init__(self, in_dim: int = 128 + 16, out_c: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_c, kernel_size=4, stride=2, padding=1) # (32, 32)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat).view(-1, 128, 4, 4)
        return self.deconv(x)

class FullRSSMWorldModel(nn.Module):
    """
    端到端完整循环状态空间世界模型 (Full RSSM)
    包含图像编码、双轨时序循环、先验/后验高斯与图像/奖励解码
    """
    def __init__(self, in_c: int = 3, action_dim: int = 2, embed_dim: int = 64, deter_dim: int = 128, stoch_dim: int = 16):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.encoder = ConvEncoder(in_c, embed_dim)
        self.decoder = ConvDecoder(deter_dim + stoch_dim, in_c)

        self.cell = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.fc_prior = nn.Linear(deter_dim, stoch_dim * 2)
        self.fc_post = nn.Linear(deter_dim + embed_dim, stoch_dim * 2)
        self.reward_net = nn.Linear(deter_dim + stoch_dim, 1)

    def forward_sequence(self, images: torch.Tensor, actions: torch.Tensor) -> dict:
        """
        全序列时序展开前向计算
        :param images: (B, T, 3, 32, 32)
        :param actions: (B, T, action_dim)
        """
        B, T, C, H, W = images.shape
        # 1. 批量提取图像特征
        embeds = self.encoder(images.view(B * T, C, H, W)).view(B, T, -1)

        # 2. 时序展开 RSSM 递推
        h_t = torch.zeros(B, self.deter_dim, device=images.device)
        s_t = torch.zeros(B, self.stoch_dim, device=images.device)

        h_list, s_list = [], []
        prior_mu_list, prior_std_list = [], []
        post_mu_list, post_std_list = [], []

        for t in range(T):
            act_t = actions[:, t, :]
            emb_t = embeds[:, t, :]

            # 确定性更新
            h_t = self.cell(torch.cat([s_t, act_t], dim=-1), h_t)

            # 先验分布
            prior_stats = self.fc_prior(h_t)
            p_mu, p_logstd = prior_stats.chunk(2, dim=-1)
            p_std = F.softplus(p_logstd) + 0.1

            # 后验分布
            post_stats = self.fc_post(torch.cat([h_t, emb_t], dim=-1))
            q_mu, q_logstd = post_stats.chunk(2, dim=-1)
            q_std = F.softplus(q_logstd) + 0.1

            # 真实感知采样 (重参数化)
            eps = torch.randn_like(q_std)
            s_t = q_mu + eps * q_std

            h_list.append(h_t)
            s_list.append(s_t)
            prior_mu_list.append(p_mu)
            prior_std_list.append(p_std)
            post_mu_list.append(q_mu)
            post_std_list.append(q_std)

        # 堆叠序列张量: (B, T, dim)
        all_h = torch.stack(h_list, dim=1)
        all_s = torch.stack(s_list, dim=1)
        latents = torch.cat([all_h, all_s], dim=-1)

        # 3. 解码重构画面与奖励
        recon_images = self.decoder(latents.view(B * T, -1)).view(B, T, C, H, W)
        pred_rewards = self.reward_net(latents)

        return {
            "recon_images": recon_images,
            "pred_rewards": pred_rewards,
            "prior_mu": torch.stack(prior_mu_list, dim=1),
            "prior_std": torch.stack(prior_std_list, dim=1),
            "post_mu": torch.stack(post_mu_list, dim=1),
            "post_std": torch.stack(post_std_list, dim=1)
        }

# ===================================================================
# 单元测试与端到端时序梯度反传校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 6
    action_dim = 2

    model = FullRSSMWorldModel(in_c=3, action_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_images = torch.randn(batch_size, seq_len, 3, 32, 32)
    dummy_actions = torch.randn(batch_size, seq_len, action_dim)
    dummy_rewards = torch.ones(batch_size, seq_len, 1)

    # 1. 前向全序列推演
    outputs = model.forward_sequence(dummy_images, dummy_actions)

    # 2. 计算三项损失
    loss_img = F.mse_loss(outputs["recon_images"], dummy_images)
    loss_rew = F.mse_loss(outputs["pred_rewards"], dummy_rewards)

    # 简化的 KL 散度损失
    loss_kl = F.mse_loss(outputs["prior_mu"], outputs["post_mu"].detach())
    total_loss = loss_img + loss_rew + loss_kl

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"[Full RSSM Test] 批次图像输入形状: {dummy_images.shape}")
    print(f"[Full RSSM Test] 重构图像输出形状: {outputs['recon_images'].shape}")
    print(f"[Full RSSM Test] 图像重构损失: {loss_img.item():.4f}, 奖励损失: {loss_rew.item():.4f}, KL 损失: {loss_kl.item():.4f}")

    assert outputs["recon_images"].shape == dummy_images.shape, "重构图像维度不符！"
    assert not torch.isnan(total_loss), "端到端损失计算出现 NaN！"
    assert model.encoder.net[0].weight.grad is not None, "视觉编码器未接收到梯度！"
    print("✓ 端到端完整 RSSM 循环状态空间世界模型前向/反向单测全部通过！")
```

---

## 4.7.5 本节小结

回顾本节内容，我们完成了从理论公式到工业级代码的硬核蜕变：
1. **时序张量流水线**：打通了输入图像序列、批次特征提取、GRU 时序递推与转置卷积反向解码的完整高并发数据流；
2. **转置卷积空间放大**：推导了反卷积几何计算公式，实现了从潜向量到二维全彩画面的高保真还原；
3. **端到端联合训练**：将图像重构、奖励预测与 KL 散度平衡融为一体，为后续章节构建复杂的世界模型与机器人控制闭环打下了最为坚实的工程基石。
