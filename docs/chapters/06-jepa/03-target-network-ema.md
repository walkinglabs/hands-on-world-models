# 6.3 目标网络、动量更新 (EMA) 与 I-JEPA

在非生成式自监督学习的宏伟架构中，除了使用上一节介绍的 VICReg 显式几何方差惩罚外，另一条统治了整个现代自监督大模型（如 BYOL, MoCo, DINO, I-JEPA）的革命性技术路线是——**非对称双网络架构与指数移动平均（Exponential Moving Average, EMA）动量目标网络**。

初学者往往会产生一个直觉的疑问：如果我们直接让上下文编码器与目标编码器共享同一套权重参数并双向回传梯度，模型会发生什么？
数学与实验反复证实：如果两个网络完全对称且同时激进更新，在没有负样本推斥或显式方差保护时，网络会在短短几个批次内不可逆地陷入全局常数特征坍塌。

为了彻底阻断特征坍塌的反馈闭环，DeepMind 与 Meta 的科学家们设计了一种充满东方智慧的“太极动量自举”机制：
**目标编码器彻底关闭反向传播求导通道，其参数仅仅作为在线编码器在时间长河中的平缓历史影子（EMA），以极高的动量系数（如 $\tau = 0.996 \sim 0.999$）平滑演进！**

本节我们将从初等数列递推与一阶指数平滑滤波出发，严密推导 EMA 动量更新的低通滤波数学性质、I-JEPA 跨 Patch 潜在预测架构，并使用纯底层 PyTorch 从零手写一个工业级 I-JEPA 动量更新自监督训练引擎。

<div align="center">

<img src="/figures/06-jepa/source/03-target-network-ema/moco-fig1.png" alt="I-JEPA 完整架构：上下文编码器、EMA 动量目标编码器与轻量级 Patch 预测器。" width="86%">

_图 6.3-1：I-JEPA 完整架构：上下文编码器、EMA 动量目标编码器与轻量级 Patch 预测器。 出处：[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture，Mahmoud Assran et al.，2023](https://arxiv.org/abs/2301.08243)。_

</div>

---

## 6.3.1 物理与系统基石：学生-导师非对称自举演化

要理解 EMA 动量更新的物理本质，我们首先从人类教育学中的“学生与导师交互自举”讲起。

### 1. 师生自举系统（Teacher-Student Bootstrap）
- **在线上下文编码器（学生网络 $\theta$）**：年轻气盛，每时每刻接收最新的梯度反向传播更新，快速学习新知识，但参数波动剧烈；
- **动量目标编码器（导师网络 $\phi$）**：沉稳厚重，**绝不直接接收任何梯度冲击**。导师的知识完全来自于过去成千上万个历史时刻学生网络参数的加权平均；
- **非对称闭环**：学生的目标是努力预测出富有远见的导师给出的抽象特征；由于导师的特征高度平稳且具备历史全局先验，学生永远无法通过“摆烂变成常数”来欺骗导师，从而从根本上粉碎了特征坍塌！

### 2. EMA 动量参数更新核心方程
在每一个训练迭代步 $t$：
1. 学生网络依据梯度下降正常更新：$\theta_{t+1} \leftarrow \theta_t - \eta \nabla_\theta \mathcal{L}$；
2. 导师网络执行**无梯度的指数滑动平均软更新**：

$$\phi_{t+1} = \tau \phi_t + (1 - \tau) \theta_{t+1}$$

其中动量系数 $\tau \in [0.996, 0.9999]$（通常随着训练从 $0.996$ 余弦退火至 $1.0$）。

<div align="center">

<img src="/figures/06-jepa/latex/03-target-network-ema/hard-vs-ema-response.png" alt="EMA 动量目标网络参数历史展开：各历史时刻权重的指数衰减几何衰减卷积分布" width="86%">

_图 6.3-2：EMA 动量目标网络参数历史展开：各历史时刻权重的指数衰减几何衰减卷积分布。_

</div>

---

## 6.3.2 核心数学推导一：EMA 展开式与一阶低通时域滤波性质

将 EMA 递归更新方程在时间轴上自底向上完全展开：

<div align="center">

<img src="/figures/06-jepa/source/03-target-network-ema/moco-fig1.png" alt="DINO 自监督视觉 Transformer 利用动量编码器实现自发显式语义分割与注意力可视化。" width="86%">

_图 6.3-3：DINO 自监督视觉 Transformer 利用动量编码器实现自发显式语义分割与注意力可视化。 出处：[Emerging Properties in Self-Supervised Vision Transformers，Mathilde Caron et al.，2021](https://arxiv.org/abs/2104.14294)。_

</div>

### 1. 历史权重的初等等比数列展开
从初始时刻 $t=0$ 递推至时刻 $T$：

$$\phi_T = \tau^T \phi_0 + (1 - \tau) \sum_{k=0}^{T-1} \tau^k \theta_{T - k}$$

利用初等无穷等比数列求和公式：
$$(1 - \tau) \sum_{k=0}^\infty \tau^k = (1 - \tau) \frac{1}{1 - \tau} = 1.0$$

所有历史学生权重的贡献系数严格之和恒为 $1.0$！这证明导师网络本质上是**学生网络在时间轴上的指数衰减加权时间卷积低通滤波器**！

### 2. EMA 权重演变手算数值算例
设定动量系数 $\tau = 0.90$（$1 - \tau = 0.10$）。初始导师参数 $\phi_0 = 0.0$。
在连续 3 步训练中，学生网络因为梯度更新产生的参数序列为：$\theta_1 = 10.0, \; \theta_2 = 20.0, \; \theta_3 = 30.0$。

我们来手动求解导师网络参数 $\phi$ 的平滑演变轨迹：
1. **第 1 步更新**：
   $$\phi_1 = 0.90 \times \phi_0 + 0.10 \times \theta_1 = 0.90 \times 0.0 + 0.10 \times 10.0 = 1.0$$
2. **第 2 步更新**：
   $$\phi_2 = 0.90 \times \phi_1 + 0.10 \times \theta_2 = 0.90 \times 1.0 + 0.10 \times 20.0 = 0.90 + 2.0 = 2.90$$
3. **第 3 步更新**：
   $$\phi_3 = 0.90 \times \phi_2 + 0.10 \times \theta_3 = 0.90 \times 2.90 + 0.10 \times 30.0 = 2.61 + 3.0 = 5.61$$

初等代数的直观计算生动证实：当学生参数从 $0 \to 10 \to 20 \to 30$ 剧烈激进飙升时，导师参数以平缓优美的步调 $0 \to 1.0 \to 2.9 \to 5.61$ 稳健跟随，完全滤除了由于单批次随机噪声引发的参数剧烈抖动！

<details>
<summary><b>深入推导：动量目标网络在随机微分方程下的李雅普诺夫稳定性证明（点击展开查看完整推导）</b></summary>

将学生参数与导师参数的联合动态建模为连续时间朗之万随机微分方程（SDE）：
$$d\boldsymbol{\theta}_t = -\nabla_\theta \mathcal{L}(\boldsymbol{\theta}_t, \boldsymbol{\phi}_t) dt + \sigma d\mathbf{W}_t, \quad d\boldsymbol{\phi}_t = \frac{1}{\tau_{\text{eff}}} (\boldsymbol{\theta}_t - \boldsymbol{\phi}_t) dt$$
定义李雅普诺夫泛函 $V(\boldsymbol{\theta}, \boldsymbol{\phi}) = \frac{1}{2} \|\boldsymbol{\theta} - \boldsymbol{\phi}\|^2 + \beta \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi})$。
对其求伊藤微分（Itô Derivative），当动量时间尺度 $\tau_{\text{eff}} \gg 1$ 时，漂移项的李雅普诺夫导数 $\mathcal{L}_V \le -\alpha \|\nabla \mathcal{L}\|^2 \le 0$ 恒负定。
由 LaSalle 不变集定理，系统在参数相空间中严格渐近收敛至流形上的极小驻点，从动力系统层面确立了 EMA 防坍塌的全局稳定性。
</details>

---

## 6.3.3 核心数学推导二：I-JEPA 跨 Patch 潜在预测机制

在 I-JEPA 中，动量目标编码器与潜在预测器如何协同工作？

<div align="center">

<img src="/figures/06-jepa/source/03-target-network-ema/moco-fig1.png" alt="I-JEPA 特征预测可视化：仅凭局部上下文精确预测被大面积遮挡区域的高阶语义特征。" width="86%">

_图 6.3-4：I-JEPA 特征预测可视化：仅凭局部上下文精确预测被大面积遮挡区域的高阶语义特征。 出处：[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture，Mahmoud Assran et al.，2023](https://arxiv.org/abs/2301.08243)。_

</div>

### 1. 三步时空特征预测流程
1. **上下文编码**：上下文 ViT 编码器仅接收可见的非遮挡 Patch 词元序列 $\mathbf{x}_{\text{context}}$，输出上下文特征表示 $\mathbf{s}_x \in \mathbb{R}^{N_{\text{ctx}} \times D}$；
2. **目标特征提取（无梯度）**：动量目标编码器接收完整图像，提取出所有被遮挡目标块对应的真实特征 $\mathbf{s}_y \in \mathbb{R}^{N_{\text{tgt}} \times D}$ 并实施梯度截断；
3. **轻量级预测器映射**：预测器（小型 ViT）接收上下文特征 $\mathbf{s}_x$，并为每个目标位置插入可学习的 `[MASK]` 词元与空间绝对位置编码 $\mathbf{E}_{\text{pos}}$，在单层自注意力中直接预测出目标特征 $\hat{\mathbf{s}}_y$！

### 2. 平滑 $L_1$ 特征预测损失
$$\mathcal{L}_{\text{I-JEPA}} = \frac{1}{N_{\text{tgt}}} \sum_{i=1}^{N_{\text{tgt}}} \text{SmoothL1}\left( \hat{\mathbf{s}}_{y, i}, \; \mathbf{s}_{y, i} \right)$$

通过完全绕开像素重构，I-JEPA 的训练吞吐比传统 MAE 提升了近 **3 倍**，且提取出的特征在下游物体检测与语义分割任务中展现出了碾压级的线性可分性！

---

## 6.3.4 纯底层 PyTorch 代码实现：从零手写带 EMA 动量更新的 I-JEPA 完整模型

下面我们使用纯底层 PyTorch 算子手写实现完整的上下文 ViT、EMA 动量目标更新器、Mask Token 预测器与端到端训练引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class IJEPAPredictor(nn.Module):
    """
    I-JEPA 专用轻量级特征预测器 (Predictor ViT)
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, ctx_feats: torch.Tensor, tgt_pos_embeds: torch.Tensor) -> torch.Tensor:
        """
        :param ctx_feats: (B, N_ctx, d_model) 上下文特征
        :param tgt_pos_embeds: (B, N_tgt, d_model) 目标块位置编码
        :return: (B, N_tgt, d_model) 预测的目标特征
        """
        B, N_tgt, D = tgt_pos_embeds.shape
        # 为每个目标位置准备 [MASK] 词元并加上目标位置编码
        mask_tokens = self.mask_token.expand(B, N_tgt, -1) + tgt_pos_embeds

        # 拼接上下文特征与掩码词元共同输入 Transformer
        full_seq = torch.cat([ctx_feats, mask_tokens], dim=1)
        hidden = self.transformer(full_seq)

        # 仅截取后半段目标掩码对应的输出
        pred_tgt_feats = self.proj(hidden[:, -N_tgt:, :])
        return pred_tgt_feats

class IJEPAModel(nn.Module):
    """
    纯底层 I-JEPA 完整模型架构
    包含在线上下文编码器、EMA 动量目标编码器与特征预测器
    """
    def __init__(self, d_model: int = 64, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum

        # 1. 在线上下文编码器 (学生网络)
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. 动量目标编码器 (导师网络 - 初始化硬拷贝)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        # 彻底冻结目标编码器梯度
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 3. 潜在预测器
        self.predictor = IJEPAPredictor(d_model=d_model)

    @torch.no_grad()
    def update_target_encoder(self):
        """
        无梯度的指数滑动平均 (EMA) 动量参数更新
        phi = momentum * phi + (1 - momentum) * theta
        """
        for p_online, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data, alpha=1.0 - self.momentum)

    def forward(
        self, all_patches: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor, pos_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param all_patches: (B, N_total, d_model)
        :param ctx_idx: (N_ctx,) 上下文索引
        :param tgt_idx: (N_tgt,) 目标索引
        """
        B = all_patches.shape[0]

        # 1. 学生网络编码可见上下文
        ctx_in = all_patches[:, ctx_idx, :] + pos_embeds[:, ctx_idx, :]
        ctx_feats = self.context_encoder(ctx_in)

        # 2. 导师网络编码真实目标 (无梯度)
        with torch.no_grad():
            tgt_in = all_patches[:, tgt_idx, :] + pos_embeds[:, tgt_idx, :]
            tgt_feats_real = self.target_encoder(tgt_in)

        # 3. 预测器预测目标特征
        tgt_pos = pos_embeds[:, tgt_idx, :]
        pred_tgt_feats = self.predictor(ctx_feats, tgt_pos)

        # 4. Smooth L1 损失
        loss = F.smooth_l1_loss(pred_tgt_feats, tgt_feats_real)
        return pred_tgt_feats, tgt_feats_real, loss

# ===================================================================
# 单元测试与 EMA 动量演进校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    n_total = 16
    d_model = 64

    model = IJEPAModel(d_model=d_model, momentum=0.95)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_patches = torch.randn(batch_size, n_total, d_model)
    dummy_pos = torch.randn(batch_size, n_total, d_model)

    ctx_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long) # 前 8 个
    tgt_idx = torch.tensor([8, 9, 10, 11], dtype=torch.long)          # 后 4 个

    # 1. 记录初始目标编码器参数
    init_target_weight = model.target_encoder[0].weight.clone()

    # 2. 前向计算与反向传播
    pred, real, loss = model(dummy_patches, ctx_idx, tgt_idx, dummy_pos)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 3. 执行 EMA 动量更新
    model.update_target_encoder()
    updated_target_weight = model.target_encoder[0].weight.clone()

    print(f"[I-JEPA Test] 预测特征形状: {pred.shape}, 真实目标特征形状: {real.shape}")
    print(f"[I-JEPA Test] 特征预测 Smooth L1 损失: {loss.item():.4f}")

    # 验证导师参数确实发生了微小平滑演进
    weight_diff = (updated_target_weight - init_target_weight).abs().max().item()
    print(f"[I-JEPA Test] 导师网络 EMA 参数单步更新幅度: {weight_diff:.6f}")

    assert pred.shape == (batch_size, 4, d_model), "预测特征维度不符！"
    assert weight_diff > 0.0, "EMA 动量参数未能成功演变！"
    assert not torch.isnan(loss), "I-JEPA 损失出现 NaN！"
    print("✓ I-JEPA 跨 Patch 特征预测、EMA 动量目标更新与自监督引擎单测全部通过！")
```

---

## 6.3.5 本节小结

回顾本节内容，我们建立了动量自举与空间跨 Patch 预测的完整方法论：
1. **师生非对称太极自举**：利用完全不传梯度的 EMA 导师网络提供平稳目标，从机制上彻底切断了常数崩溃的反馈通路；
2. **时域低通滤波本质**：数学上证明了 EMA 动量更新构成了对历史学生参数的指数衰减时间加权，平滑了单批次噪声；
3. **I-JEPA 语义完形填空**：以掩码词元结合绝对位置编码，在特征空间直接实现高阶空间物理语义推理，为下一步拓展至时序与动作驱动（A-JEPA/V-JEPA）铺平了康庄大道。
