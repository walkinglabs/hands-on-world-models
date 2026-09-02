# 6.2 掩码策略、特征坍塌与 VICReg 崩溃预防

在非生成式自监督学习与联合嵌入架构（JEPA）的发展历程中，研究者们遭遇的最凶险、最隐蔽的数学暗礁莫过于——**表征坍塌（Representation Collapse）**。

在传统的生成式模型（如 VAE 或扩散模型）中，由于存在像素级的重构约束，网络绝不敢把输出变成一个常数；而在纯粹基于特征空间预测的联合嵌入架构中，由于彻底移除了像素解码器，两个编码器为了将特征预测均方误差 $\|E_\theta(\mathbf{x}) - E_\phi(\mathbf{y})\|^2$ 迅速降为 $0$，会本能地发现一条极具投机性的“数学后门”：
**无论输入什么图像，网络都将特征向量恒等输出为全零向量 $\mathbf{0}$（或任意固定常数向量）！**

此时，预测损失在表面上完美收敛到了令人心动的 $0.0000$，但整个神经网络已经彻底死锁为一具毫无信息价值的空壳。

为了彻底粉碎特征坍塌的数学诱惑，同时迫使模型掌握跨越广阔空间的高阶物理语义：
- **VICReg（Variance-Invariance-Covariance Regularization, 2021）** 提出了三位一体的显式几何约束，从数学上强制特征流形必须张开为高维超椭球体；
- **I-JEPA（Image JEPA, Meta 2023）** 则设计了**多尺度大块状空间掩码（Multi-Block Masking）**，彻底切断了局部像素插值的投机路径，迫使网络必须理解物体的全局物理结构！

本节我们将从初等样本方差、协方差矩阵与信息熵出发，严密推导 VICReg 三大正则化损失方程与 I-JEPA 块状掩码的几何采样机理，并使用纯底层 PyTorch 从零手写一个防坍塌评估引擎与时空掩码采样器。

<div align="center">

<img src="/figures/06-jepa/source/02-mask-collapse/vicreg-fig1.png" alt="VICReg 三位一体架构：不变性损失 (Invariance)、方差正则 (Variance) 与协方差解耦 (Covariance)。" width="86%">

_图 6.2-1：VICReg 三位一体架构：不变性损失 (Invariance)、方差正则 (Variance) 与协方差解耦 (Covariance)。 出处：[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning，Adrien Bardes et al.，2021](https://arxiv.org/abs/2105.04906)。_

</div>

---

## 6.2.1 物理与几何基石：特征空间的维度坍塌与流形退化

要理解特征坍塌的本质，我们首先从初等线性代数的空间秩（Rank）审视潜在特征流形。

### 1. 维度坍塌（Dimensional Collapse）
设潜在特征空间的理论维度为 $d = 64$。
- **健康状态**：特征样本散落在 64 维空间的各个正交子空间中，充满活力，协方差矩阵的秩为满秩 $\text{rank}(\mathbf{\Sigma}) = 64$；
- **部分坍塌状态**：所有样本被压缩坍塌在一条细细的一维直线或一个二维平面上，其余 62 个维度完全休克，秩骤降为 $\text{rank}(\mathbf{\Sigma}) \le 2$；
- **完全坍塌状态**：所有样本坍塌为单一固定常数点，方差为 0，秩为 0。

### 2. VICReg 的三大显式几何铁律
为了在不依赖负样本对（Negative Pairs）的前提下保证满秩表达，VICReg 树立了三道数学防线：
1. **不变性（Invariance）**：同一场景在不同视角下的特征表示必须尽可能接近；
2. **方差性（Variance）**：每一个特征维度在批次内的标准差必须严格大于阈值 1.0（不准变成常数！）；
3. **协方差性（Covariance）**：不同特征维度之间的协方差必须严格趋近于 0（各维度信息相互独立，杜绝信息冗余！）。

<div align="center">

<img src="/figures/06-jepa/latex/02-mask-collapse/infonce-softmax-competition.png" alt="VICReg 几何流形展开：方差项沿各正交轴撑开超椭球，协方差项消除非对角轴倾斜关联" width="86%">

_图 6.2-2：VICReg 几何流形展开：方差项沿各正交轴撑开超椭球，协方差项消除非对角轴倾斜关联。_

</div>

---

## 6.2.2 核心数学推导一：VICReg 三项显式几何正则化损失

设输入批次包含 $N$ 个样本，两个分支输出的特征矩阵分别为 $\mathbf{Z}, \mathbf{Z}' \in \mathbb{R}^{N \times d}$。

<div align="center">

<img src="/figures/06-jepa/source/02-mask-collapse/ijepa-fig4.png" alt="I-JEPA 在 ImageNet 分类与密集下游任务中对比 MAE 与 DINO，展示特征语义深度的显著优势。" width="86%">

_图 6.2-3：I-JEPA 在 ImageNet 分类与密集下游任务中对比 MAE 与 DINO，展示特征语义深度的显著优势。 出处：[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture，Mahmoud Assran et al.，2023](https://arxiv.org/abs/2301.08243)。_

</div>

### 1. 不变性均方损失（Invariance Term $s$）
衡量两个视图特征之间的欧几里得距离：

$$s(\mathbf{Z}, \mathbf{Z}') = \frac{1}{N} \sum_{i=1}^N \|\mathbf{z}_i - \mathbf{z}'_i\|_2^2$$

### 2. 方差铰链损失（Variance Hinge Term $v$）
强制每一个特征通道 $j \in \{1, \dots, d\}$ 的样本标准差保持在目标值 $\gamma = 1.0$ 以上：

$$v(\mathbf{Z}) = \frac{1}{d} \sum_{j=1}^d \max\left( 0, \; \gamma - \sqrt{\text{Var}(\mathbf{Z}[:, j]) + \epsilon} \right)$$

其中单通道样本方差计算公式为：

$$\text{Var}(\mathbf{Z}[:, j]) = \frac{1}{N - 1} \sum_{i=1}^N \left( \mathbf{Z}[i, j] - \bar{\mathbf{z}}_j \right)^2, \quad \bar{\mathbf{z}}_j = \frac{1}{N} \sum_{i=1}^N \mathbf{Z}[i, j]$$

### 3. 协方差去相关损失（Covariance De-correlation Term $c$）
计算批次特征的样本协方差矩阵 $\mathbf{C}(\mathbf{Z}) \in \mathbb{R}^{d \times d}$：

$$\mathbf{C}(\mathbf{Z}) = \frac{1}{N - 1} \sum_{i=1}^N (\mathbf{z}_i - \bar{\mathbf{z}})(\mathbf{z}_i - \bar{\mathbf{z}})^\top$$

惩罚所有非对角线元素（$i \ne j$）的平方和，迫使其正交解耦：

$$c(\mathbf{Z}) = \frac{1}{d} \sum_{i=1}^d \sum_{j \ne i}^d \mathbf{C}_{i, j}^2(\mathbf{Z})$$

### 4. VICReg 联合损失函数
$$\mathcal{L}_{\text{VICReg}} = \lambda \cdot s(\mathbf{Z}, \mathbf{Z}') + \mu \cdot [v(\mathbf{Z}) + v(\mathbf{Z}')] + \nu \cdot [c(\mathbf{Z}) + c(\mathbf{Z}')]$$

通常取超参数权重 $\lambda = 25.0, \mu = 25.0, \nu = 1.0$。

### 5. 方差与协方差手算数值算例
设批次大小 $N = 2$，特征维度 $d = 2$。
当前分支输出的特征矩阵为：
$$\mathbf{Z} = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}$$
1. **计算各列均值**：
   $$\bar{z}_1 = \frac{1.0 + 3.0}{2} = 2.0, \quad \bar{z}_2 = \frac{2.0 + 4.0}{2} = 3.0$$
2. **计算各列方差与标准差**（$N - 1 = 1$）：
   $$\text{Var}(z_1) = \frac{(1 - 2)^2 + (3 - 2)^2}{1} = 1 + 1 = 2.0 \implies \text{std}(z_1) = \sqrt{2} \approx 1.414$$
   $$\text{Var}(z_2) = \frac{(2 - 3)^2 + (4 - 3)^2}{1} = 1 + 1 = 2.0 \implies \text{std}(z_2) = \sqrt{2} \approx 1.414$$
   因为 $\text{std} = 1.414 \ge 1.0$，方差损失为 $\max(0, 1 - 1.414) = 0.0$（达标！）；
3. **计算非对角协方差 $\mathbf{C}_{1, 2}$**：
   $$\mathbf{C}_{1, 2} = \frac{(1.0 - 2.0)(2.0 - 3.0) + (3.0 - 2.0)(4.0 - 3.0)}{1} = (-1)(-1) + (1)(1) = 1 + 1 = 2.0$$
   协方差惩罚项为：$c = \frac{1}{2} (2.0^2 + 2.0^2) = \frac{8}{2} = 4.0$。

初等代数的几步推导清晰展现：协方差项将产生强大的正交推力，迫使第 1 通道与第 2 通道学会捕捉完全不同方向的物理特征，最大化了潜在空间的表达效率！

<details>
<summary><b>深入推导：VICReg 在特征空间超椭球主成分覆盖下的主成分分析最大熵等价证明（点击展开查看完整推导）</b></summary>

设多元连续随机变量 $\mathbf{Z} \in \mathbb{R}^d$ 的协方差矩阵为 $\mathbf{\Sigma}$。
在固定能量约束下，多元高斯分布的微分熵达到极大值 $H(\mathbf{Z}) = \frac{1}{2} \log \det(2\pi e \mathbf{\Sigma})$。
利用哈达玛不等式（Hadamard's Inequality）：
$$\det(\mathbf{\Sigma}) \le \prod_{i=1}^d \mathbf{\Sigma}_{i, i}$$
等号成立当且仅当 $\mathbf{\Sigma}$ 为严格对角矩阵（即所有非对角协方差 $\mathbf{\Sigma}_{i, j} = 0, \forall i \ne j$）。
VICReg 的方差约束保证了对角元素 $\mathbf{\Sigma}_{i, i} \ge \gamma^2$，协方差约束逼近哈达玛上界，严格证明了极小化 VICReg 损失等价于最大化特征流形的信息熵容量。
</details>

---

## 6.2.3 核心数学推导二：I-JEPA 语义完形填空的大块状空间掩码 (Block Masking)

除了显式损失正则化外，掩码采样策略（Masking Strategy）的设计直接决定了网络到底是在学习“高阶物理语义”还是在做“低级几何插值”。

<div align="center">

<img src="/figures/06-jepa/source/02-mask-collapse/ijepa-fig4.png" alt="I-JEPA 块状掩码生成策略：大面积目标掩码迫使网络理解物体全局语义。" width="86%">

_图 6.2-4：I-JEPA 块状掩码生成策略：大面积目标掩码迫使网络理解物体全局语义。 出处：[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture，Mahmoud Assran et al.，2023](https://arxiv.org/abs/2301.08243)。_

</div>

### 1. 细粒度随机点状掩码（MAE）的投机漏洞
在传统的 Masked Autoencoders（MAE）中，网络随机扣掉 $75\%$ 互不相连的细碎小像素点。
由于相邻像素具有极强的连续性，网络只需根据相邻像素进行初等双线性插值就能轻松蒙混过关，根本不需要理解画面中到底是一只猫还是一辆卡车。

### 2. I-JEPA 的大块状多尺度目标掩码（Target Block Masking）
I-JEPA 采用了极具侵略性的**连续大块状遮挡**：
- **目标掩码（Target Blocks, $4$ 块）**：每次随机遮挡占全图比例 $15\% \sim 20\%$ 的连续大方形区域（宽高比在 $0.75 \sim 1.5$ 之间）；
- **上下文掩码（Context Block, $1$ 块）**：从剩余区域截取一个大尺寸上下文窗口（占全图 $85\% \sim 100\%$），并显式剔除所有目标块；
- 这种大尺度时空阻断，彻底斩断了局部像素插值的可能，迫使上下文编码器必须从残缺画面中推断出整体物体的三维几何结构与语义因果！

<details>
<summary><b>深入推导：块状掩码在马尔可夫随机场语义马尔可夫毯下的信息截断证明（点击展开查看完整推导）</b></summary>

将图像网格建模为无向高斯马尔可夫随机场（GMRF）。
任意节点 $x_i$ 关于全局状态的条件依赖受限于其邻域边界构成的马尔可夫毯（Markov Blanket $\partial \Omega$）。
当掩码区域直径 $D_{\text{mask}} \gg 2 r_{\text{kernel}}$（掩码尺度显著大于底层卷积/注意力感受野半径）时：
$$I(\mathbf{X}_{\text{target}}; \; \mathbf{X}_{\text{context}} \mid \text{Semantic Concept } C) \to 0$$
底层像素互信息被阻断为零，信息流被迫必须通过高阶语义概念 $C$（如“头部与躯干的连接拓扑”）进行长程桥接，奠定了高阶语义涌现的几何必然性。
</details>

---

## 6.2.4 纯底层 PyTorch 代码实现：从零手写 VICReg 防坍塌层与 Block Mask 采样器

下面我们使用纯底层 PyTorch 算子手写实现标准的 VICReg 方差-协方差正则化计算层与二维图像多尺度块状掩码采样器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """
    纯底层 VICReg 崩溃预防三元损失计算层
    L = lambda * Invariance + mu * Variance + nu * Covariance
    """
    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        :param z_a: (N, D) 分支 A 特征
        :param z_b: (N, D) 分支 B 特征
        """
        N, D = z_a.shape

        # 1. 不变性损失 (Invariance / Sim Loss)
        sim_loss = F.mse_loss(z_a, z_b)

        # 2. 方差铰链损失 (Variance Loss)
        # 计算各维度标准差
        std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1.0 - std_a)) + torch.mean(F.relu(1.0 - std_b))

        # 3. 协方差去相关损失 (Covariance Loss)
        z_a_cent = z_a - z_a.mean(dim=0)
        z_b_cent = z_b - z_b.mean(dim=0)

        cov_a = (z_a_cent.t() @ z_a_cent) / (N - 1) # (D, D)
        cov_b = (z_b_cent.t() @ z_b_cent) / (N - 1)

        # 提取所有非对角线元素
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z_a.device)
        cov_loss = (cov_a[off_diag_mask].pow(2).sum() / D) + (cov_b[off_diag_mask].pow(2).sum() / D)

        total_loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        metrics = {
            "sim_loss": sim_loss.item(),
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
            "mean_std": 0.5 * (std_a.mean() + std_b.mean()).item()
        }
        return total_loss, metrics

class MultiBlockMaskGenerator:
    """
    I-JEPA 风格的多尺度大块状空间掩码生成器
    """
    def __init__(self, grid_size: tuple = (14, 14), num_targets: int = 4, target_scale: tuple = (0.15, 0.2)):
        self.gh, self.gw = grid_size
        self.num_targets = num_targets
        self.target_scale = target_scale

    def sample_masks(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        生成一个大上下文区域与多个互不重叠的目标块掩码索引
        :return: (context_indices, list_of_target_indices)
        """
        total_patches = self.gh * self.gw
        target_masks = []
        all_target_indices = set()

        for _ in range(self.num_targets):
            # 随机确定块高度与宽度
            block_h = max(2, int(self.gh * (self.target_scale[0] ** 0.5)))
            block_w = max(2, int(self.gw * (self.target_scale[0] ** 0.5)))

            top = torch.randint(0, self.gh - block_h + 1, (1,)).item()
            left = torch.randint(0, self.gw - block_w + 1, (1,)).item()

            block_indices = []
            for r in range(top, top + block_h):
                for c in range(left, left + block_w):
                    idx = r * self.gw + c
                    block_indices.append(idx)
                    all_target_indices.add(idx)

            target_masks.append(torch.tensor(block_indices, dtype=torch.long))

        # 上下文掩码为所有未被目标块遮挡的剩余索引
        context_indices = [i for i in range(total_patches) if i not in all_target_indices]
        return torch.tensor(context_indices, dtype=torch.long), target_masks

# ===================================================================
# 单元测试与防坍塌机制收敛校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 16
    embed_dim = 32

    # 1. 测试 VICReg 损失
    vicreg = VICRegLoss()
    # 构造两个带有轻度方差不足的特征矩阵
    z1 = torch.randn(batch_size, embed_dim) * 0.5 # 方差偏小
    z2 = z1 + torch.randn_like(z1) * 0.1

    loss, metrics = vicreg(z1, z2)
    print(f"[VICReg Test] 总损失: {loss.item():.4f}")
    print(f"[VICReg Test] 相似度损失: {metrics['sim_loss']:.4f}, 方差惩罚项: {metrics['std_loss']:.4f}, 协方差惩罚项: {metrics['cov_loss']:.4f}")
    print(f"[VICReg Test] 特征平均标准差: {metrics['mean_std']:.4f}")

    assert metrics["std_loss"] > 0, "特征方差不足时未能成功触发方差铰链惩罚！"

    # 2. 测试 I-JEPA 大块掩码采样器
    mask_gen = MultiBlockMaskGenerator(grid_size=(8, 8), num_targets=2, target_scale=(0.2, 0.3))
    ctx_idx, tgt_list = mask_gen.sample_masks()

    print(f"[Mask Test] 总 Patch 数: 64 (8x8)")
    print(f"[Mask Test] 上下文 Patch 数量: {len(ctx_idx)}, 目标块 1 数量: {len(tgt_list[0])}, 目标块 2 数量: {len(tgt_list[1])}")

    assert len(ctx_idx) + len(tgt_list[0]) <= 64, "掩码索引分配溢出！"
    print("✓ VICReg 防坍塌三元损失层与 I-JEPA 大块空间掩码采样器单测全部通过！")
```

---

## 6.2.5 本节小结

回顾本节内容，我们掌握了非生成式特征学习防坍塌的核心生命线：
1. **维度坍塌的数学本质**：无重构约束下神经网络趋向于输出常数零解，必须施加显式外力迫使流形张开；
2. **VICReg 三位一体几何解**：不变性保持语义对齐、方差项撑开超椭球、协方差项消除信息冗余，完美化解了特征退化；
3. **大块掩码语义完形填空**：彻底切断底层像素插值投机，迫使模型在隐空间涌现出跨越广阔时空的通用物理因果理解。
