# 4.5 DreamerV2/V3: 离散分类隐变量与无超参自适应泛化

在世界模型的进化长河中，如果说 DreamerV1 确立了潜空间梦境强化学习的黄金范式，那么 **DreamerV2** 与 **DreamerV3** 的相继问世，则将世界模型的表达能力与通用泛化性推向了前所未有的顶峰。

在连续高斯分布主导的潜空间中，系统在处理连续机械力矩时表现优异，但在面对自然物理界中大量存在的**离散物理阶跃事件**（如机械卡扣的锁紧瞬间、电路开关的通断跳变、碰撞接触的硬状态切换）时，连续高斯分布往往会产生灾难性的“均值过渡平滑”，导致世界模型对离散状态的记忆迅速模糊崩溃。

为了攻克这一物理建模瓶颈：
- **DreamerV2（2020）** 彻底抛弃了连续高斯分布，开创性地引入了 **离散多分类隐变量（Categorical Latents）** 与 Straight-Through 梯度直通，成为人类历史上首个在雅达利（Atari）游戏基准上全面击败无模型顶级算法的模型化系统；
- **DreamerV3（2023）** 进一步引入了 **对称对数缩放（Symlog）**、**两热分布回归（Two-Hot Encoding）** 与 **动态百分位数归一化** 三大自适应机制，实现了在完全不微调任何超参数的前提下，横跨从连续机械控制、高维像素游戏到庞大 Minecraft 复杂长程任务的“通用世界模型大一统”！

<div align="center">

<img src="/figures/04-latent-dynamics/source/05-dreamer-v2-v3/dreamerv3-fig2.png" alt="DreamerV3 跨越四大不同领域基准测试，展示统一模型架构与固定超参数下的卓越表现。" width="86%">

_图 4.5-1：DreamerV3 跨越四大不同领域基准测试，展示统一模型架构与固定超参数下的卓越表现。 出处：[Mastering Diverse Domains through World Models，Danijar Hafner et al.，2023](https://arxiv.org/abs/2301.04104)。_

</div>

---

## 4.5.1 物理与数学基石：连续概率球与离散逻辑流形的碰撞

要理解离散分类隐变量的妙处，我们首先必须审视连续高斯空间与离散分类空间的表达特性差异。

### 1. 连续高斯空间的“物理平滑陷阱”
假设物理世界发生了一个突变：机械臂夹爪中的物体在时刻 $t$ 是“牢固抓紧”状态（记为 $1$），在时刻 $t+1$ 发生了“彻底滑脱”（记为 $0$）。
若使用连续高斯分布，系统在预测两者之间的潜在转移时，被迫给出均值 $\mu = 0.5$（即处于一种“既抓紧又滑脱”的不可思议非物理中间态）。

### 2. 离散多分类矩阵的巨大组合容量（$32 \times 32$ Categorical Matrix）
DreamerV2 将随机隐状态 $\mathbf{z}_t$ 构建为包含 $32$ 组独立离散分类分布的矩阵，每组包含 $32$ 个离散类别：

$$\mathbf{z}_t \in \{0, 1\}^{32 \times 32}$$

每组分类变量通过 One-Hot 独热向量表达。
其总离散状态组合容量为：
$$32^{32} = (2^5)^{32} = 2^{160} \approx 1.46 \times 10^{48} \text{ 种状态！}$$
这个天文数字级别的离散空间容量比全宇宙的原子的总数还要庞大数十个数量级，既保留了离散逻辑的硬阶跃边界，又具备了与连续空间不相上下的无穷表达精度！

<div align="center">

<img src="/figures/04-latent-dynamics/latex/05-dreamer-v2-v3/twohot-value-preservation.png" alt="DreamerV2 离散分类潜变量矩阵采样与 Straight-Through 梯度直通计算图" width="86%">

_图 4.5-2：DreamerV2 离散分类潜变量矩阵采样与 Straight-Through 梯度直通计算图。_

</div>

---

## 4.5.2 核心数学推导一：离散 Gumbel 采样与 Straight-Through 梯度直通

离散 One-Hot 采样是一个完全不可求导的阶跃操作。DreamerV2 如何让梯度无损地回传给先验和后验网络？

<div align="center">

<img src="/figures/04-latent-dynamics/source/05-dreamer-v2-v3/dreamerv2-fig2.png" alt="DreamerV2 比较连续高斯隐状态与离散分类隐状态下的画面重构与长期预测保真度。" width="86%">

_图 4.5-3：DreamerV2 比较连续高斯隐状态与离散分类隐状态下的画面重构与长期预测保真度。 出处：[Mastering Atari with Discrete World Models，Danijar Hafner et al.，2020](https://arxiv.org/abs/2010.02193)。_

</div>

### 1. Gumbel-Max 离散重参数化采样
设某一组分类变量输出的未归一化对数几率为 $\mathbf{l} = [l_1, l_2, \dots, l_K] \in \mathbb{R}^K$（$K = 32$）。
从标准 Gumbel 分布中独立采样噪声 $g_k = -\ln(-\ln(u_k))$（其中 $u_k \sim \text{Uniform}(0, 1)$）。
离散类别索引通过寻找加噪极大值得到：

$$k^* = \arg\max_{k \in \{1, \dots, K\}} (l_k + g_k)$$

$$\mathbf{z}_{\text{discrete}} = \text{OneHot}(k^*) \in \{0, 1\}^K$$

### 2. 梯度直通估计器（Straight-Through Estimator, STE）
为了打通反向传播计算图，前向输出硬离散 One-Hot 向量，反向传播则将 Softmax 连续概率的梯度无损直通：

$$\mathbf{z}_{\text{final}} = \mathbf{z}_{\text{discrete}} + \text{Softmax}(\mathbf{l}) - \text{sg}[\text{Softmax}(\mathbf{l})]$$

- 前向计算：$\mathbf{z}_{\text{discrete}} + \mathbf{p} - \mathbf{p} = \mathbf{z}_{\text{discrete}}$（完全维持离散硬逻辑）；
- 反向传播：$\frac{\partial \mathbf{z}_{\text{final}}}{\partial \mathbf{l}} = \frac{\partial \text{Softmax}(\mathbf{l})}{\partial \mathbf{l}}$（连续平滑梯度畅通回传给 Logits）！

### 3. STE 直通手算数值算例
设某一维度的 Logits 为 $\mathbf{l} = [1.0, 2.0]^\top$。
Softmax 概率为：
$$p_1 = \frac{e^1}{e^1 + e^2} \approx \frac{2.718}{2.718 + 7.389} \approx 0.269, \quad p_2 = \frac{e^2}{e^1 + e^2} \approx 0.731$$
采样得到的离散 One-Hot 结果为 $\mathbf{z}_{\text{discrete}} = [0, 1]^\top$。

在前向计算中，输出严格为硬离散值 $[0, 1]^\top$；而当上游回传梯度 $\mathbf{G} = [1.0, -1.0]$ 时，反向传播直接作用于连续概率 $\mathbf{p} = [0.269, 0.731]^\top$ 上，彻底化解了离散不可导的数学死锁！

<details>
<summary><b>深入推导：Gumbel-Max 技巧在极值顺序统计量下的严格概率等价证明（点击展开查看完整推导）</b></summary>

设 $G_k \sim \text{Gumbel}(0, 1)$，其累积分布函数为 $F(g) = \exp(-\exp(-g))$。
考察事件 $k^* = \arg\max_k (l_k + G_k)$ 的边缘发生概率：
$$P(k^* = i) = P\left( \bigcap_{j \ne i} \{l_j + G_j \le l_i + G_i\} \right) = \int_{-\infty}^\infty f(g_i) \prod_{j \ne i} F(l_i - l_j + g_i) dg_i$$
代入 Gumbel 概率密度函数 $f(g) = e^{-g} e^{-e^{-g}}$：
$$P(k^* = i) = \int_{-\infty}^\infty e^{-g_i} \exp\left( -e^{-g_i} \sum_{j=1}^K e^{l_j - l_i} \right) dg_i$$
令换元积分变量 $y = e^{-g_i}$，定积分直接化简为初等有理分式：
$$P(k^* = i) = \frac{1}{\sum_{j=1}^K e^{l_j - l_i}} = \frac{e^{l_i}}{\sum_{j=1}^K e^{l_j}} = \text{Softmax}(l_i)$$
严格证得 Gumbel-Max 采样与类别 Softmax 概率在测度论意义下的完全等价性。
</details>

---

## 4.5.3 核心数学推导二：DreamerV3 的无超参自适应三大基石

在面对跨度达数百万倍的极端物理奖励分布时，DreamerV3 提出了三项划时代的自适应数学工具：

<div align="center">

<img src="/figures/04-latent-dynamics/source/05-dreamer-v2-v3/dreamerv3-fig2.png" alt="DreamerV3 对称对数变换 (Symlog) 与动态百分位数归一化自适应机制曲线。" width="86%">

_图 4.5-4：DreamerV3 对称对数变换 (Symlog) 与动态百分位数归一化自适应机制曲线。 出处：[Mastering Diverse Domains through World Models，Danijar Hafner et al.，2023](https://arxiv.org/abs/2301.04104)。_

</div>

### 1. 对称对数变换（Symlog Transformation）
双向可逆平滑压缩函数，将跨越数个数量级的极端物理数值压缩至平稳区间：

$$\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$$

$$\text{symexp}(y) = \text{sign}(y) (\exp(|y|) - 1)$$

### 2. 两热离散分布回归（Two-Hot Categorical Value Representation）
在预测 Critic 价值与即时奖励时，DreamerV3 彻底抛弃了连续标量回归（MSE 损失在面对极端异常值时极易产生梯度爆炸），改用预设在均匀网格桶 $B = \{b_1, b_2, \dots, b_N\}$ 上的离散概率分布。

对于任意标量目标值 $y = \text{symlog}(R)$，若其落在相邻两个桶 $[b_k, b_{k+1}]$ 之间，系统将其精确投影为仅在这两个相邻桶上非零的“两热概率”（Two-Hot Probabilities）：

$$p_k = \frac{b_{k+1} - y}{b_{k+1} - b_k}, \quad p_{k+1} = \frac{y - b_k}{b_{k+1} - b_k}$$

利用离散交叉熵损失进行无爆炸优化，在还原预测值时计算数学期望：

$$\hat{y} = \sum_{j=1}^N \text{Softmax}(l_j) \cdot b_j$$

### 3. 两热投影手算数值算例
设价值网格桶为 $B = [0.0, 10.0, 20.0]$。当前目标标量为 $y = 7.0$。
- 目标值 $7.0$ 落在第 1 桶（$b_1 = 0.0$）与第 2 桶（$b_2 = 10.0$）之间；
- 桶间距 $\Delta = 10.0 - 0.0 = 10.0$；
- 计算两热权重：
  $$p_1 = \frac{10.0 - 7.0}{10.0} = 0.30, \quad p_2 = \frac{7.0 - 0.0}{10.0} = 0.70$$
- 真实两热分布为 $\mathbf{p} = [0.30, 0.70, 0.0]^\top$。

期望还原验证：
$$\mathbb{E}[y] = 0.30 \times 0.0 + 0.70 \times 10.0 + 0.0 \times 20.0 = 0.0 + 7.0 + 0.0 = 7.0$$

初等代数的几步加权完美验证：两热分布在严格保持标量期望无损的同时，将无界连续回归优雅转化为数值绝对稳定的有界分类交叉熵！

<details>
<summary><b>深入推导：两热分类编码在 Wasserstein 距离（EMD）意义下的期望保真度证明（点击展开查看完整推导）</b></summary>

设真实狄拉克测度为 $\delta_y$。在单调网格 $b_1 < \dots < b_N$ 上构造两热概率分布 $P = \sum p_i \delta_{b_i}$。
一阶 Wasserstein 距离（推土机距离）定义为：
$$\mathcal{W}_1(P, \delta_y) = \int_{-\infty}^\infty |F_P(t) - F_{\delta_y}(t)| dt$$
由于仅有相邻两桶 $b_k \le y \le b_{k+1}$ 被赋予非零权重，累积分布函数在区间 $[b_k, b_{k+1}]$ 之外完全恒等为 0 或 1。
积分计算得 $\mathcal{W}_1(P, \delta_y) = p_k(y - b_k) = \frac{(b_{k+1} - y)(y - b_k)}{b_{k+1} - b_k} \le \frac{\Delta b}{4}$。
严格证明了在任意细分网格下，两热编码的信息几何逼近误差严格受限于网格分辨率的一阶上界。
</details>

---

## 4.5.4 纯底层 PyTorch 代码实现：从零手写 DreamerV3 离散分类潜变量与 Symlog 引擎

下面我们使用纯底层 PyTorch 算子手写实现完整的离散多分类采样器、Straight-Through 梯度直通与 Symlog/两热变换模块。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalLatentSampler(nn.Module):
    """
    DreamerV2/V3 离散多分类潜在采样器
    包含 Gumbel-Max 离散采样与 Straight-Through (STE) 梯度直通
    """
    def __init__(self, num_categoricals: int = 32, num_classes: int = 32):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param logits: (B, num_categoricals * num_classes)
        :return: (z_st, z_probs) 形状为 (B, num_categoricals, num_classes)
        """
        B = logits.shape[0]
        logits = logits.view(B, self.num_categoricals, self.num_classes)
        probs = F.softmax(logits, dim=-1)

        # 1. 采样标准 Gumbel 噪声
        uniform_noise = torch.rand_like(probs).clamp(1e-6, 1.0 - 1e-6)
        gumbel_noise = - torch.log(- torch.log(uniform_noise))

        # 2. Gumbel-Max 硬离散 One-Hot
        sample_indices = torch.argmax(logits + gumbel_noise, dim=-1) # (B, num_categoricals)
        one_hot = F.one_hot(sample_indices, num_classes=self.num_classes).float() # (B, num_cat, num_cla)

        # 3. Straight-Through 梯度直通: z = one_hot + probs - sg[probs]
        z_st = one_hot + probs - probs.detach()

        return z_st, probs

class SymlogTools:
    """
    DreamerV3 对称对数数学工具箱
    """
    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    @staticmethod
    def symexp(y: torch.Tensor) -> torch.Tensor:
        return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)

# ===================================================================
# 单元测试与直通梯度流校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    num_cat = 8
    num_cla = 8

    sampler = CategoricalLatentSampler(num_categoricals=num_cat, num_classes=num_cla)

    dummy_logits = torch.randn(batch_size, num_cat * num_cla, requires_grad=True)
    z_st, z_probs = sampler(dummy_logits)

    # 模拟下游损失反向传播
    loss = z_st.sum()
    loss.backward()

    # 测试 Symlog 双向可逆性
    test_val = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    sym_val = SymlogTools.symlog(test_val)
    recovered_val = SymlogTools.symexp(sym_val)

    print(f"[DreamerV3 Test] 离散潜在张量形状: {z_st.shape}")
    print(f"[DreamerV3 Test] 原始数值: {test_val.tolist()}")
    print(f"[DreamerV3 Test] Symlog 压缩后数值: {[round(x, 4) for x in sym_val.tolist()]}")

    assert z_st.shape == (batch_size, num_cat, num_cla), "离散潜在张量形状不符！"
    assert dummy_logits.grad is not None, "STE 梯度直通未成功回传！"
    assert torch.allclose(test_val, recovered_val, atol=1e-3), "Symlog 可逆还原精度异常！"
    print("✓ DreamerV2/V3 离散分类隐变量、STE 直通梯度与 Symlog 对称缩放单测全部通过！")
```

---

## 4.5.5 本节小结

回顾本节内容，我们掌握了通用世界模型大一统的核心基石：
1. **离散分类多流形**：以 $32 \times 32$ 组合矩阵彻底替代连续高斯，攻克了物理阶跃与硬接触建模中的模糊瓶颈；
2. **Straight-Through 优雅求导**：前向硬离散、反向软概率，打通了离散空间的连续梯度反向传播通道；
3. **DreamerV3 无超参三大件**：对称对数（Symlog）、两热回归（Two-Hot）与百分位数归一化彻底降服了数值爆炸，奠定了通用具身世界模型的工业级基石。
