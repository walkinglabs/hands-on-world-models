# 7.9 联合嵌入预测架构 (JEPA) 与世界动作模型 (WAM)

在前面的章节中，我们学习了如何通过视觉-语言-动作模型（VLA）让机器人“看懂人类指令并执行动作”。然而，现有的端到端策略在本质上仍然主要依赖“当前观测到动作的单向映射”，它们缺少一种人类所独有的核心认知能力——**在脑海中推演未来的物理因果演变**。

当我们站在桌前准备推倒一个水杯时，我们在动手之前就已经在脑海中“预演”了水杯倒下、水花洒出的物理后果。如果我们想让机器人在复杂的物理世界中具备自主规划、避障与纠错的能力，机器人就必须在内心建立一个**世界模型（World Model）**。

然而，如何构建一个高效的世界模型？早期的方法试图在“像素级别”去预测未来的视频画面（例如预测下一秒摄像头的每一颗像素颜色），但这需要消耗天文数字级的算力，且极易被背景中飘动的窗帘、光影的微小晃动等无关噪点干扰。

图灵奖得主 Yann LeCun 提出的**联合嵌入预测架构（Joint Embedding Predictive Architecture, JEPA）**彻底颠覆了这一思路：不在杂乱的像素空间重构画面，而是在高度抽象的“隐空间（Latent Space）”中直接预测未来的物理表征。结合机器人物理控制，这一体系演进为了前沿的**世界动作模型（World Action Models, WAM）**。

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/vjepa-fig1.png" alt="LeCun 提出的三类自监督学习范式：自回归、自编码与联合嵌入预测架构。" width="86%">

_图 7.9-1：三类自监督学习范式：自回归（生成标记）、掩码自编码（像素重构）与联合嵌入预测架构（表征空间预测）。 出处：[A Path Towards Autonomous Machine Intelligence，Yann LeCun，2022](https://openreview.net/forum?id=BZ5a1r-kVsf)。_

</div>

---

## 7.9.1 物理与认知基石：人类大脑的抽象想象力与表征预测演进

要理解 JEPA 的革命性，我们首先需要从人类大脑如何理解物理世界的认知机制讲起。

### 1. 人脑的“抽象世界模型”与像素生成的算力灾难
在日常生活中，当你想把一个玻璃水杯推向桌沿时，你的大脑绝不会像一台 3D 渲染引擎那样，去逐个计算杯子反光表面数百万颗像素的 RGB 颜色变化。你脑海中推演的，是高度提炼的**抽象物理概念**：
- “我的手指施加推力 $\to$ 杯子向前平移 $\to$ 离开桌面支撑面 $\to$ 受重力加速下坠 $\to$ 落地碎裂”。

人类大脑的神经元天然具备“过滤细枝末节、抓住物理本质”的能力。
与之相反，经典的生成式世界模型（如基于扩散模型或视频自回归的方案）试图预测下一帧完整的 $1024 \times 1024$ 图像。这在计算上面临无法克服的物理瓶颈：
1. **巨大的算力浪费**：一张高分辨率图像包含数百万个像素，绝大多数算力被浪费在生成桌面的木质纹理、背景中树叶的微小晃动等与操作任务完全无关的高频细节上；
2. **误差的级联发散**：在像素级别推演多步后，细微的画面模糊会迅速滚雪球般累积，导致画面迅速崩坏为毫无物理意义的色块。

### 2. 从 I-JEPA、V-JEPA 到具身 WAM
针对这一痛点，Meta AI 先后推出了针对静态图像的 **I-JEPA**（Image-JEPA, 2023）与针对动态视频的 **V-JEPA**（Video-JEPA, 2024）。而在 2026 年，结合机器人多模态指令与连续动作演变的 **世界动作模型（World Action Models, WAM）** 进一步将 JEPA 推广到了具身物理交互的全闭环中。

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/ijepa-fig2.png" alt="I-JEPA 在抽象表征空间中用上下文特征预测被掩码的目标块特征。" width="86%">

_图 7.9-2：I-JEPA 在抽象表征空间中用上下文特征预测被掩码的目标块特征。 出处：[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture，Mahmoud Assran et al.，2023](https://arxiv.org/abs/2301.08243)。_

</div>

---

## 7.9.2 核心数学推导一：JEPA 隐空间预测与能量损失函数

在数学上，JEPA 不去计算像素差异，而是将当前状态与未来状态分别投影到连续的向量空间中，直接度量特征向量之间的空间距离。

### 1. 编码器与预测器的前向推演
设机器人在 $t$ 时刻接收到当前工作台图像 $x_t$ 与自然语言指令 $l$。
1. **上下文编码器（Context Encoder, $E_\theta$）**：将输入的多模态观测映射为低维的当前状态特征向量 $\mathbf{s}_t \in \mathbb{R}^{D_s}$：
   $$\mathbf{s}_t = E_\theta(x_t, l)$$
2. **动作条件预测器（Predictor, $P_\phi$）**：接收当前状态 $\mathbf{s}_t$ 与机器人即将执行的动作向量 $\mathbf{a}_t$，在隐空间中向前推演一步，预测出下一时刻的未来状态特征 $\hat{\mathbf{s}}_{t+1} \in \mathbb{R}^{D_s}$：
   $$\hat{\mathbf{s}}_{t+1} = P_\phi(\mathbf{s}_t, \mathbf{a}_t)$$
3. **目标编码器（Target Encoder, $E_{\bar{\theta}}$）**：将环境在下一时刻实际产生的真实图像 $x_{t+1}$ 提取为真实目标特征向量 $\mathbf{s}_{t+1} \in \mathbb{R}^{D_s}$：
   $$\mathbf{s}_{t+1} = E_{\bar{\theta}}(x_{t+1}, l)$$

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/wam-fig1.png" alt="上下文编码器和预测器接受梯度更新，目标编码器使用停止梯度与 EMA" width="86%">

_图 7.9-3：上下文编码器和预测器接受梯度更新，目标编码器使用停止梯度与 EMA。_

</div>

系统的训练目标，是让预测器算出的未来特征 $\hat{\mathbf{s}}_{t+1}$ 与真实未来特征 $\mathbf{s}_{t+1}$ 之间的**欧氏距离均方误差（MSE）**尽可能小：

$$\mathcal{L}_{\text{JEPA}}(\theta, \phi) = \frac{1}{D_s} \|\hat{\mathbf{s}}_{t+1} - \mathbf{s}_{t+1}\|_2^2 = \frac{1}{D_s} \sum_{k=1}^{D_s} \left(\hat{s}_{t+1, k} - s_{t+1, k}\right)^2$$

**手算代入算例**：
设特征向量维度 $D_s = 4$。
预测器输出的预测状态为 $\hat{\mathbf{s}}_{t+1} = [0.8, -0.5, 1.2, 0.0]^\top$；
目标编码器提取的真实未来状态为 $\mathbf{s}_{t+1} = [1.0, -0.5, 0.8, 0.0]^\top$。

我们计算两者的均方误差损失：
1. 逐元素差值：
   $$\hat{\mathbf{s}}_{t+1} - \mathbf{s}_{t+1} = [0.8 - 1.0, -0.5 - (-0.5), 1.2 - 0.8, 0.0 - 0.0]^\top = [-0.2, 0.0, 0.4, 0.0]^\top$$
2. 计算各分量平方和：
   $$\|\hat{\mathbf{s}}_{t+1} - \mathbf{s}_{t+1}\|_2^2 = (-0.2)^2 + 0.0^2 + 0.4^2 + 0.0^2 = 0.04 + 0.00 + 0.16 + 0.00 = 0.20$$
3. 计算均方误差损失：
   $$\mathcal{L}_{\text{JEPA}} = \frac{0.20}{4} = 0.05$$

这个计算过程极其简单清爽：整个系统完全不需要反向传播庞大的像素渲染梯度，仅需对 4 个实数标量的均方差求导，计算速度比传统的视频生成模型快了数个数量级！

<details>
<summary><b>深入推导：自监督表征学习中的平凡解坍塌（Representation Collapse）与信息熵下界分析（点击展开查看完整推导）</b></summary>

在联合嵌入架构中，如果不加结构约束，网络会轻易找到一个毫无物理意义的**平凡解（Trivial Collapse）**：
令编码器将所有输入图像恒等映射为一个全零常数向量 $E_\theta(x) \equiv \mathbf{0}$，预测器也恒等输出 $P_\phi \equiv \mathbf{0}$。
此时均方误差损失恒等于 0：
$$\mathcal{L}_{\text{JEPA}} = \|\mathbf{0} - \mathbf{0}\|_2^2 = 0$$
为了从信息论上杜绝坍塌，表征分布必须满足信息熵最大化约束。设特征向量的协方差矩阵为 $\mathbf{C} = \frac{1}{N}\sum_{i=1}^N (\mathbf{s}_i - \bar{\mathbf{s}})(\mathbf{s}_i - \bar{\mathbf{s}})^\top$。
理想的无损表征要求协方差矩阵对角线方差大于阈值 $\text{Var}(s_k) \ge 1$，且非对角线协方差尽量为零（去除冗余相关性），从而保证隐空间流形具有充分的信息表达能力。
</details>

---

## 7.9.3 核心数学推导二：非对称架构与指数移动平均（EMA）

为了彻底封死上述“全零平凡解坍塌”的漏洞，JEPA 引入了精妙的**非对称双网络架构**与**指数移动平均（Exponential Moving Average, EMA）**机制。

### 1. 停止梯度与参数慢动作更新
如果上下文编码器 $\theta$ 与目标编码器 $\bar{\theta}$ 共享同一套权重并通过梯度反向传播共同更新，两个网络就会迅速“串通作弊”，一起退化为常数输出。

因此，JEPA 做出了两项关键设计：
1. **目标网络停止梯度（Stop-Gradient）**：目标编码器 $E_{\bar{\theta}}$ 彻底脱离自动求导计算图，反向传播的梯度绝对不会流入 $\bar{\theta}$；
2. **指数移动平均慢速追踪**：目标网络的参数 $\bar{\theta}$ 不通过梯度下降更新，而是在每个训练步（Step）结束时，像慢动作滚雪球一样缓慢吸收在线上下文网络 $\theta$ 的参数：

$$\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta$$

<div align="center">

<img src="/figures/07-robot-policy/latex/09-vla-jepa-wam/ema-target-update.png" alt="旧目标参数和在线参数按 EMA 权重合成新目标参数" width="86%">

_图 7.9-5：目标参数不由当前损失反向更新，而是把旧目标与当前在线参数按 τ 和 1−τ 做跨步平滑。_

</div>

> **公式符号逐一拆解**：
> - $\tau \in [0.99, 0.999]$：**动量衰减系数（Momentum Rate）**，通常设为一个非常接近 1 的数值（例如 $0.996$）；
> - $\theta$：正在接收梯度更新的在线上下文网络权重；
> - $\bar{\theta}$：缓慢演化的目标网络权重。

**物理直觉**：由于 $\tau = 0.996$，在每一个训练步中，目标网络有 $99.6\%$ 保留自己原本的历史记忆，仅吸收 $0.4\%$ 来自在线网络的新知识。这使得目标特征 $\mathbf{s}_{t+1}$ 在训练过程中表现为一个极其平稳、缓慢移动的“定海神针”，迫使在线编码器与预测器必须扎扎实实地学习提取有意义的物理运动表征，彻底瓦解了瞬间坍塌为常数解的作弊通道。

<details>
<summary><b>深入推导：EMA 动量参数平滑更新的频域低通滤波与流形稳定性证明（点击展开查看完整推导）</b></summary>

在连续时间极限下，EMA 离散递推式 $\bar{\theta}_{k} = \tau \bar{\theta}_{k-1} + (1 - \tau) \theta_k$ 可以等价表示为一阶线性常微分方程（一阶低通滤波器）：
$$\frac{d\bar{\theta}(t)}{dt} = \frac{1}{\Delta t_{\text{eff}}} (\theta(t) - \bar{\theta}(t)), \quad \text{其中 } \Delta t_{\text{eff}} = \frac{1}{1 - \tau}$$
对该微分方程在频域进行拉普拉斯变换，系统传递函数为：
$$H(s) = \frac{\bar{\Theta}(s)}{\Theta(s)} = \frac{1}{1 + \Delta t_{\text{eff}} s}$$
该系统在频域具有 $-20\text{ dB/dec}$ 的衰减斜率，能够完全滤除随机梯度下降（SGD）在小批量采样时引入的高频参数振荡噪声。这保证了目标特征在隐空间流形上的演化轨迹处处连续且平滑，为预测器提供了绝对稳定的回归几何靶标。
</details>

---

## 7.9.4 基于 Transformer 的多模态时空架构映射

在现代机器人世界模型实现中，上述标量特征向量被自然延展为 Vision Transformer（ViT）下的时空词元序列。

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/wam-fig5.png" alt="级联 WAM 的显式动作、隐式动作与几何提取三种结构承担不同接口角色。" width="86%">

_图 7.9-6：级联 WAM 的显式动作、隐式动作与几何提取三种结构承担不同接口角色。 出处：[World Action Models: The Next Frontier in Embodied AI，Siyin Wang et al.，2026](https://arxiv.org/abs/2605.12090)。_

</div>

在 **世界动作模型（WAM）** 架构中：
1. **视觉图像与语言指令**：图像被切分为 $N$ 个图像块 Patch，经多层自注意力机制（Self-Attention）与语言词元深度融合，生成当前场景的时空状态表征 $\mathbf{S}_t \in \mathbb{R}^{N \times D_s}$；
2. **动作前置与因果推演**：机器人的连续物理动作 $\mathbf{a}_t$ 被映射为一个动作词元，并置于序列前端。预测器通过多层带有**因果掩码（Causal Mask）**的 Transformer，在不接触真实下一帧像素的前提下，直接预测下一时刻所有图像块的特征分布。

---

## 7.9.5 纯底层 PyTorch 代码实现：从零搭建 VLA-JEPA 架构

下面我们使用纯底层 PyTorch 算子手写实现一个结构完整的 VLA-JEPA 模型，包含视觉-语言多模态编码器、动作条件预测器与 EMA 动量目标网络。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class VisionLanguageEncoder(nn.Module):
    """
    视觉-语言上下文编码器 (Context Encoder)
    将高维图像特征与语言指令投影并融合为一个紧凑的状态表征向量 s_t
    """
    def __init__(self, img_dim: int = 1024, lang_dim: int = 512, latent_dim: int = 256):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.lang_proj = nn.Linear(lang_dim, latent_dim)

        # 深度特征融合网络
        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, img_feat: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        """
        :param img_feat: (B, img_dim) 视觉特征
        :param lang_emb: (B, lang_dim) 语言指令嵌入
        :return: (B, latent_dim) 状态表征 s_t
        """
        x_proj = F.gelu(self.img_proj(img_feat))
        l_proj = F.gelu(self.lang_proj(lang_emb))
        fused = torch.cat([x_proj, l_proj], dim=-1)
        s_t = self.fusion_mlp(fused)
        return s_t

class ActionPredictor(nn.Module):
    """
    动作条件状态预测器 (Action-Conditioned Predictor)
    在隐空间中计算前向演变: s_hat_{t+1} = Predictor(s_t, a_t)
    """
    def __init__(self, latent_dim: int = 256, act_dim: int = 64):
        super().__init__()
        self.act_proj = nn.Linear(act_dim, latent_dim)
        self.predictor_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        :param s_t: (B, latent_dim) 当前隐状态
        :param a_t: (B, act_dim) 物理动作控制量
        :return: (B, latent_dim) 预测的下一时刻隐状态
        """
        a_proj = F.gelu(self.act_proj(a_t))
        combined = torch.cat([s_t, a_proj], dim=-1)
        s_next_pred = self.predictor_net(combined)
        return s_next_pred

class VLA_JEPA(nn.Module):
    """
    完整的 VLA-JEPA 顶层架构
    包含在线网络 (接受梯度)、预测器与 EMA 慢速目标网络 (停止梯度)
    """
    def __init__(
        self,
        img_dim: int = 1024,
        lang_dim: int = 512,
        act_dim: int = 64,
        latent_dim: int = 256,
        ema_tau: float = 0.996
    ):
        super().__init__()
        self.ema_tau = ema_tau

        # 1. 在线上下文编码器 (接受反向传播梯度)
        self.context_encoder = VisionLanguageEncoder(img_dim, lang_dim, latent_dim)

        # 2. 在线动作预测器 (接受反向传播梯度)
        self.predictor = ActionPredictor(latent_dim, act_dim)

        # 3. 目标编码器 (深拷贝自上下文编码器，关闭梯度计算)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """
        核心函数：执行目标编码器的指数移动平均 (EMA) 更新
        theta_bar <- tau * theta_bar + (1 - tau) * theta
        """
        for param_online, param_target in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_target.data.mul_(self.ema_tau).add_((1.0 - self.ema_tau) * param_online.data)

    def forward(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        lang_emb: torch.Tensor,
        a_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向计算与均方差损失求解
        """
        # 1. 在线编码器提取当前状态特征
        s_t = self.context_encoder(x_t, lang_emb)

        # 2. 预测器在隐空间推演下一时刻特征
        s_next_pred = self.predictor(s_t, a_t)

        # 3. 目标编码器在无梯度环境下提取真实下一时刻目标特征
        with torch.no_grad():
            s_next_target = self.target_encoder(x_next, lang_emb)

        # 4. 计算隐空间特征均方误差损失
        loss = F.mse_loss(s_next_pred, s_next_target)

        return loss, s_next_pred, s_next_target

# ===================================================================
# 单元测试与动量参数追踪
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    img_dim = 1024
    lang_dim = 512
    act_dim = 64
    latent_dim = 256

    model = VLA_JEPA(
        img_dim=img_dim,
        lang_dim=lang_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        ema_tau=0.99
    )

    dummy_x_t = torch.randn(batch_size, img_dim)
    dummy_x_next = torch.randn(batch_size, img_dim)
    dummy_lang = torch.randn(batch_size, lang_dim)
    dummy_act = torch.randn(batch_size, act_dim)

    # 1. 测试前向损失计算
    loss, s_pred, s_target = model(dummy_x_t, dummy_x_next, dummy_lang, dummy_act)
    print(f"[VLA-JEPA Test] 隐状态预测张量形状: {s_pred.shape}")
    print(f"[VLA-JEPA Test] 目标状态张量形状: {s_target.shape}")
    print(f"[VLA-JEPA Test] 隐空间 MSE 损失: {loss.item():.4f}")

    assert s_pred.shape == (batch_size, latent_dim), "预测特征维度不符！"
    assert s_target.shape == (batch_size, latent_dim), "目标特征维度不符！"

    # 2. 测试梯度反向传播与 EMA 更新
    optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3
    )
    loss.backward()
    optimizer.step()

    # 记录更新前目标网络参数
    old_target_param = next(model.target_encoder.parameters()).clone()
    model.update_target_encoder()
    new_target_param = next(model.target_encoder.parameters())

    param_shift = (new_target_param - old_target_param).abs().max().item()
    print(f"[VLA-JEPA Test] 目标网络 EMA 参数单步平滑位移: {param_shift:.6e}")
    assert param_shift > 0.0, "EMA 参数未发生平滑更新！"
    print("✓ VLA-JEPA 模型前向推演与动量更新单测全部通过！")
```

---

## 7.9.6 本节小结

回顾本节内容，我们建立了一条从像素级重构走向隐空间因果推演的世界模型认知闭环：
1. **表征预测的物理哲学**：打破高消耗、易发散的像素视频生成范式，JEPA 在抽象嵌入空间中直接预测未来物理状态；
2. **防坍塌的非对称动力学**：停止梯度（Stop-Gradient）与指数移动平均（EMA）构成了一对非对称互锁机制，保证了目标特征的平稳性与流形信息丰富度；
3. **世界动作模型（WAM）的具身前景**：将动作作为因果条件置于时序预测流的前端，使机器人在真正触碰现实前，在内心完成千百次安全的物理推演。
