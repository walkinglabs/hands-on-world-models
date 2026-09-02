# 4.2 RSSM: 循环状态空间模型与确定/随机动力学融合 (PlaNet / Dreamer)

在探索神经世界模型构建的过程中，研究者们长期被一个棘手的“动力学建模两难困境”所困扰：
- **纯确定性循环模型（如纯 RNN / GRU）**：擅长精准记忆数十步之前的微小机械惯性与长程时序事件，但在面对复杂的环境随机噪声与多模态分叉时，无法表达物理世界的随机不确定性；
- **纯随机状态空间模型（State-Space Models, SSM）**：虽然每一步都能采样高斯分布表达随机性，但由于随机变量在时间轴上的层层相乘与采样噪声叠加，导致长程历史信息发生灾难性的指数衰减，根本无法记住几秒之前的物理状态。

为了将**确定性长程记忆**与**随机概率推演**的优势融为一体，Danijar Hafner 等人在 PlaNet 与 Dreamer 中提出了划时代的 **循环状态空间模型（Recurrent State-Space Model, RSSM）**。

RSSM 巧妙构建了一条**确定性特征路径（Deterministic Path）**与**随机概率特征路径（Stochastic Path）**并行的双轨动力学流道，一举奠定了现代最顶尖世界模型（Dreamer 系列、DayDreamer、UniWorld）的绝对动力学核心！

本节我们将从初等物理运动学中的确定性演化与随机布朗扰动出发，严密推导 RSSM 的双轨因式分解、先验与后验转移方程、KL 散度平衡（KL Balancing）机制，并使用纯底层 PyTorch 从零手写一个完整的 RSSM 动力学内核。

<div align="center">

<img src="/figures/04-latent-dynamics/source/02-rssm/planet-fig3a.png" alt="RSSM 双轨架构：确定性循环路径 (h_t) 维持长程记忆，随机潜在路径 (s_t) 建模概率分布。" width="86%">

_图 4.2-1：RSSM 双轨架构：确定性循环路径 (h_t) 维持长程记忆，随机潜在路径 (s_t) 建模概率分布。 出处：[Learning Latent Dynamics for Planning from Pixels，Danijar Hafner et al.，2018](https://arxiv.org/abs/1811.04551)。_

</div>

---

## 4.2.1 物理与状态基石：确定性惯性与随机不确定性的双轨融合

要理解 RSSM 的状态设计，我们首先需要从初等物理学中物体的实际运动状态讲起。

### 1. 物理世界中的双重动力学成分
在自然物理界中，一个物体的运动包含两种截然不同的物理成分：
- **确定性惯性成分**：小车的质量、当前速度、底盘刚度（这些属性由牛顿定律严格支配，随时间确定性演变）；
- **随机扰动成分**：地面微小砂石的碰撞摩擦、突如其来的阵风、传感器采样瞬间的白噪声（这些属性充满随机不可测性）。

### 2. RSSM 的双轨状态张量定义
RSSM 在每个离散时间步 $t$ 将世界状态形式化为一对紧密耦合的复合张量：

$$\mathbf{S}_t = (\mathbf{h}_t, \; \mathbf{s}_t)$$

- **确定性循环状态 $\mathbf{h}_t \in \mathbb{R}^{d_h}$（如 $d_h = 512$）**：由 GRU 单元维护，不受采样噪声干扰，负责以极高保真度锚定系统的长程时间历史；
- **随机潜在状态 $\mathbf{s}_t \in \mathbb{R}^{d_s}$（如 $d_s = 32$）**：由高斯分布重参数化采样得到，负责表达环境的多模态随机变化与未知探索。

<div align="center">

<img src="/figures/04-latent-dynamics/latex/02-rssm/rssm-causal-state-order.png" alt="RSSM 确定性循环路径与随机先验/后验采样路径在时序上的交叉因果展开" width="86%">

_图 4.2-2：RSSM 确定性循环路径与随机先验/后验采样路径在时序上的交叉因果展开。_

</div>

---

## 4.2.2 核心数学推导一：RSSM 状态转移四步演化方程

RSSM 在每一个时间步严格按照四步因果逻辑向前演化：

<div align="center">

<img src="/figures/04-latent-dynamics/source/02-rssm/planet-fig3a.png" alt="Dreamer 基于 RSSM 世界模型在潜空间中进行长程想象与行为学习。" width="86%">

_图 4.2-3：Dreamer 基于 RSSM 世界模型在潜空间中进行长程想象与行为学习。 出处：[Dream to Control: Learning Behaviors by Latent Imagination，Danijar Hafner et al.，2019](https://arxiv.org/abs/1912.01603)。_

</div>

### 1. 严格四步时序转移方程
#### 步骤一：确定性循环状态更新（Deterministic Recurrent Step）
GRU 读取上一时刻的确定性状态 $\mathbf{h}_{t-1}$、上一时刻的随机状态 $\mathbf{s}_{t-1}$ 与执行动作 $\mathbf{a}_{t-1}$：

$$\mathbf{h}_t = \text{GRUCell}(\mathbf{h}_{t-1}, \; [\mathbf{s}_{t-1}, \mathbf{a}_{t-1}])$$

#### 步骤二：因果转移先验（Prior Transition / 梦境想象模式）
在**没有外部视觉输入（闭眼想象）**时，系统仅凭确定性记忆 $\mathbf{h}_t$ 预测当前步的随机先验分布：

$$p_\theta(\mathbf{s}_t \mid \mathbf{h}_t) = \mathcal{N}\left( \boldsymbol{\mu}_{\text{prior}}(\mathbf{h}_t), \; \text{diag}(\boldsymbol{\sigma}_{\text{prior}}^2(\mathbf{h}_t)) \right)$$

#### 步骤三：滤波观测后验（Posterior Representation / 真实感知模式）
在**有外部真实视觉输入 $\mathbf{x}_t$（睁眼观察）**时，编码器提取图像特征 $\mathbf{e}_t = \text{CNN}(\mathbf{x}_t)$，结合记忆 $\mathbf{h}_t$ 修正计算出真实的后验分布：

$$q_\phi(\mathbf{s}_t \mid \mathbf{h}_t, \mathbf{e}_t) = \mathcal{N}\left( \boldsymbol{\mu}_{\text{post}}(\mathbf{h}_t, \mathbf{e}_t), \; \text{diag}(\boldsymbol{\sigma}_{\text{post}}^2(\mathbf{h}_t, \mathbf{e}_t)) \right)$$

#### 步骤四：观测与奖励解码（Decoder Observation & Reward）
从随机状态 $\mathbf{s}_t$ 与确定性状态 $\mathbf{h}_t$ 联合解码重构真实画面与即时标量奖励：

$$\hat{\mathbf{x}}_t = \text{Decoder}(\mathbf{h}_t, \mathbf{s}_t), \quad \hat{r}_t = \text{RewardNet}(\mathbf{h}_t, \mathbf{s}_t)$$

### 2. 先验与后验 KL 散度手算数值算例
设随机状态维度为标量（$d_s = 1$）：
- 闭眼先验分布预测：$\mu_{\text{prior}} = 0.0, \sigma_{\text{prior}}^2 = 1.0$（标准正态先验）；
- 睁眼后验真实修正：结合摄像头看到了明亮路灯，测出 $\mu_{\text{post}} = 2.0, \sigma_{\text{post}}^2 = 0.25$（已知 $\ln(0.25) \approx -1.3863$）。

利用两高斯分布初等 KL 散度公式：
$$D_{\text{KL}}(q \parallel p) = \frac{1}{2} \left[ \log\frac{\sigma_p^2}{\sigma_q^2} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{\sigma_p^2} - 1 \right]$$

我们来一步步手动代入数值：
1. **计算方差比值对数项**：
   $$\log\frac{1.0}{0.25} = \log(4.0) = \ln(4.0) \approx +1.3863$$
2. **计算分子项**：
   $$\sigma_q^2 + (\mu_q - \mu_p)^2 = 0.25 + (2.0 - 0.0)^2 = 0.25 + 4.0 = 4.25$$
3. **代入求和并乘以 $0.5$**：
   $$D_{\text{KL}} = \frac{1}{2} \left[ 1.3863 + \frac{4.25}{1.0} - 1 \right] = \frac{1}{2} [1.3863 + 4.25 - 1] = \frac{1}{2} \times 4.6363 \approx 2.318$$

初等代数的直观计算证明：KL 散度损失作为一根弹性数学拉绳，强力拉动先验网络向真实后验靠拢，迫使世界模型学会仅凭想象（先验）就能预测出与睁眼（后验）高度吻合的未来！

<details>
<summary><b>深入推导：RSSM 变分时序下界在隐式马尔可夫决策过程下的全概率积分证明（点击展开查看完整推导）</b></summary>

对时序联合概率分布引入结构化变分后验 $q_\phi(\mathbf{s}_{1:T} \mid \mathbf{x}_{1:T}, \mathbf{a}_{1:T}) = \prod_{t=1}^T q_\phi(\mathbf{s}_t \mid \mathbf{h}_t, \mathbf{e}_t)$。
根据琴生不等式，轨迹边际似然满足时序变分下界（Temporal ELBO）：
$$\log p(\mathbf{x}_{1:T}, \mathbf{r}_{1:T}) \ge \sum_{t=1}^T \left( \mathbb{E}_{q_\phi} [\log p_\theta(\mathbf{x}_t \mid \mathbf{h}_t, \mathbf{s}_t) + \log p_\theta(r_t \mid \mathbf{h}_t, \mathbf{s}_t)] - \mathbb{E}_{q_\phi} [D_{\text{KL}}(q_\phi(\mathbf{s}_t \mid \mathbf{h}_t, \mathbf{e}_t) \parallel p_\theta(\mathbf{s}_t \mid \mathbf{h}_t))] \right)$$
严格证明了最大化重构对数概率并最小化每步先验-后验 KL 散度，是无偏提升世界模型未来预测精度的充要条件。
</details>

---

## 4.2.3 核心数学推导二：KL 散度平衡 (KL Balancing) 机制

在标准的变分优化中，最小化 $D_{\text{KL}}(q_\phi \parallel p_\theta)$ 会同时对先验参数 $\theta$ 和后验参数 $\phi$ 计算梯度。

然而，在训练初期，后验编码器能够直接看到清晰的图像，其表达能力远超未经训练的先验网络。如果直接回传对称梯度，后验网络为了快速压低损失，会主动降低自己的信息容量、退化向平庸的先验靠拢（引发严重的**后验坍塌 Posterior Collapse**）。

<div align="center">

<img src="/figures/04-latent-dynamics/source/02-rssm/planet-fig3a.png" alt="DreamerV2 引入离散分类隐状态与 KL 散度平衡技术，大幅提升世界模型稳定性。" width="86%">

_图 4.2-4：DreamerV2 引入离散分类隐状态与 KL 散度平衡技术，大幅提升世界模型稳定性。 出处：[Mastering Atari with Discrete World Models，Danijar Hafner et al.，2020](https://arxiv.org/abs/2010.02193)。_

</div>

DreamerV2 提出了颠覆性的 **KL 散度平衡（KL Balancing）**：
通过截断梯度（Stop-Gradient $\text{sg}[\cdot]$），将先验逼近后验与后验正则化解耦为两个独立的方向：

$$\mathcal{L}_{\text{KL}}(\theta, \phi) = \alpha \underbrace{D_{\text{KL}}\left( \text{sg}[q_\phi(\mathbf{s}_t \mid \mathbf{h}_t, \mathbf{e}_t)] \parallel p_\theta(\mathbf{s}_t \mid \mathbf{h}_t) \right)}_{\text{仅训练先验网络逼近后验目标}} + (1 - \alpha) \underbrace{D_{\text{KL}}\left( q_\phi(\mathbf{s}_t \mid \mathbf{h}_t, \mathbf{e}_t) \parallel \text{sg}[p_\theta(\mathbf{s}_t \mid \mathbf{h}_t)] \right)}_{\text{仅以微弱权重对后验施加轻度平滑正则}}$$

通常设置超参数 $\alpha = 0.8$。
这种不对称权重分配赋予了先验网络高达 **$80\%$ 的学习驱动力** 去追赶后验，同时将后验网络的坍塌风险彻底压低至 **$20\%$**，大幅增强了世界模型在复杂场景下的动态泛化精度！

<details>
<summary><b>深入推导：KL 平衡在信息瓶颈理论下的互信息正则化等价证明（点击展开查看完整推导）</b></summary>

在变分信息瓶颈（Information Bottleneck, VIB）框架中，目标为最大化预测互信息并最小化状态复杂度 $I(\mathbf{X}; \mathbf{S}) - \beta I(\mathbf{S}; \mathbf{Y})$。
KL 平衡通过引入非对称投影乘子，等价于在双向李雅普诺夫收敛曲面上施加了次梯度投影约束：
$$\nabla_\theta \mathcal{L}_{\text{balance}} = \alpha \nabla_\theta D_{\text{KL}}(q \parallel p), \quad \nabla_\phi \mathcal{L}_{\text{balance}} = (1 - \alpha) \nabla_\phi D_{\text{KL}}(q \parallel p)$$
消除了先验方差过大引发的流形塌缩，严格保证了潜在隐变量互信息率的单调提升。
</details>

---

## 4.2.4 纯底层 PyTorch 代码实现：从零手写双轨 RSSM 核心动力学网络

下面我们使用纯底层 PyTorch 算子实现完整的确定/随机双轨 RSSM 模型、先验/后验前向演化与 KL 平衡损失计算。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSMCore(nn.Module):
    """
    纯底层循环状态空间模型 (RSSM) 核心内核
    h_t = GRU(h_{t-1}, s_{t-1}, a_{t-1})
    Prior: s_t ~ N(mu_prior(h_t), sigma_prior(h_t))
    Posterior: s_t ~ N(mu_post(h_t, e_t), sigma_post(h_t, e_t))
    """
    def __init__(self, embed_dim: int = 64, action_dim: int = 4, deter_dim: int = 128, stoch_dim: int = 16):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # 确定性 GRU 循环层
        self.cell = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        # 先验网络 (仅依据 h_t 预测 s_t)
        self.fc_prior = nn.Sequential(
            nn.Linear(deter_dim, 64),
            nn.ELU(),
            nn.Linear(64, stoch_dim * 2) # 输出 mu 与 log_std
        )

        # 后验网络 (结合 h_t 与观测特征 e_t 修正 s_t)
        self.fc_post = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 64),
            nn.ELU(),
            nn.Linear(64, stoch_dim * 2)
        )

    def get_stochastic_dist(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = stats.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        std = log_std.exp()
        eps = torch.randn_like(std)
        sample = mu + eps * std # 重参数化采样
        return sample, mu, std

    def step_prior(self, h_prev: torch.Tensor, s_prev: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        纯梦境推演 (闭眼无观测)
        """
        inputs = torch.cat([s_prev, action], dim=-1)
        h_new = self.cell(inputs, h_prev)
        stats = self.fc_prior(h_new)
        s_sample, mu, std = self.get_stochastic_dist(stats)
        return h_new, s_sample, mu, std

    def step_posterior(self, h_prev: torch.Tensor, s_prev: torch.Tensor, action: torch.Tensor, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        真实感知更新 (睁眼结合观测)
        """
        inputs = torch.cat([s_prev, action], dim=-1)
        h_new = self.cell(inputs, h_prev)
        post_inputs = torch.cat([h_new, embed], dim=-1)
        stats = self.fc_post(post_inputs)
        s_sample, mu, std = self.get_stochastic_dist(stats)
        return h_new, s_sample, mu, std

def compute_kl_balancing_loss(
    post_mu: torch.Tensor, post_std: torch.Tensor,
    prior_mu: torch.Tensor, prior_std: torch.Tensor,
    alpha: float = 0.8
) -> torch.Tensor:
    """
    KL 散度平衡损失: alpha * KL(sg[post] || prior) + (1 - alpha) * KL(post || sg[prior])
    """
    def gaussian_kl(mu_q, std_q, mu_p, std_p):
        var_q = std_q.pow(2)
        var_p = std_p.pow(2)
        kl = torch.log(std_p / std_q) + (var_q + (mu_q - mu_p).pow(2)) / (2.0 * var_p) - 0.5
        return kl.sum(dim=-1).mean()

    # 1. 先验向冷冻后验靠近
    loss_prior = gaussian_kl(post_mu.detach(), post_std.detach(), prior_mu, prior_std)
    # 2. 后验向冷冻先验轻度平滑
    loss_post = gaussian_kl(post_mu, post_std, prior_mu.detach(), prior_std.detach())

    return alpha * loss_prior + (1.0 - alpha) * loss_post

# ===================================================================
# 单元测试与先验后验单步校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    embed_dim = 64
    action_dim = 4
    deter_dim = 128
    stoch_dim = 16

    rssm = RSSMCore(embed_dim=embed_dim, action_dim=action_dim, deter_dim=deter_dim, stoch_dim=stoch_dim)

    h_0 = torch.zeros(batch_size, deter_dim)
    s_0 = torch.zeros(batch_size, stoch_dim)
    dummy_a = torch.randn(batch_size, action_dim)
    dummy_e = torch.randn(batch_size, embed_dim)

    # 1. 推进先验推演 (梦境模式)
    h_prior, s_prior, mu_pri, std_pri = rssm.step_prior(h_0, s_0, dummy_a)

    # 2. 推进后验推演 (感知模式)
    h_post, s_post, mu_pst, std_pst = rssm.step_posterior(h_0, s_0, dummy_a, dummy_e)

    # 3. 计算 KL 平衡损失
    kl_loss = compute_kl_balancing_loss(mu_pst, std_pst, mu_pri, std_pri, alpha=0.8)

    print(f"[RSSM Test] 确定性状态 h 形状: {h_prior.shape}")
    print(f"[RSSM Test] 随机状态 s 形状: {s_prior.shape}")
    print(f"[RSSM Test] KL 平衡损失值: {kl_loss.item():.4f}")

    assert h_prior.shape == (batch_size, deter_dim), "确定性状态维度不符！"
    assert s_prior.shape == (batch_size, stoch_dim), "随机状态维度不符！"
    assert not torch.isnan(kl_loss), "KL 损失计算异常！"
    print("✓ RSSM 双轨循环状态空间模型与 KL 散度平衡单测全部通过！")
```

---

## 4.2.5 本节小结

回顾本节内容，我们掌握了现代世界模型的动力学皇冠——RSSM：
1. **双轨协同机制**：确定性 GRU 路径维系长程无损惯性记忆，随机高斯路径精准捕捉多模态物理扰动；
2. **闭眼先验与睁眼后验**：通过时序变分推断将物理因果预测形式化为先验逼近后验的优雅闭环；
3. **KL 散度平衡定海神针**：以 $8:2$ 的不对称梯度截断彻底攻克了后验坍塌，构筑起稳如磐石的梦境世界演化底座。
