# 7.6 动作分块与ACT模型

双臂机器人整理衣物时，一只手需要固定布料，另一只手要连续完成抬起、牵引和压平。若策略每次只预测一个很短的动作，上一时刻的微小偏差就可能改变下一时刻看到的画面。ACT（Action Chunking with Transformers）的出发点，是让模型一次预测一段相互协调的动作，并在执行时持续用新观测修正它。

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/act-fig6.png" alt="ALOHA 的六类真实双臂任务展示动作分块所针对的长时精细操作。" width="86%">

_图 7.6-1：ALOHA 的六类真实双臂任务展示动作分块所针对的长时精细操作。 出处：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Tony Z. Zhao et al.，2023](https://arxiv.org/abs/2304.13705)。_

</div>

行为克隆在测试时可能访问训练集中少见的状态，使错误随闭环执行累积；DAgger 通过让专家标注当前策略访问到的状态来缓解这种分布偏移 [[Ross et al., 2011]](https://proceedings.mlr.press/v15/ross11a.html)。ACT 采取不同路线：用条件变分自编码器与 Transformer 一次预测一段动作，并用时间集成平滑重叠预测 [[Zhao et al., 2023]](https://arxiv.org/abs/2304.13705)。原论文在六项真实双臂任务上报告了结果，因此这里不把它写成对所有复合误差的普遍解决方案。

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/smile-fig2.png" alt="SMILe 与传统监督模仿的赛道表现差异展示训练分布外误差如何累积。" width="86%">

_图 7.6-2：SMILe 与传统监督模仿的赛道表现差异展示训练分布外误差如何累积。 出处：[Efficient Reductions for Imitation Learning，Stéphane Ross; Drew Bagnell，2010](https://proceedings.mlr.press/v9/ross10a.html)。_

</div>

本节先解释误差为何会沿闭环累积，再说明动作分块、时间集成，以及 ACT 中 CVAE 与 Transformer 的分工。

## 7.6.1 复合误差与动作分块的几何直觉

为了理解为何要进行“动作分块”，我们暂且抛开复杂的机器人视觉输入，将问题降维到一个最基础的高中运动学场景：假设我们需要控制一辆小车沿着一条光滑的二次曲线轨道 $y = -x^2$ 行驶。

在单步行为克隆中，模型在时间步 $t$ 接收当前坐标 $(x_t, y_t)$，并输出一个速度向量。若传感器噪声使小车偏离示范轨迹，模型就会遇到训练数据中较少出现的状态。之后的动作可能让它进一步偏离，但偏差是否翻倍或呈指数增长取决于系统动力学和策略，不能预先给出固定速率。

在数学上，我们将专家演示轨迹记为一组状态-动作对序列 $\tau^* = \{(s_0^*, a_0^*), (s_1^*, a_1^*), \dots, (s_T^*, a_T^*)\}$。传统行为克隆试图最小化单步预测误差：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s_t^*, a_t^*) \sim \tau^*} \left[ \| a_t^* - \pi_\theta(s_t^*) \|^2 \right]
$$

这个损失只直接约束单步动作。动作分块（Action Chunking）把预测目标改为从时间步 $t$ 开始的 $k$ 个动作：

我们将时间长度为 $k$ 的动作块（Chunk）定义为：

$$
A_t = [a_t, a_{t+1}, \dots, a_{t+k-1}] \in \mathbb{R}^{k \times d}
$$

其中 $d$ 为单步动作维度，策略映射变为 $\pi(o_t) \rightarrow A_t$。在小车例子中，输出不再是一个瞬时转向量，而是一段短期参考轨迹。它为模型提供了动作之间的局部一致性约束，但不会自动把已经偏离的系统拉回示范分布；闭环重规划和数据覆盖仍然重要。

## 7.6.2 时间集成（Temporal Ensembling）机制

一旦引入动作分块，还要决定如何执行。在时间步 $t$，模型观测到 $o_t$ 并预测动作块 $A_t$。若机器人开环连续执行全部 $k$ 个动作，直到 $t+k$ 才重新观测，它就无法利用中间 $k-1$ 步的新反馈，面对快速变化的环境时风险较高。

为了实现闭环控制，我们需要在每一个时间步 $t$ 都进行观测和预测。这就带来了一个有趣的现象：对于未来某一特定时间步 $t'$ 的实际物理动作 $a_{t'}$，我们在之前的多个时间步都对其进行过预测。

具体而言，假设块大小为 $k$。对于时刻 $t$ 的动作 $a_t$，它将被包含在以下 $k$ 个历史预测块中：

1. 在 $t-k+1$ 时刻，预测的 $A_{t-k+1}$ 中的最后一个动作：$\hat{a}_t^{(t-k+1)}$
2. 在 $t-k+2$ 时刻，预测的 $A_{t-k+2}$ 中的倒数第二个动作：$\hat{a}_t^{(t-k+2)}$
   ...
   $k$. 在 $t$ 时刻，预测的 $A_t$ 中的第一个动作：$\hat{a}_t^{(t)}$

我们拥有 $k$ 个对同一时刻物理动作的预测值。为了获得最终执行的稳定动作，ACT 引入了时间集成（Temporal Ensembling）。其本质是对这 $k$ 个历史预测进行加权平均。设 $w_i$ 为权重，最终在时刻 $t$ 执行的真实动作 $a_t^{\text{exec}}$ 的标量展开形式为：

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/act-fig5.png" alt="重叠动作块按时间权重集成，缓解单次预测切换造成的抖动。" width="86%">

_图 7.6-3：重叠动作块按时间权重集成，缓解单次预测切换造成的抖动。 出处：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Tony Z. Zhao et al.，2023](https://arxiv.org/abs/2304.13705)。_

</div>

$$
a_t^{\text{exec}} = \sum_{i=0}^{k-1} w_i \hat{a}_t^{(t-i)}

$$

<div align="center">

<img src="/figures/07-robot-policy/latex/06-action-chunking-act/temporal-ensemble-diagonal.png" alt="从重叠动作块的同一物理时刻对角线取值并按预测年龄加权" width="86%">

_图 7.6-4：同一时刻 t 出现在多个历史动作块的不同位置；时间集成取出这条对角线，并沿预测年龄 i 做归一化加权。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

在实际工程应用中，通常我们更信任距离当前时刻越近的观测所做出的预测。因此，权重 $w_i$ 通常被设计为指数衰减的指数权重（Exponential Weighting）：

$$
w_i = \frac{e^{-m \cdot i}}{\sum_{j=0}^{k-1} e^{-m \cdot j}}
$$

其中 $m$ 是衰减系数，$i$ 表示预测发生的相对时间差（$i=0$ 表示当前步的预测，$i=k-1$ 表示最旧的预测）。这种加权把不同预测时刻对同一物理时刻的估计合并起来。较新的预测更贴近当前观测，较旧的预测则带来一定的时间平滑；实际效果仍取决于块长度、衰减系数和控制频率。

## 7.6.3 应对多模态分布：条件变分自编码器（CVAE）

动作分块与时间集成之外，模仿数据还可能具有**多模态性（Multimodality）**。

假设目标是“避开桌面上的水杯并抓取苹果”。人类专家有时从左侧绕行，有时从右侧绕行。如果用均方误差训练确定性网络，并且观测不足以区分两种意图，条件均值可能落在两条轨迹之间，甚至指向水杯。

ACT 将条件变分自编码器（Conditional Variational Autoencoder, CVAE）与 Transformer 结合，用低维隐变量 $Z$ 表示演示中未被当前观测完全说明的动作风格。训练时，编码器读取真实动作块来估计后验；推理时，原始 ACT 实现把 $Z$ 固定为先验均值 $0$，获得确定性的动作预测，而不是每次随机抽取一种风格。

<div align="center">

<img src="/figures/07-robot-policy/source/06-action-chunking-act/act-fig4.png" alt="ACT 的 CVAE 编码器、Transformer 编解码器与动作查询共同生成动作块。" width="86%">

_图 7.6-5：ACT 的 CVAE 编码器、Transformer 编解码器与动作查询共同生成动作块。 出处：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Tony Z. Zhao et al.，2023](https://arxiv.org/abs/2304.13705)。_

</div>

我们需要推导 ACT 中 CVAE 的变分下界。令 $O$ 表示当前的视觉与关节状态观测，$A$ 表示我们需要预测的专家动作块（Action Chunk），$Z$ 为表征多模态意图的低维隐变量（Latent Variable）。

我们希望最大化在给定观测 $O$ 下生成专家动作 $A$ 的条件对数似然 $\log p_\theta(A | O)$。由于边缘化隐变量 $Z$ 通常难以直接计算，我们引入变分分布 $q_\phi(Z | A, O)$ 作为近似后验。它只在训练时读取真实动作 $A$。

对于任意 $Z$，根据概率链式法则，我们有：

$$
\log p_\theta(A | O) = \log \frac{p_\theta(A, Z | O)}{p_\theta(Z | A, O)}
$$

我们在变分分布 $q_\phi(Z | A, O)$ 的期望下展开该式：

$$
\log p_\theta(A | O) = \mathbb{E}_{Z \sim q_\phi} \left[ \log \frac{p_\theta(A, Z | O)}{q_\phi(Z | A, O)} \right] + \mathbb{E}_{Z \sim q_\phi} \left[ \log \frac{q_\phi(Z | A, O)}{p_\theta(Z | A, O)} \right]
$$

由于等式右侧第二项恰好是 KL 散度（Kullback-Leibler Divergence） $D_{KL}(q_\phi(Z | A, O) \parallel p_\theta(Z | A, O))$，且 KL 散度非负。因此，等式右侧第一项构成了对数似然的证据下界（Evidence Lower Bound, ELBO）：

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{Z \sim q_\phi} \left[ \log p_\theta(A | Z, O) \right] - D_{KL}(q_\phi(Z | A, O) \parallel p(Z | O))
$$

在 ACT 中，先验分布被极度简化为一个与观测无关的标准正态分布 $p(Z | O) = \mathcal{N}(0, I)$。
因此，ACT 的总体训练损失函数转化为最小化负的 ELBO：

1. **重构损失（Reconstruction Loss）**：$-\mathbb{E}_{Z \sim q_\phi}[\log p_\theta(A|Z,O)]$。固定方差高斯似然对应 L2/MSE；ACT 实现采用 L1 动作重构，这是具体的建模选择，不能由高斯假设直接推出。
2. **正则化损失（Regularization Loss）**：$D_{KL}(q_\phi(Z | A, O) \parallel \mathcal{N}(0, I))$。迫使编码器输出的均值 $\mu$ 和方差 $\sigma^2$ 贴近标准正态分布。

## 7.6.4 Transformer 在 ACT 中的信息流

在明确了 CVAE 的宏观数学目标后，我们来观察 ACT 是如何通过 Transformer 架构实体化该公式的。

ACT 的网络结构可以分成两条支路：

1. **CVAE 编码器（Encoder，仅训练时存在）**：
   - 输入：真实动作序列 $A$（维度 $k \times d$）和当前关节位置。原始 ACT 的 CVAE 编码器不读取相机图像；图像由后面的策略网络处理。
   - 网络：Transformer 编码器将动作、关节状态与位置编码共同处理。
   - 输出：额外引入一个特殊的 `[CLS]` token。利用该 token 经过线性层输出隐变量分布的参数 $\mu \in \mathbb{R}^{d_z}$ 和 $\sigma \in \mathbb{R}^{d_z}$。
   - 采样：使用重参数化技巧（Reparameterization Trick）得到 $z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

2. **CVAE 解码器（Decoder，即策略网络）**：
   - 在测试时，原始实现令 $z=0$，也就是采用标准正态先验的均值。
   - 网络：一个 Transformer 解码器。$z$ 被广播并追加到视觉观测特征上作为 `Memory`。解码器的输入 `Query` 是固定不变的位置嵌入（Position Embeddings），长度为 $k$。
   - 计算：通过交叉注意力（Cross-Attention）机制，长度为 $k$ 的 `Query` 不断向包含了隐意图 $z$ 和当前环境观测的 `Memory` 索取信息。
   - 输出：解码器的输出经过多层感知机（MLP）映射回物理动作空间，一次性生成完整的动作块 $\hat{A} \in \mathbb{R}^{k \times d}$。

这种设计不需要逐动作自回归采样，可以并行产生整个动作块。端到端延迟仍由图像编码器、Transformer 规模和硬件决定。

## 7.6.5 核心代码实现

下面，我们将通过代码展示 ACT 模型中最关键的变分编码器和基于查询机制的 Transformer 解码器的前向传播过程。

(**定义ACT的核心CVAE与Transformer结构**)

```python
import torch
from torch import nn
from torch.nn import functional as F

class ACTCore(nn.Module):
    def __init__(self, action_dim=14, chunk_size=100, latent_dim=32, embed_dim=512):
        """
        动作分块Transformer的核心模块。
        此处简化了视觉特征提取器(ResNet)，专注于CVAE与Transformer的集成。
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # 编码器 (q_phi(z | a, o)) 相关模块
        # [CLS] token用于汇总整个动作序列的特征
        self.cls_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.action_proj = nn.Linear(action_dim, embed_dim)

        # 编码器Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 映射到隐变量分布参数
        self.latent_proj = nn.Linear(embed_dim, latent_dim * 2) # 输出 mu 和 logvar

        # 解码器 (p_theta(a | z, o)) 相关模块
        self.z_proj = nn.Linear(latent_dim, embed_dim)

        # 固定的 Query，长度等于 action chunk size
        self.query_embed = nn.Parameter(torch.randn(chunk_size, 1, embed_dim))

        # 解码器Transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=7)

        # 最终映射到物理动作空间
        self.action_head = nn.Linear(embed_dim, action_dim)

    def forward(self, obs_features, joint_feature, action_sequence=None):
        """
        前向传播
        参数：
            obs_features: (Seq_len_obs, Batch, Embed_dim) 策略网络使用的视觉与本体特征
            joint_feature: (1, Batch, Embed_dim) CVAE编码器使用的当前关节特征
            action_sequence: (Chunk_size, Batch, Action_dim) 真实动作序列 (仅训练时提供)
        """
        batch_size = obs_features.shape[1]

        # 训练时：通过Encoder计算隐变量后验分布
        if action_sequence is not None:
            # 动作空间映射到高维嵌入
            a_embed = self.action_proj(action_sequence) # (Chunk_size, Batch, Embed_dim)
            cls_token = self.cls_embed.expand(-1, batch_size, -1) # (1, Batch, Embed_dim)

            # 原始ACT的CVAE编码器使用动作序列与当前关节位置，不读取图像特征
            # 实际ACT中通常还包括绝对位置编码，此处为简洁省略
            enc_input = torch.cat([cls_token, joint_feature, a_embed], dim=0)

            # 通过Transformer Encoder
            enc_output = self.encoder(enc_input)

            # 提取 [CLS] 对应的特征，预测 mu 和 logvar
            cls_out = enc_output[0] # (Batch, Embed_dim)
            latent_params = self.latent_proj(cls_out)
            mu, logvar = torch.split(latent_params, self.latent_dim, dim=1)

            # 重参数化技巧采样 z
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # 原始ACT推理时采用先验均值，得到确定性预测
            z = torch.zeros(batch_size, self.latent_dim, device=obs_features.device)
            mu, logvar = None, None

        # 解码器：将 z 注入观测特征
        z_embed = self.z_proj(z).unsqueeze(0) # (1, Batch, Embed_dim)
        memory = torch.cat([z_embed, obs_features], dim=0) # (Seq_len_obs + 1, Batch, Embed_dim)

        # 构建 Query
        queries = self.query_embed.expand(-1, batch_size, -1) # (Chunk_size, Batch, Embed_dim)

        # 通过Transformer Decoder
        dec_output = self.decoder(tgt=queries, memory=memory) # (Chunk_size, Batch, Embed_dim)

        # 映射回动作空间
        pred_actions = self.action_head(dec_output) # (Chunk_size, Batch, Action_dim)

        return pred_actions, mu, logvar

# 演示前向计算维度
batch_size, chunk_size, action_dim = 2, 100, 14
embed_dim = 512
obs_feat_len = 50 # 假设多个相机的特征展平后的序列长度为50

model = ACTCore(action_dim=action_dim, chunk_size=chunk_size, embed_dim=embed_dim)
dummy_obs = torch.randn(obs_feat_len, batch_size, embed_dim)
dummy_joint = torch.randn(1, batch_size, embed_dim)
dummy_actions = torch.randn(chunk_size, batch_size, action_dim)

# 训练时调用
pred_a, mu, logvar = model(dummy_obs, dummy_joint, dummy_actions)
print(f"训练模式预测动作维度: {pred_a.shape}") # 预期 (100, 2, 14)

# 测试时调用
pred_a_test, _, _ = model(dummy_obs, dummy_joint, None)
print(f"推理模式预测动作维度: {pred_a_test.shape}")
```

## 7.6.6 小结

- 单步行为克隆可能因闭环分布偏移而累积误差，但增长速度取决于具体系统。
- **动作分块**一次预测 $k$ 个动作，为局部动作序列提供一致性。
- **时间集成**融合多个重叠块对同一时刻的预测，在响应速度与平滑性之间折中。
- ACT 的 **CVAE 编码器**在训练时读取动作和关节状态；原始实现推理时令 $z=0$。
- Transformer 策略网络结合图像、本体状态与隐变量，并行输出动作块。

## 7.6.7 练习

1. 考虑时间集成权重 $w_i$。如果把衰减系数 $m$ 设为极大的正数，模型会主要采用哪一个预测块中的动作？
   - _提示：观察当 $m \to \infty$ 时，$e^{-m \cdot i}$ 对于不同的 $i$ 衰减速度有多快。这会导致只有哪个特定的预测对最终动作起主导作用？_
2. 如果我们在真实物理系统上不使用时间集成，而是严格按照模型给出的 $k$ 步预测开环执行。这对于硬件计算算力有什么好处？但在什么场景下会极度危险？
   - _提示：考虑如果机器人需要跟踪一个高速随机移动的目标物体（例如飞出的乒乓球），开环执行 $k$ 步（例如0.5秒）会产生什么后果。_
3. 如果把 KL 散度的正则化系数设得非常大，编码器学到的 $\mu$ 和 $\sigma$ 会发生什么变化？这会怎样影响动作重构？
   - _提示：回忆 KL 散度极小化时，后验分布 $q_\phi$ 将无限趋近于先验 $\mathcal{N}(0, I)$。这意味着隐变量 $Z$ 将不再包含任何关于具体动作流派的信息。_
