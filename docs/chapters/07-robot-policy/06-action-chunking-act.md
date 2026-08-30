# 7.6 动作分块与ACT模型

在之前的章节中，我们深入探讨了基于马尔可夫决策过程（MDP）的传统强化学习与模仿学习框架。在最基础的行为克隆（Behavior Cloning, BC）设定中，策略（Policy）网络通常被建模为一个函数映射 $\pi(o_t) \rightarrow a_t$，即在时间步 $t$ 根据当前观测 $o_t$ 预测下一个离散或连续的动作 $a_t$。然而，当我们试图将这些算法部署到拥有几十个自由度的双臂机器人，并执行诸如穿针引线、剥大蒜等高精度微操时，传统的单步预测模型往往会遭遇灾难性的失败。

这种失败的根源之一在于**复合误差（Compounding Errors）**。在 [[Ross et al., 2011]](https://proceedings.mlr.press/v15/ross11a.html) 提出的 DAgger 算法中，我们就曾讨论过，由于测试时模型自身的微小误差会导致状态偏离训练数据分布，这种偏离在时间上不断累积，最终使得系统崩溃。为了解决这一在真实物理世界中极为棘手的问题，[[Zhao et al., 2023]](https://arxiv.org/abs/2304.13705) 提出了动作分块Transformer（Action Chunking Transformer, ACT）。ACT 通过预测未来的**动作序列（Action Chunk）**而非单一动作，并结合时间集成（Temporal Ensembling）与条件变分自编码器（CVAE），在低成本硬件上实现了极其惊艳的精细双臂操作。

在本节中，我们将从最基础的误差累积几何直觉出发，严格推导动作分块与时间集成的数学表达，并详细解析 ACT 模型的 CVAE-Transformer 混合架构及其变分下界推导。

## 7.6.1 复合误差与动作分块的几何直觉

为了理解为何要进行“动作分块”，我们暂且抛开复杂的机器人视觉输入，将问题降维到一个最基础的高中运动学场景：假设我们需要控制一辆小车沿着一条光滑的二次曲线轨道 $y = -x^2$ 行驶。

在传统的单步行为克隆中，模型在时间步 $t$ 接收到当前坐标 $(x_t, y_t)$，并输出一个极其微小的速度矢量（即动作）。如果在某一步，由于传感器的噪声，小车的实际位置比理想曲线偏离了 $0.1$ 个单位，由于模型在训练时从未见过这种偏离状态，它输出的速度矢量极大概率是错误的。这个错误的矢量将使得下一步的偏差扩大为 $0.2$，进而引发指数级的轨迹发散。

在数学上，我们将专家演示轨迹记为一组状态-动作对序列 $\tau^* = \{(s_0^*, a_0^*), (s_1^*, a_1^*), \dots, (s_T^*, a_T^*)\}$。传统行为克隆试图最小化单步预测误差：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s_t^*, a_t^*) \sim \tau^*} \left[ \| a_t^* - \pi_\theta(s_t^*) \|^2 \right]
$$

在这个损失函数下，动作 $a_t^*$ 被视为一个孤立的标量或向量。动作分块（Action Chunking）的思想极其朴素但有效：既然单步预测容易受到高频噪声和局部偏离的干扰，我们何不让模型在每个时间步 $t$，直接预测未来 $k$ 个时间步的完整轨迹片段？

我们将时间长度为 $k$ 的动作块（Chunk）定义为：

$$
A_t = [a_t, a_{t+1}, \dots, a_{t+k-1}] \in \mathbb{R}^{k \times d}
$$

其中 $d$ 为单步动作的维度。此时，策略网络的映射关系变为 $\pi(o_t) \rightarrow A_t$。在小车沿着 $y = -x^2$ 行驶的例子中，这就好比模型不再只指示“下一步往左转一小点”，而是直接抛出一条平滑的“未来一秒的参考运动轨迹”。即便小车当前存在微小偏离，接下来的一组连续动作也能强行将其拉回预定的宏观航线上，从而极大地抑制了高频的抖动和误差的短期爆发。

## 7.6.2 时间集成（Temporal Ensembling）机制

一旦引入了动作分块，一个直观的冲突便随之产生。在时间步 $t$，模型观测到 $o_t$，并预测出动作块 $A_t$。在传统的执行方式下，机器人会开环（Open-loop）地连续执行这 $k$ 个动作，直到时间步 $t+k$ 才进行下一次观测。这种开环执行放弃了中间 $k-1$ 步的反馈，显然是不安全的。

为了实现闭环控制，我们需要在每一个时间步 $t$ 都进行观测和预测。这就带来了一个有趣的现象：对于未来某一特定时间步 $t'$ 的实际物理动作 $a_{t'}$，我们在之前的多个时间步都对其进行过预测。

具体而言，假设块大小为 $k$。对于时刻 $t$ 的动作 $a_t$，它将被包含在以下 $k$ 个历史预测块中：
1. 在 $t-k+1$ 时刻，预测的 $A_{t-k+1}$ 中的最后一个动作：$\hat{a}_t^{(t-k+1)}$
2. 在 $t-k+2$ 时刻，预测的 $A_{t-k+2}$ 中的倒数第二个动作：$\hat{a}_t^{(t-k+2)}$
...
$k$. 在 $t$ 时刻，预测的 $A_t$ 中的第一个动作：$\hat{a}_t^{(t)}$

我们拥有 $k$ 个对同一时刻物理动作的预测值。为了获得最终执行的稳定动作，ACT 引入了时间集成（Temporal Ensembling）。其本质是对这 $k$ 个历史预测进行加权平均。设 $w_i$ 为权重，最终在时刻 $t$ 执行的真实动作 $a_t^{\text{exec}}$ 的标量展开形式为：

$$
a_t^{\text{exec}} = \sum_{i=0}^{k-1} w_i \hat{a}_t^{(t-i)}
$$

在实际工程应用中，通常我们更信任距离当前时刻越近的观测所做出的预测。因此，权重 $w_i$ 通常被设计为指数衰减的指数权重（Exponential Weighting）：

$$
w_i = \frac{e^{-m \cdot i}}{\sum_{j=0}^{k-1} e^{-m \cdot j}}
$$

其中 $m$ 衰减系数，$i$ 表示预测发生的相对时间差（$i=0$ 表示当前步的预测，$i=k-1$ 表示最旧的预测）。这种简单的加权求和，不仅利用了长程轨迹的平滑性，又兼顾了当前最新观测的瞬时响应能力，成为了 ACT 能够在复杂物理交互中保持丝滑动作的核心关键。

## 7.6.3 应对多模态分布：条件变分自编码器（CVAE）

尽管动作分块与时间集成极大缓解了复合误差，但我们在模仿学习中还面临另一个致命威胁：**人类专家数据的多模态性（Multimodality）**。

假设目标是“避开桌面上的水杯并抓取苹果”。人类专家在示范时，有时会从水杯左侧绕过去（左侧轨迹），有时会从右侧绕过去（右侧轨迹）。如果我们强行使用均方误差（MSE）如式该公式去训练一个确定性的深度神经网络，模型在试图同时拟合“向左”和“向右”的数据时，极有可能输出两条轨迹的平均值——即径直撞向水杯。

为了解决多模态问题，ACT 采用了一个极为优雅的架构设计：将条件变分自编码器（Conditional Variational Autoencoder, CVAE）与 Transformer 结合。

> [!NOTE]
> 犹如让一个极其聪明的学生（解码器）去复刻大师的画作（专家轨迹），但大师的思路是多变的，可能是印象派也可能是立体派。我们不让学生直接死记硬背平均特征，而是通过一个极小维度的暗号（隐变量 $Z$）来传递大师当前的具体流派。这个暗号在测试时是从一个标准的抽奖箱（先验分布）中盲抽的；但在训练时，我们允许通过一个偷窥孔（编码器）看着大师的画作来调整抽奖箱中各类暗号的概率（后验分布），从而迫使学生根据不同的暗号画出不同流派的画作，最终实现多模态的精准映射。

我们需要推导 ACT 中 CVAE 的变分下界。令 $O$ 表示当前的视觉与关节状态观测，$A$ 表示我们需要预测的专家动作块（Action Chunk），$Z$ 为表征多模态意图的低维隐变量（Latent Variable）。

我们希望最大化在给定观测 $O$ 下，生成专家动作 $A$ 的条件对数似然：$\log p_\theta(A | O)$。由于边缘化隐变量 $Z$ 在计算上是不可行的，我们引入一个变分分布 $q_\phi(Z | A, O)$ 作为近似后验（即上文比喻中的“偷窥孔”，在训练时能够看到未来的真实动作 $A$）。

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
1. **重构损失（Reconstruction Loss）**：$-\mathbb{E}_{Z \sim q_\phi} [ \log p_\theta(A | Z, O) ]$。通常假设高斯似然，这就退化为了生成动作 $\hat{A}$ 与真实动作 $A$ 的 L1 损失。
2. **正则化损失（Regularization Loss）**：$D_{KL}(q_\phi(Z | A, O) \parallel \mathcal{N}(0, I))$。迫使编码器输出的均值 $\mu$ 和方差 $\sigma^2$ 贴近标准正态分布。

## 7.6.4 Transformer 在 ACT 中的信息流

在明确了 CVAE 的宏观数学目标后，我们来观察 ACT 是如何通过 Transformer 架构实体化该公式的。

ACT 的网络结构严格分为两条支路：
1. **CVAE 编码器（Encoder，仅训练时存在）**：
   - 输入：真实动作序列 $A$（维度 $k \times d$）与通过 ResNet 提取的当前时刻多相机特征及本体感受（Proprioception）特征。
   - 网络：一个标准的 Transformer 编码器。动作 $A$ 和观测 $O$ 拼接后附加位置编码，经过自注意力（Self-Attention）层进行全局信息交互。
   - 输出：额外引入一个特殊的 `[CLS]` token。利用该 token 经过线性层输出隐变量分布的参数 $\mu \in \mathbb{R}^{d_z}$ 和 $\sigma \in \mathbb{R}^{d_z}$。
   - 采样：使用重参数化技巧（Reparameterization Trick）得到 $z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

2. **CVAE 解码器（Decoder，即策略网络）**：
   - 在测试时，直接从标准正态分布中随机采样 $z \sim \mathcal{N}(0, I)$（代表一种特定的执行模态）。
   - 网络：一个 Transformer 解码器。$z$ 被广播并追加到视觉观测特征上作为 `Memory`。解码器的输入 `Query` 是固定不变的位置嵌入（Position Embeddings），长度为 $k$。
   - 计算：通过交叉注意力（Cross-Attention）机制，长度为 $k$ 的 `Query` 不断向包含了隐意图 $z$ 和当前环境观测的 `Memory` 索取信息。
   - 输出：解码器的输出经过多层感知机（MLP）映射回物理动作空间，一次性生成完整的动作块 $\hat{A} \in \mathbb{R}^{k \times d}$。

这种将时序动作直接映射到 Transformer `Query` 序列上的设计，彻底抛弃了自回归（Autoregressive）生成的耗时问题，使得模型能够在一瞬间并行输出高频控制所需的动作流。

## 7.6.5 核心代码实现

下面，我们将通过代码展示 ACT 模型中最关键的变分编码器和基于查询机制的 Transformer 解码器的前向传播过程。

(**定义ACT的核心CVAE与Transformer结构**)

```{.python .input}
#@tab pytorch
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

    def forward(self, obs_features, action_sequence=None):
        """
        前向传播
        参数：
            obs_features: (Seq_len_obs, Batch, Embed_dim) 视觉和本体特征
            action_sequence: (Chunk_size, Batch, Action_dim) 真实动作序列 (仅训练时提供)
        """
        batch_size = obs_features.shape[1]
        
        # 训练时：通过Encoder计算隐变量后验分布
        if action_sequence is not None:
            # 动作空间映射到高维嵌入
            a_embed = self.action_proj(action_sequence) # (Chunk_size, Batch, Embed_dim)
            cls_token = self.cls_embed.expand(-1, batch_size, -1) # (1, Batch, Embed_dim)
            
            # 拼接 [CLS], 动作嵌入, 与当前观测特征
            # 实际ACT中通常还包括绝对位置编码，此处为简洁省略
            enc_input = torch.cat([cls_token, a_embed, obs_features], dim=0)
            
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
            # 测试时：直接从标准正态分布采样
            z = torch.randn(batch_size, self.latent_dim, device=obs_features.device)
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
dummy_actions = torch.randn(chunk_size, batch_size, action_dim)

# 训练时调用
pred_a, mu, logvar = model(dummy_obs, dummy_actions)
print(f"训练模式预测动作维度: {pred_a.shape}") # 预期 (100, 2, 14)

# 测试时调用
pred_a_test, _, _ = model(dummy_obs, None)
print(f"推理模式预测动作维度: {pred_a_test.shape}")
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
from tensorflow.keras import layers

class ACTCore(tf.keras.Model):
    def __init__(self, action_dim=14, chunk_size=100, latent_dim=32, embed_dim=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        
        # 简化的[CLS]初始化
        self.cls_embed = tf.Variable(tf.random.normal((1, 1, embed_dim)))
        self.action_proj = layers.Dense(embed_dim)
        
        # 为保持极简，这里使用 MultiHeadAttention 组装简易Encoder/Decoder块
        self.enc_attn = layers.MultiHeadAttention(num_heads=8, key_dim=embed_dim//8)
        self.enc_ffn = layers.Dense(2048, activation='relu')
        self.enc_out = layers.Dense(embed_dim)
        
        self.latent_proj = layers.Dense(latent_dim * 2)
        self.z_proj = layers.Dense(embed_dim)
        
        self.query_embed = tf.Variable(tf.random.normal((chunk_size, 1, embed_dim)))
        
        self.dec_attn = layers.MultiHeadAttention(num_heads=8, key_dim=embed_dim//8)
        self.dec_ffn = layers.Dense(2048, activation='relu')
        self.dec_out = layers.Dense(embed_dim)
        
        self.action_head = layers.Dense(action_dim)

    def call(self, obs_features, action_sequence=None, training=False):
        # TensorFlow 中通常 batch 为第一维，这里为了兼容上面的维度描述我们转置或显式指定
        # 假设输入同样为 (Seq_len, Batch, Embed_dim) 形式
        batch_size = tf.shape(obs_features)[1]
        
        if action_sequence is not None:
            a_embed = self.action_proj(action_sequence)
            cls_token = tf.tile(self.cls_embed, [1, batch_size, 1])
            enc_input = tf.concat([cls_token, a_embed, obs_features], axis=0)
            
            # 简易 Encoder
            attn_out = self.enc_attn(enc_input, enc_input)
            enc_output = self.enc_out(self.enc_ffn(attn_out)) + attn_out
            
            cls_out = enc_output[0] # (Batch, Embed_dim)
            latent_params = self.latent_proj(cls_out)
            mu, logvar = tf.split(latent_params, 2, axis=-1)
            
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(std))
            z = mu + eps * std
        else:
            z = tf.random.normal((batch_size, self.latent_dim))
            mu, logvar = None, None
            
        z_embed = tf.expand_dims(self.z_proj(z), axis=0)
        memory = tf.concat([z_embed, obs_features], axis=0)
        
        queries = tf.tile(self.query_embed, [1, batch_size, 1])
        
        # 简易 Decoder (Cross Attention)
        dec_attn_out = self.dec_attn(queries, memory)
        dec_output = self.dec_out(self.dec_ffn(dec_attn_out)) + dec_attn_out
        
        pred_actions = self.action_head(dec_output)
        
        return pred_actions, mu, logvar

# 模型测试
batch_size, chunk_size, action_dim = 2, 100, 14
embed_dim = 512
obs_feat_len = 50

model = ACTCore(action_dim=action_dim, chunk_size=chunk_size, embed_dim=embed_dim)
dummy_obs = tf.random.normal((obs_feat_len, batch_size, embed_dim))
dummy_actions = tf.random.normal((chunk_size, batch_size, action_dim))

pred_a, _, _ = model(dummy_obs, dummy_actions)
print(f"TensorFlow 预测动作维度: {pred_a.shape}")
```

## 7.6.6 小结

* 单步行为克隆面临严重的复合误差问题，微小的传感器噪声极易导致系统崩溃。
* **动作分块（Action Chunking）** 通过单次预测未来连续 $k$ 个时间步的轨迹，迫使系统保持局部的平滑性。
* 预测产生的多个重叠动作可以通过**时间集成（Temporal Ensembling）**进行加权平均，在保证响应性的同时大幅提高抗噪能力。
* **条件变分自编码器（CVAE）** 是解决人类专家数据多模态问题的利器，利用隐变量 $Z$ 建模不可观测的人类意图。
* 在 ACT 架构中，Transformer 解码器利用并行生成的 Query 直接向包含视觉观测与隐意图的 Memory 索取信息，优雅且高效地完成了多模态轨迹的生成。

## 7.6.7 练习

1. 考虑式该公式中的时间集成权重。如果我们将衰减系数 $m$ 设为极大的正数（例如趋近于无穷大），模型在推理时对同一时刻动作的决策将如何表现？这等价于哪种传统的控制策略？
   - *提示：观察当 $m \to \infty$ 时，$e^{-m \cdot i}$ 对于不同的 $i$ 衰减速度有多快。这会导致只有哪个特定的预测对最终动作起主导作用？*
2. 如果我们在真实物理系统上不使用时间集成，而是严格按照模型给出的 $k$ 步预测开环执行。这对于硬件计算算力有什么好处？但在什么场景下会极度危险？
   - *提示：考虑如果机器人需要跟踪一个高速随机移动的目标物体（例如飞出的乒乓球），开环执行 $k$ 步（例如0.5秒）会产生什么后果。*
3. 在式该公式中，如果我们将 KL 散度的正则化系数（又称 $\beta$-VAE 的超参数）设置得非常大，会导致编码器学到的 $\mu$ 和 $\sigma$ 发生什么退化现象？这会对解码器重构多模态专家的动作造成什么影响？
   - *提示：回忆 KL 散度极小化时，后验分布 $q_\phi$ 将无限趋近于先验 $\mathcal{N}(0, I)$。这意味着隐变量 $Z$ 将不再包含任何关于具体动作流派的信息。*
