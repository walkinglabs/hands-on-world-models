# 3.2　VQ-VAE：离散图像 token

在 3.1 我们得到了一帧帧画面 $x_t$。现在要把每一帧变成 Transformer 好处理的短序列。直接让 Transformer 逐像素生成 $H \times W \times C$ 个值并不合适：序列太长，而且像素之间零散的局部结构也很难被一个个数字保留。

VQ-VAE 解决的办法是：先把一帧画面压成一张低分辨率的「编号网格」。比如 PixelWorld 的 $16\times16$ 图可以变成 $4\times4$ 的 token 网格，每个格子是一个整数编号。这样一来，每帧 Transformer 只要预测 16 个编号，而不是 768 个像素值。

所谓 token，在这个语境里就是一个整数，它指向一张有限码本里的某一项。

## Encoder 把画面压成连续网格

第一步，一个 CNN Encoder 把输入图片 $x$ 变成低分辨率的连续特征图。把 $H\times W$ 的图片下采样成 $h\times w$，每个空间位置是一个 $D$ 维向量 $z_e \in \mathbb{R}^D$：

$$
z_e = \mathrm{Encoder}(x), \qquad z_e \in \mathbb{R}^{h\times w\times D}
$$

以 PixelWorld 为例，$16\times16$ 图片经 Encoder 可以得到 $4\times4\times D$。$h,w$ 小了，序列也就短了，但此时 $z_e$ 还是连续的、无法直接当成编号。

## 码本与最近邻量化

VQ-VAE 准备一张有限的码本 $E \in \mathbb{R}^{K\times D}$，也就是 $K$ 个 $D$ 维「原型向量」。对 Encoder 输出的每个位置向量 $z_e$，在码本里找最近的那个编号：

$$
q(z_e) = E[k^\*], \qquad k^\* = \arg\min_{k} \lVert z_e - E[k] \rVert_2
$$

$k^\*$ 就是这个位置的 token id，取值在 $\{0,1,\dots,K-1\}$。一整帧画面就被写成 $h\times w$ 个这样的编号。我们在 B1 里取 $K=16$、$D=8$，所以一张图变成 16 个 0–15 的整数。

## 不可微的麻烦与 STE

问题来了：$\arg\min$ 这个「选最近」的操作不可微，梯度没法从 $k^\*$ 流回 Encoder。没有梯度，Encoder 就学不动。

直通估计器（Straight-Through Estimator, STE）给了一个近似办法：前向时用离散的码本向量 $q(z_e)=E[k^\*]$，反向时假装这一步是恒等映射，把 Decoder 端算出的梯度直接抄给 Encoder 端的 $z_e$。换句话说，前向离散、反向照抄。

$$
\text{前向：}\ z_q = E[k^\*]; \qquad \text{反向：}\ \frac{\partial \mathcal{L}}{\partial z_e} \approx \frac{\partial \mathcal{L}}{\partial z_q}
$$

注意 STE 只解决「能不能训」的问题，它不负责让码本用得均匀、也不提升生成质量。它是训练路径上的一个补丁。

## 三个损失一起上

VQ-VAE 的总损失由三项组成，缺一不可：

$$
\mathcal{L} = \underbrace{\lVert x - \mathrm{Decoder}(z_q) \rVert_2^2}_{\text{重建}} + \underbrace{\lVert \mathrm{sg}[z_e] - E[k^\*] \rVert_2^2}_{\text{codebook}} + \underbrace{\beta\, \lVert z_e - \mathrm{sg}[E[k^\*]] \rVert_2^2}_{\text{commitment}}
$$

其中 $\mathrm{sg}[\cdot]$ 是 stop-gradient，$\beta$ 是控制 commitment 权重的小常数（常取 $0.25$）。三项各管一件事：

- **重建损失**让 Decoder 能从量化向量还原原图，是唯一的「结果对不对」信号；
- **codebook 损失**把被选中的码本项 $E[k^\*]$ 拉向 Encoder 输出 $z_e$（梯度只更新码本）；
- **commitment 损失**反过来要求 Encoder 输出 $z_e$ 不要离它选中的码本项太远（梯度只更新 Encoder），$\beta$ 控制这一约束的强度。

之所以要分 codebook 和 commitment 两个方向，是因为码本和 Encoder 是两套参数，谁去迁就谁并不明确。分两个损失、各带一个 stop-gradient，就是让码本往 Encoder 靠、Encoder 也往码本靠，两边一起收敛。

## 码本坍缩

训练 VQ-VAE 最常见的故障是码本坍缩：$K$ 个码本项里只有少数几个被反复选中，剩下的从不使用。码本名义上有 $K$ 项，实际容量可能只有两三项，细小物体（比如 PixelWorld 里那 9 个像素的红方块）就可能在量化时被并掉。

坍缩的成因不止一个：Decoder 太强时模型可以靠 Decoder 自己脑补细节、完全忽略 token 是哪个；学习率失衡、数据太单一也会让少数码本项垄断。所以光把 $K$ 调大不一定有用。

B1 的日志因此同时记录三件事：每个 batch 实际用到的码本项数（used codes）、token 使用率的分布、以及长期未被选中的码本项。只有「used codes」接近 $K$，码本才算没浪费。

## 先热身再开 VQ

B1 用了一个实用技巧：先用普通 AE（连续 latent、没有量化）热身几十步，让 Encoder 先学会粗略的视觉特征；再 `initialize_codebook` 用这些已有特征给码本赋初值；最后才打开 VQ 和 STE。这样做在小数据上更稳，避免一开始所有位置都塌到同一个码字。

## tokenizer 的评价不能只看重建

PSNR 或重建 loss 只说明像素层面有多接近。互动视频还要单独检查：HUD 文字、物体边缘、小物体、以及和动作直接相关的区域，量化后是否还在。

一个让人警醒的例子：PixelWorld 的大部分像素是黑色背景，红色方块只占 9 个像素。如果只看平均像素 MSE，一张全黑的图也可能得到不高的误差——方块却已经消失了。所以课程实现会给前景像素更高权重，并单独测量红色物体的中心位置。重建 loss、码本使用率和小物体坐标必须一起报告，缺一不可。

漂亮重建背景的 tokenizer，完全可能把决定碰撞的小球压没。这就是为什么 3.4 的评价要列四组指标，而不仅是一张好看的截图。

## 小结

我们把一帧画面经 Encoder 量化成了离散 token 网格：用最近邻在码本里选编号，用 STE 让这条不可微的路径能训，用 codebook 损失和 commitment 损失让码本与 Encoder 彼此靠拢，并时刻警惕码本坍缩。接下来就要让 Transformer 在这些 token 上做动作条件预测。请看 [3.3 动作条件 Transformer](./03-03-action-transformer.md)，动手实验在 [B1：视频离散化与预测](/labs/route-bc)。
