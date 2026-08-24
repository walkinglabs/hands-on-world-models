# 7.3　视觉语言动作模型（VLA）

桌面上同时放着红杯和蓝杯。只看图片，小臂知道物体在哪里，却不知道这次任务要抓哪一个；只看“拿起蓝杯”这句话，又不知道杯子位于桌上的哪个角落。

**VLA**（vision-language-action）模型要做的，就是把视觉、语言和机器人状态三种输入，共同映射到一个动作。所谓 VLA，就是一台同时会看、会听、会动的策略网络。

## 三类输入，三种 token

先把三种输入各自变成模型能消化的表示。图像交给 CNN 或 ViT，得到一组视觉 token；指令交给文本编码器，得到一组语言 token；自身状态交给一个小 MLP，得到一个状态 token。

$$
\text{image}\xrightarrow{\,\text{ViT/CNN}\,}Z_v,\qquad
\text{instruction}\xrightarrow{\,\text{encoder}\,}Z_\ell,\qquad
s_t\xrightarrow{\,\text{MLP}\,}z_s.
$$

其中 $Z_v\in\mathbb{R}^{N_v\times d}$ 是 $N_v$ 个视觉 token，$Z_\ell\in\mathbb{R}^{N_\ell\times d}$ 是 $N_\ell$ 个语言 token，$z_s\in\mathbb{R}^{d}$ 是单个状态 token，三者维度都对齐到 $d$。

接着把这些 token 拼成一串，交给 Transformer。每个 token 都可以注意到另外两类，于是语言可以指向图像里某个物体，图像也可以反过来约束语言。最后由一个 **action head** 输出动作 $\hat a_t$。

```text
[ Z_v ; Z_ℓ ; z_s ] → Transformer → action head → â_t
```

## 多模态融合：交叉注意与拼接

三种 token 互相影响的方式有很多种。最直接的办法是全部拼成一条长序列，让标准自注意力去混合。这叫**早期融合**。

也可以分开处理：视觉 token 自己做几层自注意，再通过**交叉注意**（cross-attention）去查询语言 token。设视觉查询为 $Q_v$、语言键值为 $K_\ell,V_\ell$，单层交叉注意是：

$$
\text{Attn}(Q_v,K_\ell,V_\ell)=\mathrm{softmax}\!\left(\frac{Q_v K_\ell^{\top}}{\sqrt{d}}\right)V_\ell.
$$

无论哪种融合，关键是语言必须真的参与决定动作。如果删掉语言 token、输出几乎不变，模型就只是把图像和状态又用了一遍，并没有“听”这句话。

## 同一画面换指令

判断语言是否生效，最小反事实是固定图片和状态，把“拿红杯”换成“拿蓝杯”。设两次输出的动作分别为 $\hat a_t^{(\text{红})}$ 和 $\hat a_t^{(\text{蓝})}$，我们希望两者有可测量的差异：

$$
\Delta a=\bigl\lVert \hat a_t^{(\text{红})}-\hat a_t^{(\text{蓝})}\bigr\rVert_2\;\gg\;0.
$$

如果训练数据里红杯总在左边、蓝杯总在右边，模型可能只学到位置捷径，根本不读语言。测试集需要交换颜色与位置，才能逼出语言是否真在被使用。

## 预训练视觉语言模型提供什么

预训练的 **VLM**（vision-language model）见过海量图文，可以提供物体与语言的通用表示，减少从零学习语义的开销。但它通常不输出符合机器人坐标系、控制频率和安全约束的关节动作。

换句话说，VLM 给的是一个会看会说的底座。VLA 还要补上动作示范、机器人自身状态和控制接口。会看会说，不等于已经会动。

## 冻结还是微调

数据少的时候，可以冻结视觉或语言编码器，只训练融合层与 action head。设可训练参数为 $\theta_{\text{fuse}}$ 和 $\theta_{\text{head}}$，冻结参数为 $\theta_{\text{enc}}$，则一步更新只动前者：

$$
\theta_{\text{fuse}},\theta_{\text{head}}\leftarrow \theta_{\text{fuse}},\theta_{\text{head}}-\eta\,\nabla_{\theta_{\text{fuse}},\theta_{\text{head}}}\,\mathcal{L}_{\text{BC}}.
$$

这样省显存、也减少过拟合。若相机视角和预训练图像差异很大，再逐步解冻后层。公平对照应当依次比较 state-only、image+state、image+language+state 三档，而不是一次加全部模态后只报最好的结果。

## 小结

- [ ] VLA 用 Transformer 把图像、语言、proprio 三种 token 融合成一个动作。
- [ ] 交叉注意是让语言查询图像、图像约束语言的标准工具。
- [ ] 同一画面换指令，是检查语言条件是否生效的最小反事实。
- [ ] VLM 表示不能替代动作数据与控制接口。

下一篇让这台 VLA 一次输出一小段动作，并讨论当多种动作都正确时该怎么建模。动手实验见 [7.8 动手：从零实现 VLA 与世界模型检查器](/chapters/07-robot-vla/08-robot-vla) 的第一份 Notebook「搭一台小型 VLA」。
