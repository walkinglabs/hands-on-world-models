# 5.5　记忆、漂移与实时生成

一台视频模型可以在一步预测上表现很好，连续运行后却迅速忘记起点。人物衣服换色、场景结构改变、小物体消失，都是同一问题的不同表现：模型的输出正在变成自己的输入。

## 动作怎样进入模型

动作先通过 embedding 或 MLP 变成向量，再用不同方式进入视频网络：

| 方法            | 直观理解                           | 本章安排             |
| --------------- | ---------------------------------- | -------------------- |
| additive        | 把动作向量直接加到视觉 token       | 必做基线             |
| FiLM / AdaLN    | 用动作改变特征的缩放、平移或归一化 | 5.6 做 tiny 对照     |
| cross-attention | 让视觉位置主动读取动作 token       | 论文阅读，不作为默认 |

复杂方法不一定自动更好。低维按键只包含少量信息，直接相加往往已经足够。公平实验必须保持数据、模型宽度、训练步数和 seed 相同，一次只替换动作注入方式；同时保留 `no-action` 模型，检查视觉历史本身能解释多少变化。

## 模型要记住什么

最近几帧负责速度和局部细节，较远历史负责场景、身份和已经发生的事件。只把固定长度窗口塞给 Transformer，会遇到两个限制：窗口外的事被忘掉，窗口内的计算随长度迅速增加。

论文中的解决办法大致分为三类：

1. **隐式记忆**：滑动窗口、压缩远处历史、KV Cache、检索旧帧或循环状态；
2. **显式 3D 记忆**：把点云、三平面、Gaussian 或空间 token 留在世界坐标中；
3. **物理约束**：把碰撞、重力或已知模拟器加入结构、损失或奖励。

本章只实现第一类的最小版本，并比较短历史与长历史。显式 3D 记忆转到第 8 章；物理约束作为 9.4 研究题。二维视频生成得很清楚，并不表示它已经建立了稳定的三维世界。

## 专门检查训练和生成的落差

至少运行两种模式：

```text
teacher-forced：每一步都给真实历史
free rollout：第二步开始读取模型自己的预测
```

若两条曲线很快分开，问题不在“一步不会预测”，而在“模型没有学会从自己的小错误中恢复”。Self-Forcing、Causal Forcing、Rolling Forcing、error recycling 等名字，都可以放回这个具体缺口中理解。

## 指标要跟结论对应

| 想声称什么     | 至少检查什么                                      |
| -------------- | ------------------------------------------------- |
| 画面接近真实   | PSNR、SSIM、LPIPS；数据量足够时再用 FVD           |
| 动作有效       | 固定历史换动作，测方向、位置、接触或状态变化      |
| 长时间稳定     | 不同 horizon 的曲线、首个失败时间、身份与场景漂移 |
| 物理合理       | 碰撞、穿透、重力、对象永久性等任务指标            |
| 可以交互       | 编码、模型、解码的端到端 FPS、延迟和峰值显存      |
| 可以做世界模型 | 下游规划或任务成功率是否比无模型基线更好          |

PSNR 很高可能只是复制上一帧；FVD 较好也不能证明按键有效。Nano World Model 使用固定验证子集和 seed，并同时输出样例视频与指标文件，这种可复现做法比某一个具体数值更值得课程吸收。

## 从小实验到研究系统

本章按下面的台阶推进：

```text
PixelWorld 时间对齐
→ VQ tokenizer 与 token AR
→ no-action / additive / FiLM 对照
→ teacher-forced / free rollout 对照
→ tiny 逐帧噪声实验
→ 真实动作视频迁移
→ 长时记忆、实时加速或下游规划
```

最后三项属于进阶。当前仓库内可直接生成的是 PixelWorld；PushT 与 CarRacing 的课程 loader 尚未发布，所以 5.7 的必做结论只能建立在 PixelWorld，不能把“来源已知”写成“数据已就绪”。

## 读论文时怎样定位

- **IRIS、STORM 一类**：关注离散 tokenizer、因果预测和 RL 用途；
- **DIAMOND、GameNGen、Oasis 一类**：关注像素或连续 latent、动作条件与少步去噪；
- **Genie 一类**：关注无动作视频中的 latent action，以及怎样把它变成可用控制；
- **Diffusion Forcing、Nano World Model 一类**：关注逐帧噪声、采样调度、预测目标和可复现实验轴；
- **长视频与实时 AR-Diffusion**：关注 train–inference gap、记忆、少步蒸馏、稀疏注意力和缓存。

模型名称会继续变化，上面五个问题不会很快过时。

## 资料入口

- [Awesome World Models](https://github.com/knightnemo/Awesome-World-Models)：按游戏、驾驶、具身、科学与通用方法定位工作。
- [Nano World Model](https://github.com/simchowitzlabpublic/nano-world-model)：查看最小 Diffusion Forcing 代码、数据格式、消融、评价和 MPC 应用。
- [Awesome Video World Models with AR Diffusion](https://github.com/gracezhao1997/Awesome-Video-World-Models-with-AR-Diffusion)：按算法、应用和实时基础设施继续阅读。
- [Evolution of Video Generative Foundations](https://arxiv.org/abs/2604.06339)：从 GAN、Diffusion、AR 到视频世界模型的总览。
- [Survey of Video Diffusion Models](https://arxiv.org/abs/2504.16081)：Diffusion 架构、训练、数据、评价与应用综述。

## 小结

- [ ] 动作注入、预测目标和模型规模应当分轴消融，不要一起更换。
- [ ] 长时生成要区分局部细节、远期记忆、三维一致性和物理一致性。
- [ ] 自由 rollout 才能暴露训练与部署的落差。
- [ ] 评价要覆盖画面、动作、长时、速度和下游用途，不能只报生成 loss。
