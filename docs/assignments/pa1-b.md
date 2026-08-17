# PA1-B · 动手：做出一个听从按键的视频小世界

目标不是生成最好看的片段，而是做一次完整、可复现的视频世界模型实验：从同一段历史出发，更换动作，未来随之改变；模型连续读取自己的输出后，仍能说明从哪里开始失效。

## 必做

1. 按 episode 与 seed 切分 PixelWorld，画出 `frame_t, action_t, frame_{t+1}`；
2. 实现复制上一帧和 no-action 两个基线；
3. 训练 VQ-VAE，报告重建误差、码本使用率与小物体位置误差；
4. 训练 additive action-conditioned token AR 模型；
5. 固定历史和随机源，分别替换五种动作，测量物体位移方向；
6. 分别运行 teacher-forced 与 free rollout，在 1、5、15、30、100 步画漂移曲线；
7. 在相同数据、更新数和 seed 下比较 no-action 与 additive，说明动作条件带来了什么；
8. 记录 tokenizer、动态模型、Decoder 和端到端的延迟、峰值显存、总时间与 checkpoint 哈希。

## 二选一对照

### 方向 1：动作怎样进入模型

在 additive 基线之外实现 FiLM 或 AdaLN。保持其余条件不变，比较动作一致性、参数量和延迟。更复杂的方法没有提升也可以得到满分，只要实验公平、解释诚实。

### 方向 2：连续 latent 与逐帧噪声

训练 tiny conditional denoiser，为一段视频中的不同帧采样不同噪声等级；从 `x / epsilon / v` 中选择两个目标做 smoke 或短训练。比较生成质量与采样步数、延迟的关系。

## 真实数据迁移（拔高，不是当前必做）

可以迁移到 DINO-WM PushT 或 CarRacing 小数据，但必须提交 loader、许可、动作时间、控制频率和固定 split。当前仓库只发布了这些来源的数据合约，尚未发布可复现 artifact，因此不能把下载链接当作已完成实验。

## 结果页至少包含

- 同一起点的动作反事实网格；
- no-action、additive 与所选对照的统一表格；
- teacher-forced 与 free rollout 曲线；
- 固定 seed 的成功与失败片段；
- 画面指标、动作指标、首个失败时间和端到端延迟；
- 数据版本、配置、环境、显存峰值与 checkpoint 哈希。

## 24GB 目标

建议从 `64×64 RGB`、每帧最多 64 个 token、短上下文和小型 Transformer 开始，单卡 peak reserved 设计目标不超过 22GB。先用 smoke 配方验证数据与 loss，再扩大 batch 或 horizon。

这是课程设计预算，不是实测结果。当前没有 PA1-B 的 24GB 完整运行记录；在日志、曲线和 checkpoint 齐全前，不得标为“24GB 已验证”。

## 不接受的结论

- 只交最好看的一段视频；
- logits 随动作改变，就声称解码画面可控；
- 只报 PSNR、FVD 或 one-step loss，不做动作反事实；
- 使用无动作视频，却声称学到了真实控制；
- 在 teacher forcing 下表现好，就声称可以长时间生成；
- 同时更换模型、数据和训练预算，再把差异归因于某一个组件。
