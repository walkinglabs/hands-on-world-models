# 附录 A　数学、代码与术语速查

本附录只收纳正文中反复使用、适合随时查阅的工具，不引入新的课程主线。

## 概率与期望

\[
\mathbb{E}_{x\sim p(x)}[f(x)]
=

\sum_x p(x)f(x)
\quad\text{或}\quad
\int p(x)f(x)\,dx.
\]

条件分布 \(p(s_{t+1}\mid s_t,a_t)\) 表示在当前状态和动作已知时，下一个状态的可能性。世界模型通常学习的正是这种条件关系。

## 梯度与链式法则

若损失 \(\mathcal{L}\) 依赖中间变量 \(z=f_\theta(x)\)，则

\[
\frac{\partial \mathcal{L}}{\partial \theta}
=

\frac{\partial \mathcal{L}}{\partial z}
\frac{\partial z}{\partial \theta}.
\]

多步 rollout 会让梯度穿过重复使用的动力学模型。梯度爆炸时先检查序列长度、归一化和梯度裁剪，不要只调学习率。

## KL 与熵

\[
D_{\mathrm{KL}}(q\|p)
=

\mathbb{E}_{q(z)}
\left[
\log q(z)-\log p(z)
\right].
\]

KL 衡量两个分布的差异，不是对称距离。RSSM 中，posterior 使用当前观测，prior 只使用历史与动作；KL 让 prior 学会在没有未来观测时预测 posterior。

熵

\[
\mathcal{H}(p)=-\mathbb{E}_{p(x)}[\log p(x)]
\]

衡量分布的不确定性。熵高不等于模型正确，只表示模型给出的分布更分散。

## PyTorch 与 JAX/Flax

| 任务     | PyTorch              | JAX/Flax                          |
| -------- | -------------------- | --------------------------------- |
| 参数存放 | `nn.Module` 内部状态 | 显式参数树                        |
| 前向计算 | `y = model(x)`       | `y = model.apply(params, x)`      |
| 梯度     | `loss.backward()`    | `jax.grad(loss_fn)`               |
| 优化器   | `optimizer.step()`   | `updates, state = tx.update(...)` |
| 随机数   | 全局或 `Generator`   | 显式传递 `PRNGKey`                |
| 编译     | `torch.compile`      | `jax.jit`                         |
| 批处理   | 张量批维             | `jax.vmap`                        |

阅读 JAX 项目时，先找到参数树、随机 key 和纯函数边界；阅读 PyTorch 项目时，先找到 `Module`、`forward` 和优化器更新。

## 常用术语

| 中文             | 英文                                    | 缩写 |
| ---------------- | --------------------------------------- | ---- |
| 世界模型         | world model                             | WM   |
| 动力学模型       | dynamics model                          | —    |
| 循环状态空间模型 | recurrent state-space model             | RSSM |
| 模型预测控制     | model predictive control                | MPC  |
| 交叉熵方法       | cross-entropy method                    | CEM  |
| 联合嵌入预测架构 | joint embedding predictive architecture | JEPA |
| 视觉语言动作模型 | vision-language-action model            | VLA  |
| 鸟瞰图           | bird's-eye view                         | BEV  |
| 分布外           | out of distribution                     | OOD  |
| 从仿真到现实     | sim-to-real                             | —    |

术语首次出现在正文时使用“中文（英文，缩写）”，后文优先使用缩写或中文短名。
