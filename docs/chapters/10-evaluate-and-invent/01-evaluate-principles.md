# 10.1　怎样评价世界模型

> **第 9 章 · 评测与研究设计**
>
> 训练损失下降，只说明优化器完成了它收到的任务。单步预测准确，不代表多步推演不会漂移；生成视频清晰，也不代表动作后果正确。本章只回答两个问题：怎样判断世界模型真正有用，怎样把一次稳定失效变成可检验的研究问题。
>
> 核心实验：[9.2 动手：世界模型的系统评测](/chapters/10-evaluate-and-invent/02-systematic-evaluation) 与 [9.3 动手：从失效到下一台世界模型](/chapters/10-evaluate-and-invent/03-failure-analysis)。

一段预测视频可以很清晰，却把动作的后果画错；一段画面略显模糊的预测，也可能准确保留碰撞、接触和可达性。评价世界模型时，首先要分开两个问题：

- **感知质量**：预测结果看起来是否真实、清晰、连贯；
- **功能效用**：预测结果是否足以支持规划、控制和决策。

二者相关，但不能相互替代。

## 两类评价回答不同问题

感知指标比较预测 \(\hat{o}_{t+1:t+H}\) 与真实观测 \(o_{t+1:t+H}\)：

\[
S_{\text{perceptual}}
=

\operatorname{Sim}
\left(
\hat{o}_{t+1:t+H},
o_{t+1:t+H}
\right).
\]

它可以衡量像素误差、特征距离、视频连贯性或人类偏好。功能指标则把预测交给一个使用者 \(g\)，观察它能否完成任务：

\[
S_{\text{functional}}
=

\operatorname{TaskScore}
\left(
g(\hat{o}_{t+1:t+H}, a_{t:t+H-1})
\right).
\]

这里的 \(g\) 可以是规划器、策略、碰撞检查器、机器人控制器或驾驶决策模块。评价对象不再只是“画面”，而是“画面中的信息能否被正确使用”。

## 一个最小反例

设目标在右侧，墙在中间。模型 A 生成的画面锐利，但让智能体穿墙；模型 B 的画面较模糊，却正确预测绕行轨迹。

| 模型 | 画面质量 | 动作后果 | 规划结果 |
| ---- | -------- | -------- | -------- |
| A    | 高       | 错       | 失败     |
| B    | 中       | 对       | 成功     |

如果只报告感知质量，A 会胜出；如果任务是规划，B 才是更有用的世界模型。这个反例说明：**生成质量高，不等于世界建模正确。**

## 三层证据

评价一台世界模型时，证据应从低到高排列：

1. **预测层**：一步误差、多步漂移、校准和不确定性；
2. **因果层**：固定起点，只改变动作，预测是否产生正确差异；
3. **任务层**：模型是否提高真实环境中的成功率、回报或安全性。

预测层用于定位错误，因果层检查模型是否真正听从动作，任务层回答模型是否值得使用。三层不能压成一个总分。

## 路线专属指标

不同路线应使用不同的最终指标：

- 决策与规划：真实回报、样本效率、模型利用程度；
- 交互式视频：动作一致性、长时稳定性、实时延迟；
- JEPA：表示质量、动作敏感性、下游任务表现；
- 具身智能：任务成功率、碰撞率、恢复能力和真机迁移差距；
- 空间世界：几何一致性、占用预测、重访一致性和规划增益。

统一评测框架提供共同问题，但不要求所有路线共享同一个总分。

## 评价协议

一个可信的评价协议至少写清楚：

- 数据、切分与预测视野；
- 弱基线和强基线；
- 单步、多步与反事实测试；
- 分布外条件；
- 下游使用者及其预算；
- 随机种子、方差与失败样例。

算力配置、泄漏检查和复现字段统一放在[附录 B](/appendices/data-compute-delivery)，避免正文被工程清单打断。

## 小结

感知质量回答“像不像”，功能效用回答“能不能用”。世界模型的评价必须同时检查预测、动作响应与下游任务，尤其不能用一段漂亮的演示视频替代功能证据。

下一节将把这些原则变成一套可运行的[系统评测](/chapters/10-evaluate-and-invent/02-systematic-evaluation)。

---

## 参考资料

### 实践博客

1. [WorldScore 排行榜与文档](https://worldscore.stanford.edu/) —— 统一评测基准的官方站点：指标定义、榜单与提交方式，配 9.1。
2. [WorldModelBench 项目页](https://worldmodelbench-team.github.io/) —— 视频世界模型评测集的官方页面，列出物理、常识与幻觉三个维度。
3. [Your AI Product Needs Evals (Hamel Husain)](https://hamel.dev/blog/posts/evals/) —— 工程界公认的评测实践博客：怎样从真实失败里长出评测集，配 9.2。
4. [Open X-Embodiment 项目页](https://robotics-transformer-x.github.io/) —— 跨本体机器人数据协作的官方页面，展示多来源数据怎样汇总与评测。
5. [LIBERO 基准文档](https://libero-project.github.io/) —— VLA 评测基准的官方页面，含任务套件与协议，是 LIBERO-Plus 分析的基础。

### 经典文献

1. [DeepMind Control Suite (Tassa et al., 2018)](https://arxiv.org/abs/1801.00690) —— 连续控制基准套件，本章基线对比与多步评价常用的实验场。
2. [Mastering Continuous Control from Raw Pixels: DrQ-v2 (Yarats et al., 2022)](https://arxiv.org/abs/2107.09645) —— 像素输入连续控制的公平基线范例，示范了基线与消融该怎么写。
3. [WorldScore: A Unified Evaluation Benchmark for World Generation (Duan et al., 2025)](https://arxiv.org/abs/2504.00983) —— 把视觉质量、动态一致性与指令跟随拆开的统一评测基准，配 9.1。
4. [WorldModelBench: Judging Video Generation Models As World Models (Huang et al., 2025)](https://arxiv.org/abs/2502.20694) —— 物理规律、常识与幻觉三个维度的评测集，配 9.2 的反事实与 OOD 检查。
5. [LIBERO-Plus: In-depth Robustness Analysis of Vision-Language Action Models (Liu et al., 2025)](https://arxiv.org/abs/2510.13626) —— 对 VLA 模型做系统性鲁棒性分析的范例，展示“审问模型”该问哪些问题。
