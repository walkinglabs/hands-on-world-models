# 第 9 章　评测与研究设计

训练损失下降，只说明优化器完成了它收到的任务。单步预测准确，不代表多步推演不会漂移；生成视频清晰，也不代表动作后果正确。本章只回答两个问题：怎样判断世界模型真正有用，怎样把一次稳定失效变成可检验的研究问题。

前两页先区分感知质量与功能效用，再用同一套解析例子完成多步、反事实、分布外、不确定性和规划器漏洞测试。最后一页从一个可复现的失效出发，提出竞争解释、做最小改动，并用公平对照决定解释是否站得住。

## 本章内容

1. [9.1 感知质量与功能效用](./01-perception-and-utility.md)：区分“看起来真实”与“足以支持决策”。
2. [9.2 动手：世界模型的系统评测](./02-systematic-evaluation.md)：完成多步、反事实、分布外、校准与规划器漏洞测试。
3. [9.3 动手：从失效到下一台世界模型](./03-failure-to-next-model.md)：把稳定失效变成竞争假设、最小改动与可证伪实验。

不同路线不使用同一个总分。决策控制看真实回报，交互式视频看动作一致性，JEPA 看下游效用，机器人看任务成功率与碰撞率，空间世界看几何一致性与未来占用。共同要求只有三项：公平基线、多步与反事实测试、可复现的失败样例。

---

**本章目标**：判断模型的能力边界，并完成一次从失效分析到模型改进的最小研究循环。

## 参考资料

### 实践博客（5 篇）

1. [WorldScore 排行榜与文档](https://worldscore.stanford.edu/) —— 统一评测基准的官方站点：指标定义、榜单与提交方式，配 9.1。
2. [WorldModelBench 项目页](https://worldmodelbench-team.github.io/) —— 视频世界模型评测集的官方页面，列出物理、常识与幻觉三个维度。
3. [Your AI Product Needs Evals (Hamel Husain)](https://hamel.dev/blog/posts/evals/) —— 工程界公认的评测实践博客：怎样从真实失败里长出评测集，配 9.2。
4. [Open X-Embodiment 项目页](https://robotics-transformer-x.github.io/) —— 跨本体机器人数据协作的官方页面，展示多来源数据怎样汇总与评测。
5. [LIBERO 基准文档](https://libero-project.github.io/) —— VLA 评测基准的官方页面，含任务套件与协议，是 LIBERO-Plus 分析的基础。

### 原始论文（5 篇）

1. [DeepMind Control Suite (Tassa et al., 2018)](https://arxiv.org/abs/1801.00690) —— 连续控制基准套件，本章基线对比与多步评价常用的实验场。
2. [Mastering Continuous Control from Raw Pixels: DrQ-v2 (Yarats et al., 2022)](https://arxiv.org/abs/2107.09645) —— 像素输入连续控制的公平基线范例，示范了基线与消融该怎么写。
3. [WorldScore: A Unified Evaluation Benchmark for World Generation (Duan et al., 2025)](https://arxiv.org/abs/2504.00983) —— 把视觉质量、动态一致性与指令跟随拆开的统一评测基准，配 9.1。
4. [WorldModelBench: Judging Video Generation Models As World Models (Huang et al., 2025)](https://arxiv.org/abs/2502.20694) —— 物理规律、常识与幻觉三个维度的评测集，配 9.2 的反事实与 OOD 检查。
5. [LIBERO-Plus: In-depth Robustness Analysis of Vision-Language Action Models (Liu et al., 2025)](https://arxiv.org/abs/2510.13626) —— 对 VLA 模型做系统性鲁棒性分析的范例，展示“审问模型”该问哪些问题。
