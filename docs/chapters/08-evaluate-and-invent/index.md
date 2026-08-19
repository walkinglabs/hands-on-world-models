# 第 8 章　评价与研究设计

训练损失的下降仅表明模型在优化目标上取得了进展，但这并不足以证明模型真正理解了世界。单步预测精度高不代表长时序滚动不会漂移，生成视频视觉质量好不代表动作条件响应正确，smoke测试通过也不代表模型能够在24GB显存预算下完成完整训练。本章的目标是建立一套严谨的世界模型评测方法论，让你能够系统性地验证模型的能力边界。

在前面的章节中，你已经学会了如何实现一个能够运行的世界模型。然而，一个能够运行的模型和一个真正有用的模型之间存在本质区别。本章我们将首先学习如何通过基线对比、反事实测试、分布外泛化检测和可复现性记录来严格评估模型；随后，我们将从可复现的失败案例出发，提出改进假设，进行最小化代码修改，并通过公平对照实验完成一次完整的研究迭代，最终实现你自己的改进版世界模型。

## 本章内容

1. [8.1　基线与多步评价](./08-01-baselines-and-horizons.md)：建立弱基线，对比单步预测与开环多步滚动的本质差异，通过horizon曲线分析误差累积特性
2. [8.2　反事实、分布外与鲁棒性](./08-02-counterfactual-and-ood.md)：通过控制变量法验证动作条件响应，测试模型在分布外输入下的泛化能力与不确定性校准
3. [8.3　运行证据与复现](./08-03-hardware-evidence.md)：规范实验记录标准，明确smoke测试、目标配置与研究配置的区别，保证结果可复现
4. [8.4　失败分析与下一台世界模型](./08-04-next-world-model.md)：从可复现的稳定失败出发，学习提出竞争假设、设计可证伪实验的研究方法
5. [8.5　动手：审问世界模型](./08-05-interrogate-world-model.md)：加载PA1训练得到的模型，完成一次完整的系统化评测
6. [8.6　动手：实现自己的世界模型](./08-06-next-model-proposal.md)：基于失败分析提出最小改进，实现代码并通过公平对照实验验证假设
7. [8.7　动手：审问一台世界模型](./08-07-test-a-world-model.md)：对 PA1 模型做基线、多步、反事实与 OOD 检查

不同应用路线的主指标存在差异：决策控制路线关注真实环境回报与样本效率，视频生成路线关注动作一致性与长时稳定性，JEPA路线关注表示质量与下游任务效用，机器人路线关注任务成功率与碰撞率，空间世界路线关注几何一致性与未来预测准确率。但所有路线的评测都必须包含六个核心要素：公平基线对比、多步预测曲线、反事实控制测试、分布外压力测试、完整资源记录以及典型失败样例分析。

---

**本章目标**：学完本章后，你将掌握一套严谨的模型评测方法论，能够客观判断模型的能力边界，并具备从失败中迭代改进模型的研究能力。

## 参考资料

### 实践博客（5 篇）

1. [WorldScore 排行榜与文档](https://worldscore.stanford.edu/) —— 统一评测基准的官方站点：指标定义、榜单与提交方式，配 8.1。
2. [WorldModelBench 项目页](https://worldmodelbench-team.github.io/) —— 视频世界模型评测集的官方页面，列出物理、常识与幻觉三个维度。
3. [Your AI Product Needs Evals (Hamel Husain)](https://hamel.dev/blog/posts/evals/) —— 工程界公认的评测实践博客：怎样从真实失败里长出评测集，配 8.5。
4. [Open X-Embodiment 项目页](https://robotics-transformer-x.github.io/) —— 跨本体机器人数据协作的官方页面，展示多来源数据怎样汇总与评测。
5. [LIBERO 基准文档](https://libero-project.github.io/) —— VLA 评测基准的官方页面，含任务套件与协议，是 LIBERO-Plus 分析的基础。

### 原始论文（5 篇）

1. [DeepMind Control Suite (Tassa et al., 2018)](https://arxiv.org/abs/1801.00690) —— 连续控制基准套件，本章基线对比与多步评价常用的实验场。
2. [Mastering Continuous Control from Raw Pixels: DrQ-v2 (Yarats et al., 2022)](https://arxiv.org/abs/2107.09645) —— 像素输入连续控制的公平基线范例，示范了基线与消融该怎么写。
3. [WorldScore: A Unified Evaluation Benchmark for World Generation (Duan et al., 2025)](https://arxiv.org/abs/2504.00983) —— 把视觉质量、动态一致性与指令跟随拆开的统一评测基准，配 8.1。
4. [WorldModelBench: Judging Video Generation Models As World Models (Huang et al., 2025)](https://arxiv.org/abs/2502.20694) —— 物理规律、常识与幻觉三个维度的评测集，配 8.2 的反事实与 OOD 检查。
5. [LIBERO-Plus: In-depth Robustness Analysis of Vision-Language Action Models (Liu et al., 2025)](https://arxiv.org/abs/2510.13626) —— 对 VLA 模型做系统性鲁棒性分析的范例，展示“审问模型”该问哪些问题。
