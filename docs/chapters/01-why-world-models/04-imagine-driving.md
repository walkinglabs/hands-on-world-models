# 1.4　动手：在想象中驾驶

> **本节目标**：不写代码。先玩一台会「猜下一步」的世界模型，再看 2018 年那台在梦里学会开车的系统长什么样。体验结束时，你应当能用自己的话回答：模型看见了什么、记住了什么、猜对或猜错意味着什么。

> **前置知识**：[1.1 观察、状态与变化](/chapters/01-why-world-models/01-observation-and-state)、[1.2 什么是世界模型](/chapters/01-why-world-models/02-what-is-a-world-model)、[1.3 经典世界模型](/chapters/01-why-world-models/03-classic-world-models)。

---

2018 年，David Ha 与 Jürgen Schmidhuber 写了一篇可以在浏览器里打开的文章：[World Models](https://worldmodels.github.io/)。里面有一段赛车：画面糊，车却在开。作者说，那不是录屏——一个 867 个参数的线性控制器，完全在模型的想象里学会了转弯。

在你亲手训练它之前（那是 [4.6](/chapters/04-decision-and-planning/06-reproduce-world-models) 的事），先用下面这个更小的世界，把同一件事摸一遍。

## 先玩：模型怎样猜你的下一步

用方向键（或按钮）把小人送到右下角的旗帜，避开中间的陷阱。

右边那块画布是**模型想象**。它是一台刚开始一片空白的表格世界模型，唯一的本领是数数：把你经历过的每个「格子 + 方向 → 下一格」记下来，再据此预测下一步。刚开始它满屏问号；你走得越多，它猜得越准。

注意右侧的两拍节奏：**按下方向的瞬间，小人还在原地，模型先亮出预测**；片刻后小人才真正移动，预测同时被判定——绿色 ✓ 猜对、橙色 ✗ 猜错、问号表示从未见过。

<PlayWorldModel />

玩的时候盯住三件事：

1. **问号阶段**：没见过的「状态–动作」只能回答「不知道」。这就是后面要讲的数据覆盖。
2. **撞墙**：在边界按方向键，小人原地不动。模型见过几次后也会预测「原地不动」——它不知道什么是墙，但它学到了墙的效果。
3. **陷阱**：故意掉进去一次，再重置。模型已经「记住」了那个坑。经历，而不是规则，是它全部的知识来源。

这就是世界模型的最小形态：**观察当前格子，根据你选的动作，在内部预演下一格，再拿真实结果修正自己。**

手写这张转移表、再用它做规划，放到 [3.5 动手：表格型世界模型的从零开始实现](/chapters/03-data-and-first-model/05-learn-a-table-world)。本节只要求你玩到它、看见它。

## 再看：在想象中开车

把格子换成像素，把「下一格」换成「下一帧画面」，就是 2018 年那台 World Models。视觉模块把赛道压成几十个数，记忆模块在这串数字上滚动未来，控制器只在想象里试动作——全程可以不碰真实环境。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/dream-generation.png" alt="世界模型生成的梦境赛道" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">训练好的世界模型生成的「梦境世界」：控制器在记忆模块的想象中开了 200 步，全程未接触真实环境。从左到右，画面从清晰逐渐模糊——复合误差在累积，但赛道、车身、草地的结构始终可辨。</div>
</div>

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/real-evaluation.png" alt="把梦里学会的控制器放回真实赛道" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">真正的验收不在梦里。把控制器接回真实 CarRacing，看它还能不能转弯。梦境分数高不等于真实分数高——控制器完全可能钻模型的空子。</div>
</div>

上面两张图来自课程对 World Models 的复现。本节不要求你跑通训练；只要记住对照：

| 你刚刚玩的九格 | 2018 年的赛车梦 |
| --- | --- |
| 状态是格子坐标 | 状态是压缩后的画面向量 |
| 预测下一格 | 预测下一帧（及其奖励） |
| 问号 = 没见过 | 模糊 = 复合误差在累积 |
| 猜错后表格被改写 | 猜错后网络权重被更新 |

## 这一节结束时你应当能说清

- 世界模型先在内部预演，再拿真实结果对照；预演和对照不是同一步。
- 「没见过」和「猜错」是两种失败：前者是覆盖，后者是模型。
- 画面好看不是验收标准。梦里能开、真实赛道也能开，才算用上了这台模型。

共同基础从下一章开始：张量、编码器、记忆、压缩。九格的手写实现在 [3.5](/chapters/03-data-and-first-model/05-learn-a-table-world)；把梦境赛车真正训出来在 [4.6](/chapters/04-decision-and-planning/06-reproduce-world-models)。

## 参考文献

1. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. _NeurIPS 2018_. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
2. Ha, D., & Schmidhuber, J. (2018). World Models. 交互式文章：[worldmodels.github.io](https://worldmodels.github.io/)
