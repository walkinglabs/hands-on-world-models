# 5.4　行动以前检查后果

直接 VLA 根据当前输入输出动作。它可以很快，却不一定在执行前比较“从左侧接近”和“从右侧接近”会发生什么。

我们增加一个小型后果模型，让 VLA 提出候选动作，再预测各动作的一步结果。

## 两个模块

```text
VLA：image + instruction + proprio → candidate actions
World model：current state + candidate action → next state / success / collision
```

Planner 或 reranker 使用后果模型重新排序候选动作。

## 为什么数据要包含坏动作

示范数据大多是专家成功动作。若后果模型只见过好动作，它无法判断碰撞动作，只会在训练分布之外自信外推。

Tabletop 生成器可以从同一状态执行安全与不安全候选，记录 next state、success 和 collision。坏动作是 checker 的必要监督。

## Direct 与 lookahead 的公平比较

固定同一候选生成器和计算预算，比较：

1. 直接执行最高概率动作；
2. 用一步后果模型重新排序；
3. 使用真实模拟器后果排序，作为上限参考。

如果 learned checker 没有接近真实后果上限，要检查模型误差和 OOD，而不是只增加候选数量。

## Checker 也会被利用

候选生成器可能找到后果模型错误预测为安全的动作。需要不确定性阈值、动作约束、真实反馈和失败数据回流。

世界模型不是安全保证。真实机器人仍需要独立的速度、力矩、工作空间和急停约束。

## PA1-D

D1 从 state BC 逐步加入图片、语言和 action chunk。D2 加入候选动作与 one-step checker。主指标是成功率、碰撞率、OOD 和端到端延迟。

## 小结

- [ ] VLA 输出动作，世界模型预测动作后果。
- [ ] 后果数据必须包含失败与坏动作。
- [ ] learned checker 要与直接执行和真实模拟器上限比较。
- [ ] 世界模型检查不能代替独立安全约束。
