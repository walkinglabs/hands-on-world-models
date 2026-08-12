# 第 5 章　VLA 与机器人世界模型

视觉语言模型可以描述桌上的杯子，机器人却需要输出能执行的动作。本章先实现动作模型，再加入一台小世界模型检查候选动作的后果。

## 本章文章

1. [机器人数据与行为克隆](./05-01-robot-data-and-bc.md)
2. [图像、语言和机器人状态怎样合在一起](./05-02-vision-language-action.md)
3. [Action Chunk 与多种可行动作](./05-03-action-chunk.md)
4. [行动以前检查后果](./05-04-world-model-checker.md)

## 本章实验与作业

- [D1–D2：机器人路线实验](/labs/route-de)
- [PA1-D：Tiny VLA](/assignments/pa1-d)

直接 VLA 的输出是动作；世界模型的输出是动作后果。两者可以连接，但不能用同一个名字代替。
