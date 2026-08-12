# 第 3 章　可交互视频世界

普通视频模型生成一种看起来合理的后来。本章要求更严格：固定同一个起点，只更换按键，画面中的物体也要按相应方向变化。

## 本章文章

1. [先把画面和按键接对](./03-01-video-data.md)
2. [VQ-VAE：把一帧画面变成 token](./03-02-vq-tokenizer.md)
3. [动作条件 Transformer：一帧一帧生成未来](./03-03-action-transformer.md)
4. [Diffusion、多步漂移与实时性](./03-04-diffusion-and-evaluation.md)

## 本章实验与作业

- [B1–B2：互动视频实验](/labs/route-bc)
- [PA1-B：可以按键控制的小世界](/assignments/pa1-b)

主指标是动作一致性、多步稳定和每帧延迟。单张清楚的截图不能代替这些检查。
