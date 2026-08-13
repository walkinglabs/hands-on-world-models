# 第 1 章　世界模型的常用组件

九格世界只有几个整数。现实中的一次观察可能包含图片、相机、语言和机器人状态，一段经历还要包含动作、奖励与时间顺序。

这一章建立五条路线共用的语言。组件只讲到足以判断输入、输出和用途；完整实现留到真正使用它的路线文章。第 2 章再把时间、数据和第一台模型接起来。

## 本章文章

1. [张量、时间与轨迹](./01-01-tensors-and-trajectories.md)：读懂 `[B,T,C,H,W]`，把动作放在正确的两帧之间。
2. [CNN 与 ViT：把图片变成特征](./01-02-cnn-and-vit.md)：比较局部卷积与 patch token。
3. [RNN、Transformer 与 RSSM：把过去带到现在](./01-03-memory-and-dynamics.md)：从速度线索走到随机隐状态。
4. [VAE、VQ-VAE 与 Diffusion：压缩和生成](./01-04-compression-and-generation.md)：比较连续 latent、离散 token 和多种未来。
5. [从相机到 BEV 与 Occupancy](./01-05-space-representations.md)：认识三维坐标、点云和空间占用。
6. [Value、Policy 与 Planner](./01-06-value-policy-planner.md)：说明预测怎样被用于选择动作。

## 本章实验

- [F1–F2：共同组件实验](/labs/foundations)
- [PA0：第一台可学习世界](/assignments/pa0)

文章可以细分，实验不按名词拆散：F1 接起视觉、记忆和压缩，F2 接起空间、评价和规划。F3 放在第 2 章，用一份 Notebook 接起数据、学习与闭环检查。

## 学完以后怎样选路

先完成第 2 章，把连续经历整理成数据并学出第一台模型。随后写下模型最需要交出的结果：latent、画面、feature、机器人动作，还是三维占用，再从第三部分选择一条路线。
