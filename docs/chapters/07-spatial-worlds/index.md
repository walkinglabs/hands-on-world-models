# 第 7 章　空间世界与自动驾驶

照片里的像素会随相机移动而变化，房间和道路却不动。要把跨视角、跨时间的观察拼成一个一致的空间，就需要相机几何与空间表示。

本章从一张照片怎样投影开始，走到 BEV 和占用网格，再走到 NeRF 与 3DGS 这类连续三维表示，最后把时间加进来，落到驾驶世界模型对未来占用的预测。

## 本章文章

1. [7.1 相机几何与投影](./01-camera-geometry.md)：针孔模型、内外参，以及为什么一个像素只是一条射线。
2. [7.2 BEV、占用网格与 LSS](./02-bev-and-occupancy.md)：点云、鸟瞰图、占用体素，以及 Lift-Splat-Shoot 怎样把图像放进 BEV。
3. [7.3 NeRF、3DGS 与网格](./03-nerf-3dgs-mesh.md)：辐射场、体渲染方程、高斯泼洒和三角面片，三种静态三维表示的取舍。
4. [7.4 四维世界（4D）](./04-four-dimensional-worlds.md)：规范场加变形场、动作条件的 4D，以及会动与会随指定动作动的区别。
5. [7.5 驾驶世界模型与未来占用](./05-driving-world-models.md)：为什么预测的是未来占用而不是视频，以及开环与闭环的边界。
6. [7.6 动手：空间世界实验](./06-spatial-world.md)

静态场景重建不是动态世界模型；没有 ego action 和模拟器的未来预测也不是闭环驾驶。本章会分别标出这些边界。

## 参考资料

### 实践博客（5 篇）

1. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving (Wayve, 2025)](https://wayve.ai/thinking/gaia-2/) —— Wayve 官方博客，展示可控反事实场景怎样服务驾驶评测，配 7.5。
2. [Introducing GAIA-1 (Wayve, 2023)](https://wayve.ai/thinking/introducing-gaia1/) —— GAIA-1 官方博客：视频 token 自回归世界模型在驾驶上的首个完整配方。
3. [The Rise of 3D Gaussian Splatting (Magnopus)](https://www.magnopus.com/blog/the-rise-of-3d-gaussian-splatting/) —— 面向工程师的 3DGS 科普博客，讲清它相对 NeRF 的取舍，配 7.3。
4. [Nerfstudio 官方文档](https://docs.nerf.studio/) —— 模块化 NeRF 框架的文档与教程，是动手做新视角合成的实用入口。
5. [3D Gaussian Splatting 官方项目页 (Kerbl et al.)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) —— 官方可视化 demo 与代码，直观感受显式表示的实时渲染。

### 原始论文（5 篇）

1. [NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) —— 神经辐射场原始论文，配 7.3 的体渲染方程。
2. [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) —— 高斯泼洒原始论文，显式表示与实时渲染的取舍。
3. [Lift, Splat, Shoot: LSS (Philion & Fidler, 2020)](https://arxiv.org/abs/2008.05711) —— 从单目图像升到 BEV 的经典方法，配 7.2。
4. [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images (Li et al., 2022)](https://arxiv.org/abs/2203.17270) —— 多相机 BEV 感知的代表作，展示了时空注意力怎样接历史帧。
5. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving (Wayve, 2025)](https://arxiv.org/abs/2503.20523) —— 可控多视角驾驶世界模型，反事实生成服务评测，配 7.5。
