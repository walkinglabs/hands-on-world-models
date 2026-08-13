# 第 7 章　空间世界：怎样保持三维结构与运动一致

相机移动以后，图片中的每个像素都可能变化，房间和道路却没有随之移动。本章研究怎样用相机几何和空间表示保持跨视角、跨时间的一致性。

## 本章文章

1. [相机怎样把三维世界变成图片](./07-01-camera-geometry.md)
2. [点云、BEV、LSS 与 Occupancy](./07-02-bev-and-occupancy.md)
3. [NeRF、3DGS 与 Mesh](./07-03-nerf-3dgs-mesh.md)
4. [从静态三维重建到 4D 世界](./07-04-four-dimensional-worlds.md)
5. [自动驾驶为什么预测未来占用](./07-05-driving-world-models.md)

## 本章实验与作业

- [E1–E2：空间路线实验](/labs/route-de)
- [PA1-E：3D/4D 或驾驶二选一](/assignments/pa1-e)

静态场景重建不是动态世界模型；没有 ego action 和模拟器的未来预测也不是闭环驾驶。本章会分别标出这些边界。
