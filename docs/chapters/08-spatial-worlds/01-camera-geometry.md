# 8.1　相机模型与多视角投影

> **第 8 章 · 空间世界与自动驾驶**
>
> 照片里的像素会随相机移动而变化，房间和道路却不动。要把跨视角、跨时间的观察拼成一个一致的空间，就需要相机几何与空间表示。
>
> 本章从一张照片怎样投影开始，走到 BEV 和占用网格，再走到 NeRF 与 3DGS 这类连续三维表示，最后把时间加进来，落到驾驶世界模型对未来占用的预测。
>
> 静态场景重建不是动态世界模型；没有 ego action 和模拟器的未来预测也不是闭环驾驶。本章会分别标出这些边界。
>
> 动手实验：[8.6 动手：占用网格预测的从零开始实现](/chapters/08-spatial-worlds/06-occupancy-prediction)；本章收尾是 [8.7 动手：驾驶场景下的四维世界模型](/chapters/08-spatial-worlds/07-four-d-driving)。

桌上放着一个杯子。我们从左边拍一张，杯子出现在照片右侧；再从右边拍一张，同一个杯子又跑到照片左侧。

照片里的像素位置会随相机移动而改变，杯子本身却没有动。要把多张照片放进同一个房间，就需要一条规则：给定一个三维点和一个相机，它在照片上的哪个像素。

## 针孔相机：三维点怎样落到像素上

最简单的相机模型是针孔相机。它假设光线通过一个小孔，在成像面上投出一张倒立的图。我们暂时不管倒立，只关心一个点 $(x,y,z)$ 在相机坐标里怎样变成像素 $(u,v)$。

规则只有两步。先把三维点沿深度方向归一化，得到它在成像面上的位置 $\bigl(\tfrac{x}{z},\,\tfrac{y}{z}\bigr)$；再乘上焦距并加上成像中心，得到像素坐标：

$$
u = f_x\,\frac{x}{z} + c_x,\qquad v = f_y\,\frac{y}{z} + c_y.
$$

四个数 $f_x,f_y,c_x,c_y$ 称为相机的**内参**。$f_x,f_y$ 描述焦距（像素单位），$c_x,c_y$ 是成像中心，通常接近图片的中央。内参只和相机本身有关，跟它被搬到哪里无关。

用一个具体数字检查。取相机前方 $z=4$ 米处的一个点 $(0,\,0,\,4)$，焦距 $f_x=f_y=600$ 像素，中心 $c_x=c_y=320$ 像素：

$$
u = 600\cdot\frac{0}{4}+320 = 320,\qquad v = 600\cdot\frac{0}{4}+320 = 320.
$$

正前方的点落在图片正中央。若把点向右挪到 $(2,\,0,\,4)$，则 $u=600\cdot\tfrac{2}{4}+320=620$，离中心右移 $300$ 像素。深度越小、离光轴越远，点在图片上跑得越远，这正是透视的来源。

整个投影可以写成矩阵形式，方便批量处理一整张图的所有点：

$$
z\begin{bmatrix}u\\ v\\ 1\end{bmatrix}
=
\underbrace{\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}}_{K}
\begin{bmatrix}x\\ y\\ z\end{bmatrix}.
$$

矩阵 $K$ 就是内参矩阵。等号左边多出的 $z$ 说明这是一个**齐次**方程：真正的像素是除以 $z$ 之后的结果。

## 从像素走回去：一条射线

很多任务要反过来问：照片上这个像素，对应三维空间里的什么。把投影公式解出来，得到

$$
x = \frac{(u-c_x)\,z}{f_x},\qquad y = \frac{(v-c_y)\,z}{f_y}.
$$

这里多了一个未知量 $z$，也就是该点的深度。照片本身没有给深度，所以**一个像素并不能确定一个三维点，只能确定一条从相机出发的射线**。射线上的所有点投影到同一个像素：

$$
(x,y,z) = \lambda\left(\frac{u-c_x}{f_x},\,\frac{v-c_y}{f_y},\,1\right),\quad \lambda>0.
$$

要在这条射线上锁定具体位置，必须额外提供深度。深度可以来自 LiDAR、来自立体相机、来自学习的深度估计，也可以来自下一节的多视角融合。没有深度，所谓的"三维重建"就只是射线束。

## 外参：把相机搬进世界

真实的机器人不会只有一个相机，相机还会随车移动。内参只管成像，不管相机被放在哪里。描述相机在世界中位置和朝向的，是**外参**。

外参由一个旋转 $R$ 和一个平移 $\mathbf{t}$ 组成，把相机坐标里的点变到世界坐标：

$$
\mathbf{p}_{\text{world}} = R\,\mathbf{p}_{\text{camera}} + \mathbf{t}.
$$

把 $R$ 和 $\mathbf{t}$ 拼成 $4\times 4$ 的外参矩阵 $T_{\text{cw}}$，用齐次坐标一次相乘即可。多台相机的点云要合并、或一个相机在不同时刻的点要拼成地图，第一步都是先通过各自的外参送进同一个世界坐标系。

标定（calibration）就是估计内参和外参的过程。内参通常出厂后测一次，外参则随相机安装或车辆姿态而变。

## 标定误差会怎样传播

把平移 $\mathbf{t}$ 写错 $0.3$ 米，整个点云就会整体偏移 $0.3$ 米。深度估计偏差、焦距读错、相机姿态抖动、多相机时间不同步，都会以相似方式污染下游空间。

神经网络可能在固定偏差的训练集上适应下来，但它学到的是"如何抵消这个特定的错误几何"，而非"如何理解空间"。换一辆车、换一个安装角度，偏差就变。先验证几何，再训练网络。

8.7 的第一份 Notebook 用一个合成立方体和已知参数做数值检查：投影到像素、再反投影回三维、再做坐标变换，应当回到原点附近。先用数字确认 $K$、$R$、$\mathbf{t}$ 都写对了，再谈学习。

## 小结

- 内参矩阵 $K$ 描述相机怎样成像，外参 $R,\mathbf{t}$ 描述相机在世界中的位置和朝向。
- 一个像素只确定一条射线，要锁定三维点必须额外给出深度。
- 多视角融合前，先统一坐标并检查标定与时间同步，否则错误会顺着几何传播到所有下游表示。

下一篇把这条射线变成一种模型爱用的空间结构：BEV 与占用网格。

---

## 参考资料

### 实践博客

1. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving (Wayve, 2025)](https://wayve.ai/thinking/gaia-2/) —— Wayve 官方博客，展示可控反事实场景怎样服务驾驶评测，配 8.5。
2. [Introducing GAIA-1 (Wayve, 2023)](https://wayve.ai/thinking/introducing-gaia1/) —— GAIA-1 官方博客：视频 token 自回归世界模型在驾驶上的首个完整配方。
3. [The Rise of 3D Gaussian Splatting (Magnopus)](https://www.magnopus.com/blog/the-rise-of-3d-gaussian-splatting/) —— 面向工程师的 3DGS 科普博客，讲清它相对 NeRF 的取舍，配 8.3。
4. [Nerfstudio 官方文档](https://docs.nerf.studio/) —— 模块化 NeRF 框架的文档与教程，是动手做新视角合成的实用入口。
5. [3D Gaussian Splatting 官方项目页 (Kerbl et al.)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) —— 官方可视化 demo 与代码，直观感受显式表示的实时渲染。

### 经典文献

1. [NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) —— 神经辐射场原始论文，配 8.3 的体渲染方程。
2. [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) —— 高斯泼洒原始论文，显式表示与实时渲染的取舍。
3. [Lift, Splat, Shoot: LSS (Philion & Fidler, 2020)](https://arxiv.org/abs/2008.05711) —— 从单目图像升到 BEV 的经典方法，配 8.2。
4. [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images (Li et al., 2022)](https://arxiv.org/abs/2203.17270) —— 多相机 BEV 感知的代表作，展示了时空注意力怎样接历史帧。
5. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving (Wayve, 2025)](https://arxiv.org/abs/2503.20523) —— 可控多视角驾驶世界模型，反事实生成服务评测，配 8.5。
