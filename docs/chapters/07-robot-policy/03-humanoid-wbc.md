# 7.3 人形机器人与全身控制

在上一节中，我们深入剖析了多指灵巧手如何在接触力学与摩擦锥约束下完成手内物体的精细旋转与操作。在那类场景中，机械臂与灵巧手的底座通常被螺栓牢牢固定在实验台或移动底盘之上。在经典控制理论中，这被称为**固定基座系统（Fixed-Base System）**——机械臂的基坐标系在惯性参考系中是绝对静止或完全已知的。

然而，当我们把目光投向拥有双腿、双臂与躯干的**人形机器人（Humanoid Robot）**时，整个物理图景发生了根本性的颠覆：机器人的躯干漂浮在三维空间中，没有任何螺栓将其与大地固定。它的双脚在地面上交替踏步、腾空、着地，身体的全部重量与运动加速度必须仅仅依靠脚底与地面那几块极其有限的接触区域来支撑。这类系统被称为**浮动基座系统（Floating-Base System）**。

如何指挥一台拥有数十个自由度、在重力场中时刻面临翻倒倾覆风险的人形机器人，既保持双足动态平衡，又能同时伸出手臂精准端起水杯？这正是本节的核心主题——**全身控制（Whole-Body Control, WBC）**。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/kuind-fig1.png" alt="Atlas 在障碍、泥地和坡面上行走，展示全身约束必须同时满足。" width="86%">

_图 7.3-1：Atlas 在障碍、泥地和坡面上行走，展示全身约束必须同时满足。 出处：[Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot，Scott Kuindersma et al.，2014](https://arxiv.org/abs/1311.1839)。_

</div>

---

## 7.3.1 物理与生理基石：双足平衡反射与浮动基座动力学

要让人形机器人稳健站立与行走，我们首先必须回到生物进化生理学与经典牛顿-拉格朗日力学的起点。

### 1. 人类双足直立行走的生理奇迹
人类是地球上极少数能够实现高效、长距离直立双足行走的哺乳动物。直立行走赋予了人类解放双手的巨大生存优势，但在物理力学上却极其危险：
- **高质心与窄支撑区**：人类身体的质心（Center of Mass, CoM）位于肚脐深处的骨盆上方（离地约占身高的 $55\% \sim 58\%$），而双脚着地时构成的物理支撑底面积（Support Polygon）却非常狭小。这种“上重下轻、立于窄基”的构型，在经典力学中属于天然的倒立摆不稳定系统；
- **前庭系统与姿态反射**：人类内耳前庭器官中，三个互相垂直的**半规管**（Semicircular Canals）负责实时检测头部的三维角加速度；两个**耳石器**（球囊与椭圆囊）则利用碳酸钙结晶的惯性位移感知头部相对于重力矢量的倾斜角度与三维线性加速度。
- **小脑的高频协调**：内耳前庭信号、足底深层机械感受器（压力与剪切感应）以及视觉光流信号以数十毫秒的时延汇聚至小脑。小脑以前馈预补偿与闭环反射调控全身数百块骨骼肌的协同收缩，使得人类在奔跑跳跃中即使遭遇微小绊倒，也能瞬间调整关节刚度与步幅恢复平衡。

在物理本质上，**双足直立行走并不是静态的“永久稳固”，而是一个“在失衡倒下之前迅速迈出下一步支撑托起”的周期性动态极限环过程**。

### 2. 经典力学起点：从单倒立摆到零力矩点（ZMP）
在经典物理学中，我们早已熟悉了杠杆原理与力矩平衡。如果一个物体的支撑点位于质心下方，只要质心略微偏离竖直铅垂线，重力就会产生一个让物体进一步倾倒的加速翻转力矩。

为了在基础力学框架内直观量化这种倾覆临界状态，经典机器人学引入了**单质点线性倒立摆模型（Linear Inverted Pendulum Model, LIPM）**（将整机质量近似集中于质心点，且假定质心高度 $z_c$ 近似恒定）。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/sentis-fig9.png" alt="自由浮动人形的支撑接触与反作用力说明基座运动由接触间接产生。" width="86%">

_图 7.3-2：自由浮动人形的支撑接触与反作用力说明基座运动由接触间接产生。 出处：[A Whole-Body Control Framework for Humanoids Operating in Human Environments，Luis Sentis; Oussama Khatib，2006](https://doi.org/10.1109/ROBOT.2006.1642100)。_

</div>

设机器人的总质量为 $m$，质心高度固定为 $z_c$，重力加速度为 $g$。当质心向前以加速度 $\ddot{x}_c$ 运动时，水平推力为 $F_x = m \ddot{x}_c$，竖直支撑力为 $F_z = m g$。

在地面接触面上，必然存在一个使得地面接触力产生的**水平净翻转力矩完全抵消为零的作用点**——**零力矩点（Zero Moment Point, ZMP）**。

列出围绕该点的力矩平衡方程 $\tau_y = F_z (x_c - x_{\text{zmp}}) - F_x z_c = 0$，将 $F_x, F_z$ 代入并约去质量 $m$，即可解出极具几何直觉的 ZMP 坐标公式：

$$x_{\text{zmp}} = x_c - \frac{z_c}{g} \ddot{x}_c$$

$$y_{\text{zmp}} = y_c - \frac{z_c}{g} \ddot{y}_c$$

> **公式符号逐一拆解**：
> - $x_{\text{zmp}}$：地面零力矩点在 $x$ 轴上的位置坐标（单位：米 $\text{m}$）；
> - $x_c$：机器人质心在 $x$ 轴上的位置坐标（单位：米 $\text{m}$）；
> - $z_c$：机器人质心距离地面的恒定高度（单位：米 $\text{m}$）；
> - $g$：地球表面重力加速度常数（$9.8\text{ m/s}^2$）；
> - $\ddot{x}_c$：机器人质心向前运动的瞬时加速度（单位：$\text{m/s}^2$）。

**手算代入算例**：
设一台人形机器人的质量 $m = 60\text{ kg}$，质心高度为 $z_c = 0.8\text{ m}$，重力加速度 $g = 9.8\text{ m/s}^2$。当机器人从静止向前加速，质心加速度达到 $\ddot{x}_c = 1.225\text{ m/s}^2$。

我们计算 ZMP 偏离质心投影的距离 $\Delta x = x_c - x_{\text{zmp}}$：

$$\Delta x = \frac{z_c}{g} \ddot{x}_c = \frac{0.8\text{ m}}{9.8\text{ m/s}^2} \times 1.225\text{ m/s}^2 = \frac{0.8 \times 1.225}{9.8}\text{ m} = 0.10\text{ m} = 10\text{ cm}$$

如果单脚脚掌从脚后跟到脚尖的总长度为 $24\text{ cm}$（质心投影位于脚掌中心时，脚尖与脚后跟各有 $12\text{ cm}$ 裕度）。由于 $10\text{ cm} \le 12\text{ cm}$，计算出的 ZMP 依然落在脚底物理接触多边形内部，机器人双脚绝不会翘起翻倒！

<details>
<summary><b>深入推导：三维单质点线性倒立摆（LIPM）的状态空间微分方程与轨道能量守恒（点击展开查看完整推导）</b></summary>

将 ZMP 公式变形，得到质心运动的二阶线性非齐次微分方程：
$$\ddot{x}_c = \frac{g}{z_c}(x_c - x_{\text{zmp}})$$
定义自然固有频率 $\omega_0 = \sqrt{\frac{g}{z_c}}$（例如 $z_c = 0.8\text{ m}$ 时，$\omega_0 = \sqrt{9.8/0.8} \approx 3.5\text{ rad/s}$）。方程化为：
$$\ddot{x}_c - \omega_0^2 x_c = -\omega_0^2 x_{\text{zmp}}$$
对于常数控制输入的 $x_{\text{zmp}}$，该双曲特征方程的通解为：
$$x_c(t) = (x_c(0) - x_{\text{zmp}})\cosh(\omega_0 t) + \frac{\dot{x}_c(0)}{\omega_0}\sinh(\omega_0 t) + x_{\text{zmp}}$$
同时定义倒立摆的**轨道能量（Orbital Energy）**：
$$E = \frac{1}{2}\dot{x}_c^2 - \frac{g}{2z_c}(x_c - x_{\text{zmp}})^2 = \text{常数}$$
当 $E > 0$ 时质心将越过支撑点继续向前跨步，当 $E < 0$ 时质心将在支撑点前减速折返，当 $E = 0$ 时质心恰好在支撑点正上方平稳渐近停止。
</details>

### 3. 浮动基座欧拉-拉格朗日多刚体动力学
单质点倒立摆模型是一种高度精简的力学近似。对于真实的人形机器人，全身数十个转动连杆具有复杂的质量分布，系统满足宏大的多刚体动力学方程：

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \mathbf{S}^\top \boldsymbol{\tau} + \sum_{k=1}^{N_c} \mathbf{J}_{c,k}^\top \mathbf{f}_k$$

直观上，该方程就是经典牛顿第二定律（“质量 $\times$ 加速度 = 合外力”）在高维多连杆系统下的向量化推广：
- $\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}$：各关节加速运动所需的惯性力；
- $\mathbf{C}\dot{\mathbf{q}} + \mathbf{G}$：高速旋转产生的离心力/科氏力阻力与重力矩；
- $\mathbf{S}^\top \boldsymbol{\tau}$：电机实际施加的主动驱动力矩；
- $\sum \mathbf{J}_{c,k}^\top \mathbf{f}_k$：脚掌受到地面反作用力传递到全身各处的支撑力矩。

特别注意选择矩阵 $\mathbf{S}^\top \boldsymbol{\tau} = [\mathbf{0}_{6 \times 1}^\top, \boldsymbol{\tau}^\top]^\top$ 的前 6 行全为 0。这揭示了人形机器人的**欠驱动本质**——躯干基座没有任何电机直接推拉，其前进或腾空必须全部通过脚掌挤压地面产生的接触反作用力间接驱动。

<details>
<summary><b>深入推导：浮动基座欧拉-拉格朗日动力学方程的变分法与分块矩阵结构（点击展开查看完整推导）</b></summary>

定义系统的动能 $T(\mathbf{q}, \dot{\mathbf{q}}) = \frac{1}{2}\dot{\mathbf{q}}^\top \mathbf{M}(\mathbf{q})\dot{\mathbf{q}}$ 与势能 $V(\mathbf{q})$，拉格朗日函数为 $L = T - V$。
根据欧拉-拉格朗日方程：
$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\mathbf{q}}}\right) - \frac{\partial L}{\partial \mathbf{q}} = \boldsymbol{\Xi}_{\text{ext}}$$
其中广义坐标拆分为 6 维浮动基座坐标 $\mathbf{q}_b \in \text{SE}(3)$ 与 $n$ 维受控关节转角 $\mathbf{q}_a \in \mathbb{R}^n$。
惯性矩阵分块展开为：
$$\begin{bmatrix} \mathbf{M}_{bb}(\mathbf{q}) & \mathbf{M}_{ba}(\mathbf{q}) \\ \mathbf{M}_{ab}(\mathbf{q}) & \mathbf{M}_{aa}(\mathbf{q}) \end{bmatrix} \begin{bmatrix} \ddot{\mathbf{q}}_b \\ \ddot{\mathbf{q}}_a \end{bmatrix} + \begin{bmatrix} \mathbf{h}_b(\mathbf{q}, \dot{\mathbf{q}}) \\ \mathbf{h}_a(\mathbf{q}, \dot{\mathbf{q}}) \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \boldsymbol{\tau} \end{bmatrix} + \begin{bmatrix} \mathbf{J}_{c,b}^\top \\ \mathbf{J}_{c,a}^\top \end{bmatrix} \mathbf{f}_c$$
第一行方程 $\mathbf{M}_{bb}\ddot{\mathbf{q}}_b + \mathbf{M}_{ba}\ddot{\mathbf{q}}_a + \mathbf{h}_b = \mathbf{J}_{c,b}^\top \mathbf{f}_c$ 无直接驱动力矩输入，严格约束了接触反作用力 $\mathbf{f}_c$ 与基座加速度 $\ddot{\mathbf{q}}_b$ 的动力学一致性。
</details>

---

## 7.3.2 半个世纪的经典工程探索：从 WABOT、ASIMO 到现代 WBC

人类为了征服浮动基座与双足平衡这道力学难关，经历了半个世纪波澜壮阔的工程探索。

### 1. 从静态步行到基于 ZMP 的预设轨迹控制
- **20 世纪 60 至 70 年代（萌芽期）**：1973 年，日本早稻田大学加藤一郎（Ichiro Kato）教授团队研发出了世界上第一台全尺寸人形机器人 **WABOT-1**。受限于当时的算力与理论，WABOT-1 采用极其缓慢的“静态平衡步态”，每走一步都需要先把整体质心完全平移到单腿支撑面内，走一步耗时长达 45 秒。与此同时，前南斯拉夫科学家 Miomir Vukobratović（1968/1972）发表了划时代的零力矩点（ZMP）理论，为动态双足步态分析奠定了理论基石；
- **20 世纪 90 年代至 2000 年代初（ASIMO 时代）**：日本本田公司（Honda）秘密研发十余年，相继推出 P2、P3 并在 2000 年推出了举世瞩目的 **ASIMO** 机器人。ASIMO 采用基于 ZMP 的离线轨迹规划器：工程师预先计算出一条满足 ZMP 稳定条件的理想质心与足底轨迹，并在机器人内部运行高精度的关节位置跟踪 PID 控制器与在线地面倾角姿态稳定器。ASIMO 实现了平稳步行、小跑与上下楼梯，成为经典控制时代的人形机器人巅峰。

### 2. 传统预设步态方法的硬性极限
然而，当研究人员试图让这类基于 ZMP 轨迹预设的人形机器人走出平整的实验室展厅，走向崎岖碎石、泥泞斜坡或遭遇外界人员突然推搡时，传统方法的致命弱点暴露无遗：
1. **强平整地面假设的瓦解**：ZMP 理论高度依赖地面为刚性平面的几何假设。在台阶边缘、碎石或松软泥地上，足底与地面发生点线接触，支撑多边形瞬间退化为一条线或一个点，离线规划的轨迹瞬间失效；
2. **高刚度位置控制的脆性**：为了高精度追踪预设步态，ASIMO 等机器人的关节电机减速比极高、位置刚度极大。当脚底遇到凸起障碍物时，刚性位置控制会引发巨大的地面冲击反作用力，导致电机剧烈过载震荡甚至直接震坏减速器；
3. **无法实现全身多任务动态协调**：如果机器人在行走的同时需要用手臂去接住飞来的重物或推开一扇沉重的铁门，传统方法无法动态在“保持身体平衡”与“手臂发力操作”之间进行瞬时力学分配。

这一系列物理困境，催生了以斯坦福大学 Oussama Khatib 教授为代表提出的**任务空间控制（Operational Space Control, OSC）**以及融合接触力学优化的现代**全身控制（Whole-Body Control, WBC）**。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/sentis-fig3.png" alt="任务、约束和姿态原语按优先级投影，直观呈现全身控制层级。" width="86%">

_图 7.3-3：任务、约束和姿态原语按优先级投影，直观呈现全身控制层级。 出处：[A Whole-Body Control Framework for Humanoids Operating in Human Environments，Luis Sentis; Oussama Khatib，2006](https://doi.org/10.1109/ROBOT.2006.1642100)。_

</div>

---

## 7.3.3 核心数学推导一：雅可比矩阵与任务空间动力学映射

在控制人形机器人时，我们最关心的任务目标往往并不是“某个电机转了多少度”，而是空间中的直观物理任务——例如“右手末端执行器保持在空间坐标 $(x, y, z)$”、“躯干质心高度维持在 $0.8\text{ m}$”或“头部相机朝向前方物体”。这些在三维笛卡尔空间中定义的目标被称为**任务空间（Operational / Task Space）**。

### 1. 速度与力矩的直观几何映射
末端空间位置 $\mathbf{x}$ 与关节角度 $\mathbf{q}$ 的微小变化关系由**雅可比矩阵（Jacobian Matrix）** $\mathbf{J}(\mathbf{q})$ 决定：

$$\dot{\mathbf{x}} = \mathbf{J}(\mathbf{q})\dot{\mathbf{q}}$$

根据经典物理的虚功原理（末端推力做的功等于电机力矩做的总功 $\mathbf{F}^\top \delta \mathbf{x} = \boldsymbol{\tau}^\top \delta \mathbf{q}$），我们可以得到极为优美的力矩映射关系：

$$\boldsymbol{\tau} = \mathbf{J}^\top \mathbf{F}$$

这表明：**只需将任务空间的虚拟期望推力 $\mathbf{F}$ 乘以雅可比矩阵的转置 $\mathbf{J}^\top$，就能直接算出各电机所需输出的关节力矩**！

<details>
<summary><b>深入推导：虚功原理严格证明与任务空间等效惯性矩阵 $\boldsymbol{\Lambda}(\mathbf{q})$ 的严格解析反演（点击展开查看完整推导）</b></summary>

1. **加速度映射**：对 $\dot{\mathbf{x}} = \mathbf{J}\dot{\mathbf{q}}$ 求导得 $\ddot{\mathbf{x}} = \mathbf{J}\ddot{\mathbf{q}} + \dot{\mathbf{J}}\dot{\mathbf{q}}$；
2. **联立关节动力学方程** $\mathbf{M}\ddot{\mathbf{q}} + \mathbf{h} = \boldsymbol{\tau}$（其中 $\mathbf{h} = \mathbf{C}\dot{\mathbf{q}} + \mathbf{G}$）：
   $$\ddot{\mathbf{q}} = \mathbf{M}^{-1}(\boldsymbol{\tau} - \mathbf{h})$$
3. **代入加速度映射并应用静力学对偶 $\boldsymbol{\tau} = \mathbf{J}^\top \mathbf{F}$**：
   $$\ddot{\mathbf{x}} - \dot{\mathbf{J}}\dot{\mathbf{q}} = \mathbf{J}\mathbf{M}^{-1}\mathbf{J}^\top \mathbf{F} - \mathbf{J}\mathbf{M}^{-1}\mathbf{h}$$
4. **定义任务空间等效惯性矩阵** $\boldsymbol{\Lambda}(\mathbf{q}) = (\mathbf{J}\mathbf{M}^{-1}\mathbf{J}^\top)^{-1} \in \mathbb{R}^{k \times k}$，两边同乘 $\boldsymbol{\Lambda}$，即可精确推导出 Khatib 操作空间动力学方程：
   $$\boldsymbol{\Lambda}(\mathbf{q})\ddot{\mathbf{x}} + \boldsymbol{\mu}(\mathbf{q}, \dot{\mathbf{q}}) + \mathbf{p}(\mathbf{q}) = \mathbf{F}$$
   其中 $\boldsymbol{\mu} = \boldsymbol{\Lambda}(\mathbf{J}\mathbf{M}^{-1}\mathbf{C}\dot{\mathbf{q}} - \dot{\mathbf{J}}\dot{\mathbf{q}})$ 为任务空间科氏力项，$\mathbf{p} = \boldsymbol{\Lambda}\mathbf{J}\mathbf{M}^{-1}\mathbf{G}$ 为任务空间重力补偿项。
</details>

---

## 7.3.4 核心数学推导二：动力学一致零空间投影（Null-Space Projection）

人形机器人通常拥有 20 到 40 多个自由度，多出来的自由度构成了**运动冗余（Kinematic Redundancy）**。在实际控制中，机器人需要同时兼顾多个任务：
- **主任务（优先级 1）**：维持躯干平衡与质心高度绝对稳定；
- **次任务（优先级 2）**：右手向前伸展端平托盘；
- **低级任务（优先级 3）**：各关节尽量保持在自然默认姿态并施加阻尼。

<div align="center">

<img src="/figures/07-robot-policy/latex/03-humanoid-wbc/nullspace-secondary-torque.png" alt="主任务力矩与经零空间投影的次任务力矩合流" width="86%">

_图 7.3-4：次任务先经动态一致零空间投影再与主任务合流，因此它在主任务加速度映射中的贡献为零。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

### 1. 零空间投影的直观物理法则
在线性代数中，矩阵 $\mathbf{J}$ 的**零空间（Null Space）**是指所有经过 $\mathbf{J}$ 映射后结果为零的向量集合。

为了让次任务力矩 $\boldsymbol{\tau}_2$ 绝不破坏主任务的平衡，我们构造一个特殊的**零空间投影过滤矩阵** $\mathbf{N} = \mathbf{I} - \overline{\mathbf{J}}\mathbf{J}$（其中 $\overline{\mathbf{J}} = \mathbf{M}^{-1}\mathbf{J}^\top \boldsymbol{\Lambda}$ 为动力学一致伪逆）。

最终下发给电机的总力矩法则为：

$$\boldsymbol{\tau} = \mathbf{J}_1^\top \mathbf{F}_1 + \mathbf{N}_1^\top \boldsymbol{\tau}_2$$

直观上，主任务力矩 $\mathbf{J}_1^\top \mathbf{F}_1$ 拥有最高通行特权；次任务力矩 $\boldsymbol{\tau}_2$ 必须先经过 $\mathbf{N}_1^\top$ 滤除掉所有可能影响主任务平衡的力学分量，然后“静悄悄”地叠加到电机上。

<details>
<summary><b>深入证明：动力学一致零空间投影的严格代数消除与无干涉性证明（点击展开查看无跳步证明）</b></summary>

根据牛顿-欧拉方程，次级力矩 $\mathbf{N}_1^\top \boldsymbol{\tau}_2$ 引起的关节角加速度为：
$$\ddot{\mathbf{q}}_2 = \mathbf{M}^{-1} (\mathbf{N}_1^\top \boldsymbol{\tau}_2)$$
将 $\mathbf{N}_1^\top = \mathbf{I} - \mathbf{J}_1^\top \boldsymbol{\Lambda}_1 \mathbf{J}_1 \mathbf{M}^{-1}$ 代入展开：
$$\ddot{\mathbf{q}}_2 = \left(\mathbf{M}^{-1} - \mathbf{M}^{-1}\mathbf{J}_1^\top \boldsymbol{\Lambda}_1\mathbf{J}_1\mathbf{M}^{-1}\right)\boldsymbol{\tau}_2$$
通过主任务雅可比 $\mathbf{J}_1$ 计算其在主任务空间产生的加速度贡献 $\ddot{\mathbf{x}}_{1, \text{from } 2}$：
$$\ddot{\mathbf{x}}_{1, \text{from } 2} = \mathbf{J}_1 \ddot{\mathbf{q}}_2 = \left(\mathbf{J}_1\mathbf{M}^{-1} - (\mathbf{J}_1\mathbf{M}^{-1}\mathbf{J}_1^\top)\boldsymbol{\Lambda}_1\mathbf{J}_1\mathbf{M}^{-1}\right)\boldsymbol{\tau}_2$$
根据定义 $(\mathbf{J}_1\mathbf{M}^{-1}\mathbf{J}_1^\top)\boldsymbol{\Lambda}_1 = \boldsymbol{\Lambda}_1^{-1}\boldsymbol{\Lambda}_1 = \mathbf{I}$，代回可得：
$$\ddot{\mathbf{x}}_{1, \text{from } 2} = \left(\mathbf{J}_1\mathbf{M}^{-1} - \mathbf{I} \cdot \mathbf{J}_1\mathbf{M}^{-1}\right)\boldsymbol{\tau}_2 = (\mathbf{J}_1\mathbf{M}^{-1} - \mathbf{J}_1\mathbf{M}^{-1})\boldsymbol{\tau}_2 = \mathbf{0}$$
代数消除完成！这证明了次任务力矩在主任务空间引起的加速度恒等于零。
</details>

> **想一想**
>
> 上述零空间投影在解析上极其完美。但在真实物理机器人上，如果次级任务需要的力矩极大，导致算出的总力矩 $\boldsymbol{\tau} = \boldsymbol{\tau}_1 + \mathbf{N}^\top \boldsymbol{\tau}_2$ 超出了电机物理能够承受的最大扭矩限幅（$\boldsymbol{\tau}_{\max}$），或者导致足底接触力超出了三维摩擦锥边界，系统会发生什么？
>
> **解答**：此时电机硬件会发生饱和截断（Clipping）。一旦实际力矩被强行截断，原本精巧平衡的零空间正交抵消条件瞬间被破坏，次级任务的残余力矩会立刻严重污染主任务，导致机器人质心失衡摔倒。解析零空间方法无法在公式中显式处理电机力矩限幅、足底摩擦锥等大量“小于等于”的不等式约束。为了解决这一痛点，现代 WBC 全面转向了基于二次规划的优化方法。

---

## 7.3.5 现代优化架构：基于二次规划的全身控制（QP-based WBC）

在真实世界中，机器人必须严格遵守一系列刚性的**物理不等式约束**：
1. **电机扭矩限幅**：$\boldsymbol{\tau}_{\min} \le \boldsymbol{\tau} \le \boldsymbol{\tau}_{\max}$；
2. **摩擦锥不打滑**：$\sqrt{F_{x,k}^2 + F_{y,k}^2} \le \mu_k F_{z,k}$ 且 $F_{z,k} \ge 0$。

为了在毫秒级时间内兼顾多任务追踪与全套物理不等式，现代 WBC 将控制求解转化为一个凸**二次规划（Quadratic Programming, QP）**问题。

<div align="center">

<img src="/figures/07-robot-policy/source/03-humanoid-wbc/kuind-fig3.png" alt="多面体摩擦锥近似把接触可行域转化为 QP 可处理的线性约束。" width="86%">

_图 7.3-5：多面体摩擦锥近似把接触可行域转化为 QP 可处理的线性约束。 出处：[Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot，Scott Kuindersma et al.，2014](https://arxiv.org/abs/1311.1839)。_

</div>

### 1. 概念模型与物理优化目标
在每个控制周期（通常为 $1\text{ ms}$，即 $1000\text{ Hz}$），机器人求解如下优化：

$$\begin{aligned}
\min_{\ddot{\mathbf{q}}, \mathbf{f}_c, \boldsymbol{\tau}} \quad & \sum_{i} w_i \|\text{任务空间加速度追踪误差}\|_2^2 + w_\tau \|\text{电机力矩能耗}\|_2^2 \\
\text{s.t.} \quad & \mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{G} = \mathbf{S}^\top \boldsymbol{\tau} + \mathbf{J}_c^\top \mathbf{f}_c \quad & (\text{全系统多刚体动力学严格等式}) \\
& \mathbf{f}_c \in \text{摩擦锥可行域} \quad & (\text{接触防滑边界}) \\
& \boldsymbol{\tau}_{\min} \le \boldsymbol{\tau} \le \boldsymbol{\tau}_{\max} \quad & (\text{电机硬件保护边界})
\end{aligned}$$

<details>
<summary><b>深入推导：四棱锥多面体摩擦锥矩阵构造与凸二次规划（QP）标准型装配（点击展开查看完整公式）</b></summary>

1. **四棱锥线性不等式**：将圆锥近似为内接四棱锥：
   $$\begin{cases} |F_x| \le \frac{\mu}{\sqrt{2}} F_z \\ |F_y| \le \frac{\mu}{\sqrt{2}} F_z \\ F_z \ge 0 \end{cases} \implies \mathbf{A}_{\text{cone}} \mathbf{f}_k \le \mathbf{0}, \quad \mathbf{A}_{\text{cone}} = \begin{bmatrix} 1 & 0 & -\mu/\sqrt{2} \\ -1 & 0 & -\mu/\sqrt{2} \\ 0 & 1 & -\mu/\sqrt{2} \\ 0 & -1 & -\mu/\sqrt{2} \\ 0 & 0 & -1 \end{bmatrix}$$
2. **标准 QP 矩阵形式**：设决策变量 $\mathbf{y} = [\ddot{\mathbf{q}}^\top, \mathbf{f}_c^\top, \boldsymbol{\tau}^\top]^\top$，QP 问题写为标准型：
   $$\min_{\mathbf{y}} \frac{1}{2}\mathbf{y}^\top \mathbf{H} \mathbf{y} + \mathbf{g}^\top \mathbf{y} \quad \text{s.t.} \quad \mathbf{A}_{\text{eq}}\mathbf{y} = \mathbf{b}_{\text{eq}}, \quad \mathbf{A}_{\text{ineq}}\mathbf{y} \le \mathbf{b}_{\text{ineq}}$$
   利用高效的在线活动集法（Active-Set）或算子分裂锥规划（OSQP），可在 $0.2 \sim 0.8\text{ ms}$ 内获得全局最优力矩解。
</details>

---

## 7.3.6 纯底层 PyTorch 代码实现：全身任务空间与零空间投影引擎

下面我们使用纯底层 PyTorch 张量算子实现一个结构完整的全身控制任务空间与动力学一致零空间投影引擎，并进行严格的数值精度与动力学解耦单测。

```python
import torch

class WholeBodyNullSpaceEngine:
    """
    纯底层 PyTorch 全身控制任务空间与零空间投影引擎
    实现了基于动力学一致伪逆 (Dynamically Consistent Pseudo-inverse) 的多任务分层控制。
    """
    def __init__(self, num_dof: int, eps: float = 1e-6):
        """
        初始化全身控制引擎
        :param num_dof: 机器人广义坐标自由度数 (浮动基座 + 关节数)
        :param eps: 正则化阻尼系数，防止矩阵求逆陷入数值奇异
        """
        self.n = num_dof
        self.eps = eps

    def compute_dynamically_consistent_nullspace(
        self, M: torch.Tensor, J: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算任务空间惯性矩阵 Lambda、动力学一致伪逆 J_bar 与零空间投影矩阵 N
        :param M: (n, n) 系统的广义惯性矩阵
        :param J: (k, n) 任务空间的运动学雅可比矩阵
        :return: (Lambda, J_bar, N)
        """
        # 1. 计算广义惯性矩阵的逆 M^-1: (n, n)
        M_inv = torch.linalg.inv(M)

        # 2. 计算任务空间等效逆惯性矩阵: Lambda^-1 = J * M^-1 * J^T, 形状 (k, k)
        Lambda_inv = J @ M_inv @ J.T
        k = J.shape[0]

        # 为保证浮点数值绝对稳定，加入微小阻尼项后求逆得到 Lambda: (k, k)
        damping = self.eps * torch.eye(k, device=M.device, dtype=M.dtype)
        Lambda = torch.linalg.inv(Lambda_inv + damping)

        # 3. 计算动力学一致伪逆: J_bar = M^-1 * J^T * Lambda, 形状 (n, k)
        J_bar = M_inv @ J.T @ Lambda

        # 4. 计算零空间投影矩阵: N = I - J_bar * J, 形状 (n, n)
        I = torch.eye(self.n, device=M.device, dtype=M.dtype)
        N = I - J_bar @ J

        return Lambda, J_bar, N

    def compute_hierarchical_torques(
        self,
        M: torch.Tensor,
        J1: torch.Tensor,
        F1: torch.Tensor,
        J2: torch.Tensor,
        F2: torch.Tensor,
        tau_posture: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        分层多任务全身力矩解算
        :param M: (n, n) 广义惯性矩阵
        :param J1: (k1, n) 主任务雅可比 (如质心与躯干位姿保持)
        :param F1: (k1,) 主任务空间控制合力
        :param J2: (k2, n) 次任务雅可比 (如手臂末端操作)
        :param F2: (k2,) 次任务空间控制合力
        :param tau_posture: (n,) 低优先级关节姿态阻尼力矩
        :return: (tau_total, N1) 最终合成力矩与主任务零空间投影矩阵
        """
        # 1. 解算主任务动力学与零空间
        Lambda1, J1_bar, N1 = self.compute_dynamically_consistent_nullspace(M, J1)
        # 主任务静力学映射: tau_1 = J1^T * F1, 形状 (n,)
        tau_1 = J1.T @ F1

        # 2. 解算次任务并向主任务零空间投影
        tau_2_raw = J2.T @ F2
        # tau_2_projected = N1^T * tau_2_raw, 形状 (n,)
        tau_2_proj = N1.T @ tau_2_raw

        # 3. 姿态与阻尼任务（最低优先级，投影到零空间）
        if tau_posture is not None:
            tau_posture_proj = N1.T @ tau_posture
        else:
            tau_posture_proj = torch.zeros(self.n, device=M.device, dtype=M.dtype)

        # 4. 最终全任务力矩叠加
        tau_total = tau_1 + tau_2_proj + tau_posture_proj
        return tau_total, N1

# ===================================================================
# 单元测试与动力学解耦正交性验证
# ===================================================================
if __name__ == "__main__":
    # 模拟一个 7 自由度的动力学系统
    n_dof = 7
    engine = WholeBodyNullSpaceEngine(num_dof=n_dof)

    # 构造一个严格对称正定的惯性矩阵 M = A * A^T + 0.5 * I
    torch.manual_seed(42)
    A = torch.randn(n_dof, n_dof)
    M = A @ A.T + 0.5 * torch.eye(n_dof)

    # 主任务：控制 3 维笛卡尔空间质心 (k1 = 3)
    J1 = torch.randn(3, n_dof)
    F1 = torch.tensor([15.0, -10.0, 50.0]) # 虚拟推力

    # 次任务：控制 2 维手臂末端姿态 (k2 = 2)
    J2 = torch.randn(2, n_dof)
    F2 = torch.tensor([5.0, -2.0])

    # 姿态阻尼任务
    tau_post = torch.randn(n_dof) * 2.0

    # 解算全身多任务合成力矩
    tau_total, N1 = engine.compute_hierarchical_torques(M, J1, F1, J2, F2, tau_post)

    print(f"[WBC Test] 自由度数目: {n_dof}")
    print(f"[WBC Test] 合成关节力矩形状: {tau_total.shape}")
    print(f"[WBC Test] 合成关节力矩数值: {tau_total.numpy().round(3)}")

    # ---------------------------------------------------------------
    # 核心物理验证：检验次任务力矩在主任务空间产生的加速度是否为 0
    # acc_1_from_task2 = J1 * M^-1 * (N1^T * tau_2_raw)
    # ---------------------------------------------------------------
    M_inv = torch.linalg.inv(M)
    tau_2_raw = J2.T @ F2
    acc_1_from_task2 = J1 @ M_inv @ (N1.T @ tau_2_raw)
    max_interference = acc_1_from_task2.abs().max().item()

    print(f"[WBC Test] 次任务对主任务产生的加速度最大干涉: {max_interference:.6e}")
    assert max_interference < 1e-4, "动力学零空间正交解耦失败，次任务破坏了主任务！"
    print("✓ 动力学解耦与零空间正交性单测全部通过！")
```

---

## 7.3.7 现代演进：全身控制与强化学习/世界模型的融合前沿

在具身智能的最新演进中，**全身控制（WBC）**与**深度强化学习（RL）/世界模型（World Models）**并非相互替代的对立关系，而是形成了极具威力的**分层融合架构**：

1. **底层高频 WBC（500 ~ 1000 Hz）——“物理安全守护者”**：
   - 负责直接与硬件电机交互；
   - 严格求解二次规划，实时确保每一个毫秒内的接触力不脱离摩擦锥、电机不发生力矩超限、受到突发强碰撞时通过阻抗控制柔顺卸力；
2. **顶层低频策略/世界模型（10 ~ 50 Hz）——“高级大脑规划者”**：
   - 负责处理多模态相机图像与语言指令，利用世界模型推演未来物理轨迹；
   - 策略网络不直接输出底层的微观电机电流，而是输出高层的**任务空间目标**（如期望质心速度 $\mathbf{v}_{\text{cmd}}$、落脚点坐标 $(x_{\text{foot}}, y_{\text{foot}})$ 与末端操作旋量）。

这种“顶层大模型构想意图，底层 WBC 严守物理规律”的分层体系，成功兼顾了端到端智能的泛化创造力与经典动力学控制的确定性安全保障。

---

## 7.3.8 本节小结

回顾本节内容，我们建立了一条从单倒立摆力学走向高维浮动基座全身优化的完整体系：
1. **浮动基座与欠驱动本质**：人形机器人的躯干无任何外力锚定，其空间运动必须完全依赖关节驱动配合地面接触反作用力间接产生；
2. **零力矩点（ZMP）的物理判据**：只要合外力在该点处的水平净力矩为零且落在足底支撑多边形内部，机器人便具备抵抗倾覆的力学平衡；
3. **任务空间与雅可比转置映射**：雅可比矩阵将关节速度映射到任务空间，其转置 $\mathbf{J}^\top$ 将空间虚拟推力直接转化为电机关节力矩；
4. **动力学一致零空间投影**：利用广义惯性矩阵 $\mathbf{M}$ 构建的投影矩阵 $\mathbf{N} = \mathbf{I} - \overline{\mathbf{J}}\mathbf{J}$，在代数上完美保证了次级操作任务对主平衡任务的零动力学干涉；
5. **二次规划（QP-WBC）的工程落地**：通过多面体摩擦锥近似与在线凸优化，QP 架构使机器人在毫秒级实时解算出满足全套物理不等式约束的全局最优全身力矩。
