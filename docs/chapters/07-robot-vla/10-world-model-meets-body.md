# 7.10　动手：把世界模型接上身体

> **本节目标**：完成一次具身智能的完整实验——先用行为克隆训练 Tiny VLA，再训练一个 World-Model Checker 在动作执行前预测后果。最后从三种用法中选一种，把世界模型真正接上一个身体，做出可验证的证据。
>
> **本节代码**：[搭一台小型 VLA](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_robot/build-a-tiny-vla.ipynb) · [行动前检查动作](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/07_robot/check-actions-before-moving.ipynb) · [robot.py](https://github.com/walkinglabs/hands-on-world-models/blob/main/src/hwm/robot.py)
>
> **前置知识**：第 7 章全部。

---

全世界的机器人教材到这里都差不多：接口、模仿、VLA、sim-to-real，一章一章讲完就结束。这一节是本书和它们的差异点——**世界模型不是中间件，是一等公民**。

本节分为两部分。第一部分是一次“小整机”带练，在桌面格子里用两份 Notebook 走通 VLA 与后果检查的完整闭环，让你亲眼看到“换指令后动作确实改变”、“Checker 压下碰撞但进展也变差”等真实实验现象；第二部分是你的毕业设计，用真正的模拟器（或真机）和更大的算力把世界模型装进机器人的身体。

## 第一部分：Tiny VLA 与世界模型检查器（教学版带练）

到这里，世界模型一直待在格子里、赛道上、小方块旁。现在它要伸进真实一点的身体里：一只桌面上的手。机器人需要知道「朝红块走过去，会不会撞上中间那块蓝障碍」。句话其实还是同一句：**在动手之前，先预见动手的后果。**

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-tabletop.png" alt="桌面世界" style="max-width:min(900px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">这就是我们要让模型学会的「世界」：32×32 的深色桌面，白点是抓手，红点和绿点是两个目标，蓝块是障碍。模型从未被告知「白的是手、蓝的不能碰」，它要从示范里自己发现「这句话指向哪一个点、直走会不会撞」。</div>
</div>

### 本次会得到什么

运行结束后，你会得到：

- 160 条桌面示范：图片、8 维状态、中文指令、未来 3 步动作
- state-only 行为克隆的损失：\(0.498 \rightarrow 0.377\)
- Tiny VLA 的 chunk 损失：\(0.525 \rightarrow 0.277\)，逐步 MSE 大约 0.27
- 同一张图换指令后，动作平均差 0.120
- 闭环 12 步：成功率 0.188，平均碰撞 3.281
- 后果模型损失：\(0.977 \rightarrow 0.538\)；64 个必撞场景里，直达碰撞 1.000，重排后 0.328，平均进展 \(-0.036\)

### 怎样运行

两份 Notebook 在：

```text
notebooks/07_robot/build-a-tiny-vla.ipynb
notebooks/07_robot/check-actions-before-moving.ipynb
```

需要 PyTorch。

```bash
python -m pip install -r requirements-neural.txt
```

安装 Jupyter 后，在仓库根目录打开：

```bash
jupyter lab
```

即使暂时不跑 Notebook，也可以先跑测试：

```bash
PYTHONPATH=src python -m unittest tests.test_routes_de -v
```

教学版在 CPU 上几分钟就能跑完。下面每一段数字，都是用 Notebook 里同一套种子、同一次循环数跑出来的。

### 第一步：看清这张桌子

第一份 Notebook「搭一台小型 VLA」的世界不是机械臂，也不是 6-DOF。状态是 8 个数：抓手 \((x, y)\)、红色目标、绿色目标、障碍。动作是平面上的单位方向，每步最多走 \(0.12\)。障碍半径 \(0.13\)，走进去就停在原地，记一次碰撞。

渲染函数把这 8 个数画成 32×32 的图：深红、翠绿、钴蓝、白色四个圆点，底色是接近黑的灰。语言只有两句中文：

```python
INSTRUCTIONS = ("移动到红色目标", "移动到绿色目标")
```

专家策略很朴素：朝当前指令对应的目标走；如果下一步会撞，就比较左右两个垂直方向，选离障碍更远的那个。示范不是完美绕行，但专家自己几乎不撞——160 条里第一步碰撞率是 \(0\)。

```python
from hwm.robot import INSTRUCTIONS, make_tabletop_dataset

data = make_tabletop_dataset(num_samples=160, chunk_size=3, seed=0)
for name, value in data.items():
    print(f'{name:14s}', tuple(value.shape))
print('第一条指令:', INSTRUCTIONS[int(data['instructions'][0])])
```

**运行这一步，你会看到什么？**

```
images         (160, 3, 32, 32)
states         (160, 8)
instructions   (160,)
action_chunks  (160, 3, 2)
next_states    (160, 8)
collisions     (160,)
第一条指令: 移动到绿色目标
```

注意三件事。第一，图片是 \(32\times 32\)，通道在前，不是旧讲义里写过的 \(16\times 16\)。第二，一次存未来三个二维动作，所以 chunk 是 \((160, 3, 2)\)。第三，160 条里红绿指令大约一半一半（76 / 84），大约 95 条把障碍故意放在抓手和目标中间——否则模型永远学不会躲。

**这就是行为克隆的全部原料**：同一时刻对齐图片、指令、本体状态和动作。缺任何一项，后面的 VLA 都只是在猜。

### 第二步：先不看图，只模仿第一步

先不问视觉有没有用。只用 8 维状态加上指令的 one-hot，预测专家的第一个动作。行为克隆的训练目标就是均方误差：

$$
\mathcal{L}_{\text{BC}}
= \mathbb{E}_{(s_t,\, i_t,\, a_t^*)\sim\mathcal{D}}
\bigl\| a_t^* - \pi_\theta(s_t, i_t) \bigr\|_2^2
$$

其中 \(a_t^*\) 是专家第一步，\(\pi_\theta\) 是两层 MLP。第一份 Notebook 不在这一步输出整个 chunk，只检查监督学习能不能从状态里读出方向。

```python
import torch
import torch.nn.functional as F

state_policy = torch.nn.Sequential(
    torch.nn.Linear(8 + 2, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
    torch.nn.Tanh(),
)
instruction_onehot = F.one_hot(data['instructions'], 2).float()
state_input = torch.cat((data['states'], instruction_onehot), dim=-1)
target = data['action_chunks'][:, 0]
opt = torch.optim.Adam(state_policy.parameters(), lr=3e-3)
losses = []
for _ in range(50):
    opt.zero_grad()
    prediction = state_policy(state_input)
    loss = F.mse_loss(prediction, target)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
print('state BC loss:', round(losses[0], 3), '→', round(losses[-1], 3))
```

**运行这一步，你会看到什么？**

```
state BC loss: 0.498 → 0.377
```

损失在降，但只降了一点。8 维状态里已经有目标和障碍的坐标，理论上足够算出专家方向；50 步、160 条、32 个隐单元，还拟合不完。这条曲线的用处不是「状态策略已经够好」，而是后面给 VLA 当对照：如果加上图片和语言，损失还停在 0.38 附近，视觉就没帮上忙。

**一个值得做的实验**：把循环从 50 提到 300，看 state-only 还能降到哪。它通常会继续往下走——说明现在的 0.377 是训练预算，不是任务上限。VLA 要赢的，是同一预算下的对照，不是一条训满的 MLP。

### 第三步：图片、语言、状态一起出三个动作

Tiny VLA 大约 1.1 万个参数。CNN 把 \(32\times 32\) 压成 80 维，Embedding 把两句指令变成 8 维，再和 8 维状态拼在一起，两层 MLP 一次吐出 \(3\times 2\) 个数，最后用 `Tanh` 限制在 \([-1, 1]\)：

$$
\hat A_t
= \tanh\bigl(
    \mathrm{MLP}\bigl(
        [\mathrm{CNN}(o_t);\; e_{i_t};\; s_t]
    \bigr)
\bigr)
\in [-1, 1]^{3\times 2}
$$

损失不再只看第一步，三个未来动作一起回归：

$$
\mathcal{L}_{\text{chunk}}
= \bigl\| \hat A_t - A_t^* \bigr\|_2^2
$$

这就是 action chunk：一次决策覆盖一小段未来，执行时却可以只走第一步、下一步重新看图。第一份 Notebook 的闭环正是这么做的。

```python
from hwm.robot import TinyVLA

model = TinyVLA(chunk_size=3)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
losses = []
for _ in range(60):
    opt.zero_grad()
    chunks = model(data['images'], data['instructions'], data['states'])
    loss = F.mse_loss(chunks, data['action_chunks'])
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
print('multimodal chunk loss:', round(losses[0], 3), '→', round(losses[-1], 3))
print('output:', tuple(chunks.shape))
```

**运行这一步，你会看到什么？**

```
multimodal chunk loss: 0.525 → 0.277
output: (160, 3, 2)
```

60 步之后，整体 chunk MSE 是 0.272。拆开三步看：

```
第 1 步 MSE: 0.281
第 2 步 MSE: 0.266
第 3 步 MSE: 0.269
```

并没有「越远越大」。教学数据里专家三步都很短，复合误差还来不及长出来。不要把「chunk 越长误差越大」写成这条曲线已经证明的事——它是后面值得做的实验，不是眼前的事实。

和 state-only 比：状态策略只拟合第一步，落到 0.377；VLA 拟合三步，落到 0.277。预算差不多，多模态更低。这只能说明图片和语言在这份数据上有用，不能说明闭环已经会抓。

### 第四步：同一张图，换一句指令

如果文字真的参与决策，固定图片和状态，把「去红色」改成「去绿色」，动作应该动一下。

```python
same_image = data['images'][:1].expand(2, -1, -1, -1)
same_state = data['states'][:1].expand(2, -1)
with torch.no_grad():
    two_goals = model(same_image, torch.tensor([0, 1]), same_state)
difference = (two_goals[0] - two_goals[1]).abs().mean()
print('换目标后的动作差异:', round(float(difference), 4))
print('红目标第一步:', [round(x, 3) for x in two_goals[0, 0].tolist()])
print('绿目标第一步:', [round(x, 3) for x in two_goals[1, 0].tolist()])
```

**运行这一步，你会看到什么？**

```
换目标后的动作差异: 0.1203
红目标第一步: [-0.026, 0.215]
绿目标第一步: [-0.063, 0.471]
```

差异不是零。两条第一步都略微向下，绿色那条更用力。这只说明语言进了前向，不说明方向指对了——第一条样本的绿色目标在右下，红色在左上，模型并没有干净地朝两个相反方向走。

**判断标准在这里**：差异为零一定有问题；差异非零只是最低门槛。RT-2 和 OpenVLA 用的是同一条反事实，只是它们的指令来自网页规模的视觉语言，不是两个 Embedding 向量。

### 第五步：loss 降了，手还在桌子上打转

监督损失看的是「这一步像不像专家」。真正要的是：每一步重新看图，执行 chunk 的第一步，走完 12 步之后离目标还有多远、撞了几次。

```python
from hwm.robot import evaluate_vla

test_data = make_tabletop_dataset(32, chunk_size=3, seed=17)
metrics = evaluate_vla(
    model, test_data['states'], test_data['instructions'], max_steps=12
)
for name in ('success_rate', 'mean_collisions',
             'initial_distance', 'final_distance'):
    print(name, round(metrics[name], 3))
```

**运行这一步，你会看到什么？**

```
success_rate     0.188
mean_collisions  3.281
initial_distance 0.375
final_distance   0.332
```

32 条测试里，大约 6 条在 12 步内走进半径 0.15 的成功圈。平均每条撞 3.3 次。距离从 0.375 收到 0.332——动了一点，远没到。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-vla-closedloop.png" alt="VLA 闭环轨迹" style="max-width:min(760px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">seed=17 的第一条测试。指令是「移动到红色目标」，红线是抓手 12 步走出来的轨迹。chunk 损失从 0.525 降到 0.277，这条手却几乎贴着障碍转，没有碰到红点。动作 MSE 下降，不保证闭环成功。</div>
</div>

**这就是第一份 Notebook 真正要你看见的事**：模仿专家的单步，和自己连续走 12 步，不是同一个分布。专家 demonstrator 每一步都根据最新状态改方向；模型一旦偏了，下一帧就是它训练时没见过的图。0.6 里没见过的 \((s, a)\) 会让转移表缺项，这里换成神经网络，缺口还在。

**一个值得做的实验**：把 `max_steps` 从 12 提到 30，再报一次成功率和最终距离。如果距离几乎不动、碰撞继续加，模型不是「差几步到终点」，而是在错误的状态分布里打转。

第一份 Notebook 到此结束。模型已经能从图像、指令和状态吐出动作，但它没有预测动作后果。第二份 Notebook「行动前检查动作」会让另一个模型先检查候选。

### 第六步：后果数据和示范数据不是同一份

直接 VLA 给出动作以后就结束了。第二份 Notebook 训练一个独立的 outcome model：输入当前状态和候选动作，输出下一状态和碰撞 logits。

如果只拿专家动作来训，碰撞标签全是 0——专家本来就不撞。模型会学会一件没用的事：无论怎么走都安全。所以后果数据要故意掺随机动作，一半样本把障碍塞到动作正前方。

```python
from hwm.robot import make_outcome_dataset

data = make_outcome_dataset(400, seed=1)
print('collision ratio:', round(float(data['collisions'].mean()), 3))
```

**运行这一步，你会看到什么？**

```
collision ratio: 0.527
```

400 条里大约一半会撞。语言不进这份数据。语言负责指定目标；世界模型只学习「这么推一下，桌子会变成什么样」。

### 第七步：下一步状态加一次碰撞

后果模型大约 5 千个参数。状态和动作拼成 10 维，两层 64 宽的 MLP 分出两个头：一个回归 8 维下一状态，一个给出碰撞 logit。训练损失是两项相加，权重为 1：

$$
\mathcal{L}_{\text{outcome}}
= \bigl\| \hat s_{t+1} - s_{t+1} \bigr\|_2^2
+ \mathrm{BCEWithLogits}\bigl(\hat c_t,\, c_t\bigr)
$$

源码里没有另设 \(\lambda\)。状态 MSE 和碰撞交叉熵直接加在一起。

```python
from hwm.robot import TabletopOutcomeModel, outcome_loss

model = TabletopOutcomeModel()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
losses = []
for _ in range(80):
    opt.zero_grad()
    loss, state_loss, collision_loss = outcome_loss(
        model, data['states'], data['actions'],
        data['next_states'], data['collisions'],
    )
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
print('outcome loss:', round(losses[0], 3), '→', round(losses[-1], 3))
print('state/collision:',
      round(float(state_loss), 3), round(float(collision_loss), 3))
```

**运行这一步，你会看到什么？**

```
outcome loss: 0.977 → 0.538
state/collision: 0.013 0.525
```

拆开看更清楚：状态 MSE 从 0.284 降到 0.013，几乎学会了「走一步手会到哪」；碰撞 BCE 从 0.694 只降到 0.525，刚好比随机猜好一点。总损失被状态项拖下来了，碰撞头还很糊。后面单点样例选错，就是从这里长出来的。

### 第八步：四个候选，模型选了会撞的那一个

构造一个障碍挡在抓手和红色目标之间的状态。向右是直达，会撞；斜上斜下也擦边；后退安全，但离目标更远。

重排分数把「离目标更近」和「碰撞概率」加在一起：

$$
\mathrm{score}(a)
= -\bigl\| \hat s_{t+1}^{\,\mathrm{grip}} - g \bigr\|_2
- \lambda\, \sigma(\hat c_t),\qquad \lambda = 2
$$

\(g\) 是当前指令对应的目标，\(\sigma\) 是 sigmoid。分数越大越好。Notebook 这一格用默认 \(\lambda=2\)；后面批量评价会把 \(\lambda\) 提到 4，让安全更重。

```python
from hwm.robot import rerank_actions, step_tabletop

state = torch.tensor([0.20, 0.50, 0.85, 0.50, 0.20, 0.85, 0.31, 0.50])
candidates = torch.tensor([
    [1.0,  0.0],
    [0.7, -0.7],
    [0.7,  0.7],
    [-1.0, 0.0],
])
chosen, scores = rerank_actions(
    model, state, instruction=0, candidates=candidates
)
print('scores:', [round(float(x), 3) for x in scores], 'chosen:', chosen)
true_results = [
    step_tabletop(state.numpy(), action.numpy())[1]
    for action in candidates
]
print('真实碰撞:', true_results)
```

**运行这一步，你会看到什么？**

```
scores: [-0.589, -0.702, -0.811, -2.119] chosen: 0
真实碰撞: [True, True, True, False]
预测碰撞概率: [0.098, 0.138, 0.186, 0.756]
direct candidate collision: True
reranked collision: True
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-vla-checker.png" alt="四个候选动作" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">白点是手，蓝块挡在红目标正前方。红线是真实会撞的候选，绿线是后退。模型给直达的碰撞概率只有 0.098，给唯一安全的后退 0.756——判断反了，于是选了直达。</div>
</div>

这一格不是 Notebook 失败。碰撞头本来就没学干净，单点样例又刚好踩在它的盲区上。PA 必须比较真实闭环，不能只信模型自己的评分。

### 第九步：64 个必撞场景，不能靠一个手挑的例子

把障碍固定放在「抓手朝目标走一步」的正前方，采样 64 个这样的场景。候选只有四个：直达、左垂直、右垂直、后退。比较直接执行与重排后的真实碰撞，同时报告每步离目标近了多少。

```python
from hwm.robot import evaluate_reranker

safety = evaluate_reranker(
    model, num_cases=64, seed=23, collision_weight=4.0
)
for name, value in safety.items():
    print(name, round(value, 3))
```

**运行这一步，你会看到什么？**

```
direct_collision_rate    1.000
reranked_collision_rate  0.328
reranked_mean_progress  -0.036
```

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/de-checker-batch.png" alt="批量碰撞对照" style="max-width:min(720px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">直达 64 次全撞，这是构造出来的。重排之后碰撞掉到 0.328，平均进展却是 −0.036——每走一步，离目标更远了一点。安全地停住，并不等于会绕行。</div>
</div>

碰撞确实少了。但平均进展是负的：模型更常选「别往前走」，而不是「从旁边绕过去」。候选里其实有垂直方向，专家策略用的也是这一招；后果模型会躲，Planner 还不会选那条能前进的安全边。

**判断标准在这里**：只报碰撞下降，可以写成「Checker 有用」；把进展一并写上，才会看见它有用在哪、没用在哪。

**一个值得做的实验**：把 `collision_weight` 从 1 改到 8，观察碰撞率和进展怎样对调。权重太大，模型会学会原地不动；太小，又回到直达。第二份 Notebook 默认的 4.0 只是这条曲线上的一个点。

---

## 第二部分：毕业设计——三种用法，选一种

教学版的 Notebook 只是证明了接口连通。它的数据太少，碰撞头太弱，并不能算作“实验”。现在，把第 4 章的想象 rollout 与规划器、第 5 章的动作条件视频、本章的扩散策略等零件装成一台整机，回答一个属于你自己的问题。

## 三种用法，选一种

| 用法 | 一句话 | 世界模型干什么 | 现成的零件 |
| --- | --- | --- | --- |
| **1. 后果检查器** | VLA 出候选，世界模型预测碰撞，重排后再执行 | 执行前给每个候选动作打分 | 7.3 的 Tiny VLA + `check-actions-before-moving.ipynb` 的 outcome model |
| **2. 数据引擎** | 世界模型合成轨迹，只拿合成数据训策略 | 在梦里生成训练数据 | 第 3 章的 RSSM + 7.6 的策略训练流水线 |
| **3. 点轨迹零样本** | 预测未来点轨迹，跟踪控制器执行，不训策略 | 预测本身当策略用 | 第 5 章的视频模型 + 7.1 的 PD/IK 控制栈 |

三种用法共用同一句反事实检查：固定当前观察，只换候选动作，预测必须分开。分不开，你手里的就不是世界模型，只是视频生成或行为克隆——先回 [7.3](/chapters/07-robot-vla/03-vision-language-action)。

## 通用交付要求

不管选哪种用法，以下六条一条不能少：

1. **一个身体**：桌面格子（CPU 可跑）、PyBullet/MuJoCo 机械臂（单卡 24 GB）、或 SO-101 真机，三选一。真机不是必须的；把仿真做诚实比把真机做潦草值钱得多。
2. **一个任务**：成功判据、初始条件分布、最大步数，全部写下来再动手。
3. **一个基线**：没有世界模型的同预算对照组。没有基线的「它能工作」不构成证据。
4. **一组统计**：$n \geq 20$ 次试验、置信区间、失败分类。$20$ 次试验的半宽约 $\pm 0.20$，别把噪声当结论。
5. **一段失败分析**：世界模型错在哪——是预测错了、候选集里根本没有好动作、还是重排目标设偏了。
6. **一个公开 artifact**：数据集或模型权重发布到 Hugging Face Hub（或 ModelScope），附数据卡/模型卡，让结果可被发现、可被复现。实在不能公开，写明原因。

## 交付清单：检查器路线的证据

选用法 1 时，烟雾版 Notebook 只证明接口连通。提交必须交出下面这份清单，缺一项就按该项零分——这是把烟雾扩成可评价实验的最低证据，不是完整 walkthrough。

- 按**场景 seed** 切分的数据卡：样本数、字段 shape、train/val/test 边界、seed
- state-only BC 的 loss 曲线，以及第一步动作 MSE
- Tiny VLA 的 chunk loss 曲线，参数量，以及同一张图换指令后的动作差
- 闭环指标：成功率、平均碰撞次数、初始距离、最终距离；测试集不得与训练集同 seed
- outcome model 的总 loss、状态 MSE、碰撞 BCE；以及训练数据里的碰撞比例
- 至少 64 个「直达必撞」场景上的对照：`direct_collision_rate`、`reranked_collision_rate`、`reranked_mean_progress`
- 至少一组 Checker 失败的逐候选拆解：真实碰撞、预测概率、被选中的动作
- 资源清单：设备、步数、耗时、checkpoint SHA256

烟雾版的参照数字（同一套种子，不是你的毕业结论）：state-only BC $0.498\rightarrow 0.377$；Tiny VLA chunk $0.525\rightarrow 0.277$；换指令动作差 $0.120$；闭环 12 步成功率 $0.188$、平均碰撞 $3.281$；后果模型 $0.977\rightarrow 0.538$；64 个必撞场景里直达碰撞 $1.000$、重排后 $0.328$、平均进展 $-0.036$。动作 MSE 下降不等于闭环成功；碰撞下降配上负进展，说明检查器可能只是学会了原地不动。

负结果可以拿满分：只要对照公平，并写清 Checker 为什么没有帮上忙。

## 用法 1：后果检查器

**做法**：策略（7.6 的任一档，或 7.3 的 Tiny VLA）采样 $K$ 个候选 chunk；世界模型对每个候选做想象 rollout，预测碰撞概率与终点进展；按 $\text{进展} - \lambda \cdot \text{风险}$ 重排，执行第一名。

**在烟雾版基础上的毕业要求**：

- 检查器从一步 outcome model 换成**多步想象 rollout**（[7.3](/chapters/07-robot-vla/03-vision-language-action) 的转移方程接 $H$ 步）；
- 在 OOD 布局上评测（障碍位置、目标位置超出训练分布）；
- 扫 $\lambda$：画出「碰撞率—进展」曲线，找到既减碰撞又不学会原地不动的区间。

**证据**：与无检查器的同策略基线相比，碰撞率显著下降且任务进展不塌。两个数必须同时报——只报碰撞下降，交上来的可能是一个学会不动的模型。

## 用法 2：数据引擎

**做法**：先用真实数据训一个世界模型（RSSM 或动作条件视频模型）；从真实初始状态出发，在模型里做短视野的想象 rollout（branched rollout，每段从真实状态重新出发），生成合成轨迹；**只用合成数据**训策略；回真实环境评测。

**毕业要求**：

- 扫合成数据比例：$0\%$、$25\%$、$50\%$、$75\%$、$100\%$ 合成，各训一档策略，画出「合成比例—真实成功率」曲线；
- 扫想象视野 $H$：找到模型误差开始复合的拐点——$H$ 超过它，合成数据从帮助变成毒害；
- 对照组：同样数量的真实数据训出的策略。

**证据**：你的曲线应该复现 MBPO 的核心发现——短视野的合成数据有帮助，长视野的合成数据被模型误差污染。拐点的位置就是你的世界模型的诚实度。

## 用法 3：点轨迹零样本

**做法**：不训策略。给世界模型一张当前图像和一个目标，预测未来若干关键点的轨迹（抓手或物体上的点）；用一个跟踪控制器（IK + PD，见 [7.1](/chapters/07-robot-vla/01-robot-interfaces) 的经典控制栈）让真实身体跟着预测的点走。预测即控制。

**毕业要求**：

- 点轨迹的预测质量单独评测：预测点与真实点的距离随视野的衰减曲线；
- 闭环任务成功率与「训出来的策略」（7.6 的任一档）同任务对比；
- 明确回答：什么样的任务零样本够用（慢、准直、接触少），什么样的任务必须训策略（快、接触密集、需要力）。

**证据**：跟踪误差、任务成功率、与训练策略的对照表。这条路线如果走通，你就亲手验证了 7.3 里最激进的那句话：世界模型可以直接当策略用。

## 评分表

| 维度 | 合格 | 优秀 |
| --- | --- | --- |
| 问题定义 | 任务、判据、初始分布写清楚 | 还写了「为什么这个问题值得用世界模型」 |
| 基线 | 有无世界模型的同预算对照 | 基线本身被认真调过，不是稻草人 |
| 证据 | $n \geq 20$、置信区间、失败分类 | 关键结论有反事实检查或消融支撑 |
| 失败分析 | 说出世界模型错在哪 | 给出修复方向的定量预测，并验证了一条 |
| 可复现性 | 权重、种子、配置存档 | 别人按文档能复现主结果 |

## 已知简化与坑

- **三种用法的难度差很多**。用法 1 有 Tiny VLA 与后果检查的烟雾版打底，最稳；用法 2 要训两个模型再加对照组，算力最贵；用法 3 概念最新，工程坑最多（点跟踪本身是个感知问题）。选题时把预算算进去。
- **想象 rollout 的误差是复利**。$H$ 每加一，预测误差乘一次；所有用法里，视野都是第一个要扫的超参。
- **真机不是必选项**，但「以跨模拟器差异替代真机」这句话必须写进报告，不能默默省略。
- **别把三个用法都做**。做透一个，胜过三个都冒烟。
- **桌面烟雾版是 $32\times 32$ 的合成图**，两条指令、圆盘障碍。这不是真实机器人；专家示范几乎无碰撞，用示范训练碰撞头会得到常数分类器。

## 本节小结

- 世界模型接上身体有三种接法：**检查它（后果检查器）、喂养它（数据引擎）、成为它（零样本策略）**。
- 三种接法共用同一句判据：换动作，预测必须分开。
- 毕业设计的标准不是「跑起来了」，而是基线、统计、失败分析三件套齐全。
- 从 1.1 的「为什么需要世界模型」到这一节，全书只讲了一句话：**在行动之前，先在内部预见行动的后果——现在你的身体也知道了。**

下一章把空间从机器人本体扩展到整个场景与道路，见 [8.1 相机几何与投影](/chapters/08-spatial-worlds/01-camera-geometry)。

[上一篇 7.9　动手：现实迁移的简洁实现](/chapters/07-robot-vla/09-sim2real) · [下一章 → 8.1　相机几何与投影](/chapters/08-spatial-worlds/01-camera-geometry)

## 参考文献

1. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. _NeurIPS 2019_. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) —— 数据引擎用法的理论基础：短视野想象、模型只在可信区间里用。
2. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) —— DreamerV3：在想象里训策略的完整配方。
3. Wen, C., et al. (2024). Any-point Trajectory Modeling for Policy Learning. _RSS 2024_. [arXiv:2401.00025](https://arxiv.org/abs/2401.00025) —— 点轨迹作为策略表示，用法 3 的直接参照。
4. Yang, M., et al. (2023). Learning Interactive Real-World Simulators (UniSim). [arXiv:2310.06114](https://arxiv.org/abs/2310.06114) —— 世界模型作为数据引擎的规模化样板。
5. Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. _RSS 2023_. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) —— 用法 1 里候选动作的来源之一。
