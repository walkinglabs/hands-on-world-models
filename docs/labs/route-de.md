# D1–E2 · 机器人与空间实验

## 路线 D：Tiny VLA 与后果检查

- `D1-build-a-tiny-vla.ipynb`：state BC → image/language/state → action chunk。
- `D2-check-actions-before-moving.ipynb`：候选动作 → next-state/collision model → reranking。

直接 VLA 交出动作；D2 的 outcome model 才负责预测动作后果。D1 会把训练好的 policy 放回 Tabletop 连续执行，报告成功、碰撞和最终距离，不用动作 MSE 代替闭环结果。

D2 也不靠一个手挑样例宣布 checker 有用。它批量构造“直达动作会碰撞”的场景，同时报告重排前后碰撞率和目标进展。若碰撞减少、但每步反而离目标更远，这说明后果模型学到了安全，Planner 还没有学会绕行。

## 路线 E：空间共同基础后二选一

- `E1-from-camera-to-space.ipynb`：深度 → 三维点 → 外参 → Occupancy；所有空间学生必做。
- `E2a-build-a-small-4d-world.ipynb`：静态坐标神经场 → 时间与动作条件动态场 → 反事实查询。
- `E2b-predict-driving-space.ipynb`：动作条件 future Occupancy。

E2a 与 E2b 只选一份。E2a 的 moving-sphere 是 4D 接口 smoke，不是多视角场景重建；离线 Occupancy 也不称为闭环驾驶。

## 运行

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_routes_de -v
```
