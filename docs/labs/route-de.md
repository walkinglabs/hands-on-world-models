# D1–E2 · 机器人与空间实验

## 路线 D：Tiny VLA 与后果检查

- `D1-build-a-tiny-vla.ipynb`：state BC → image/language/state → action chunk。
- `D2-check-actions-before-moving.ipynb`：候选动作 → next-state/collision model → reranking。

直接 VLA 交出动作；D2 的 outcome model 才负责预测动作后果。

## 路线 E：空间共同基础后二选一

- `E1-from-camera-to-space.ipynb`：深度 → 三维点 → 外参 → Occupancy；所有空间学生必做。
- `E2a-build-a-small-4d-world.ipynb`：坐标神经场与 3D/4D 接口。
- `E2b-predict-driving-space.ipynb`：动作条件 future Occupancy。

E2a 与 E2b 只选一份。静态神经场不称为 4D；离线 Occupancy 不称为闭环驾驶。

## 运行

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_routes_de -v
```
