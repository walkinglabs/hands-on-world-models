# B1–C2 · 互动视频与 JEPA 实验

路线 B 与 C 使用同一份 PixelWorld 数据，方便直接比较预测目标。

## 路线 B：预测可见未来

`B1-compress-and-predict-video.ipynb`：

```text
动作—帧对齐 → AE 预热 → VQ-VAE / STE → token Transformer
```

`B2-make-video-controllable.ipynb`：

```text
反事实动作 → 自回归多步 → tiny denoiser 对照
```

CPU smoke 只证明数据流、梯度和动作敏感性能够运行。真正的 PA 要解码画面并检查物体位移方向。

## 路线 C：预测有用特征

`C1-learn-video-features.ipynb`：

```text
video patch → online/target encoder → mask → EMA → collapse 检查
```

`C2-test-and-control-features.ipynb`：

```text
linear probe → action conditioning → 反事实 feature → 一步动作选择
```

被动视频结果与动作条件结果必须分开报告。没有动作标签的数据不能证明 controllability。

## 运行

```bash
python -m pip install -r requirements-neural.txt
python -m unittest tests.test_routes_bc -v
```

完成一条路线即可进入对应 PA，不要求同时完成 B 与 C。
