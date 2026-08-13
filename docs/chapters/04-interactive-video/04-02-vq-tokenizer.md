# 4.2　VQ-VAE：把一帧画面变成 token

直接让 Transformer 逐像素生成 `64×64×3` 个值，序列过长，也不容易保留局部结构。VQ-VAE 先把画面压成较短的离散网格。

## 编码与量化

Encoder 把图片变成低分辨率 feature map。每个位置的向量在码本中寻找最近项：

```text
frame → encoder features → nearest codebook entries → token grid
```

例如 `64×64` 图片可以变成 `8×8` token 网格。视频 Transformer 每帧只需预测 64 个编号。

## STE 与码本更新

最近邻选择没有普通梯度。直通估计器在前向使用离散码本向量，反向把 Decoder 的梯度近似传回 Encoder。

训练还包含重建损失、codebook 损失和 commitment 损失。后两项让码本与 Encoder 输出彼此靠近。

## 码本坍缩

若只有少数 token 被频繁使用，码本容量浪费，细小物体容易消失。日志应记录 token 使用率、perplexity 和长期未使用项。

增加码本大小不一定解决问题。Decoder 太强、学习率失衡或数据过于单一都可能造成坍缩。

## tokenizer 的评价

PSNR 或重建 loss 只能说明像素接近程度。互动视频还要查看 HUD、边缘、小物体与动作相关区域是否被保留。

一个 tokenizer 可以重建漂亮背景，却把决定碰撞的小球压没。B1 会把重建图与 token 使用统计并列展示。

PixelWorld 的红色方块只占 9 个像素，大部分画面都是黑色。若只使用普通像素 MSE，一张全黑图片也可能得到不高的平均误差，方块却已经消失。

课程实现会给前景像素更高权重，并单独测量红色物体中心。重建 loss、码本使用率和小物体位置必须一起报告。

## 小结

- [ ] VQ-VAE 把一帧图片变成较短的离散 token 网格。
- [ ] STE 让量化路径可以训练，codebook 与 commitment loss 维持码本。
- [ ] tokenizer 需要检查码本使用和任务相关细节，不只看平均重建误差。
