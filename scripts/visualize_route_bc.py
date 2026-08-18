#!/usr/bin/env python3
"""
路线 B/C 真实可视化：运行 VQ-VAE + Transformer 和 JEPA 的实际代码。
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from hwm.video import TinyVQVAE
from hwm.jepa import TinyVideoJEPA, feature_spread, jepa_batch_from_episodes, patchify_video
from hwm.data import make_pixelworld_dataset


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def visualize_vq_transformer(output_path):
    """路线 B: VQ-VAE 压缩 + Transformer 预测"""
    episodes = make_pixelworld_dataset(num_episodes=4, length=8, seed=42)
    
    # 创建 VQ-VAE
    vqvae = TinyVQVAE(codebook_size=16, embedding_size=8)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
    
    # 收集视频帧
    import numpy as np
    frames_list = []
    for ep in episodes:
        frames_list.append(ep.observations)
    frames_np = np.concatenate(frames_list)  # [N, H, W, C]
    videos_t = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    
    # 训练几步
    losses = []
    for step in range(20):
        frames = videos_t[:8]  # 取前 8 帧作为 batch
        result = vqvae(frames)
        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # 可视化
    cell = 140
    padding = 20
    title_h = 40
    
    img_w = cell * 4 + padding * 5
    img_h = cell + title_h + padding * 2 + 60
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(13)
    title_font = try_load_font(20)
    
    # 1. 原始帧
    frame = (videos_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    frame_pil = Image.fromarray(frame).resize((cell, cell), Image.NEAREST)
    canvas.paste(frame_pil, (padding, title_h + padding))
    draw.text((padding + 5, title_h + padding + cell + 5), "原始帧", fill='black', font=font)
    
    # 2. 重建帧
    with torch.no_grad():
        result = vqvae(videos_t[0:1])
    recon = (result["reconstruction"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    recon_pil = Image.fromarray(recon).resize((cell, cell), Image.NEAREST)
    canvas.paste(recon_pil, (cell + padding * 2, title_h + padding))
    draw.text((cell + padding * 2 + 5, title_h + padding + cell + 5), "VQ-VAE 重建", fill='black', font=font)
    
    # 3. 码本信息
    codebook_img = Image.new('RGB', (cell, cell), 'white')
    cd = ImageDraw.Draw(codebook_img)
    cd.text((10, 10), f"码本大小: {vqvae.codebook_size}", fill='black', font=font)
    cd.text((10, 35), f"训练损失: {losses[-1]:.4f}", fill='blue', font=font)
    cd.text((10, 60), f"重建 MSE: {result['reconstruction_loss'].item():.4f}", fill='green', font=font)
    cd.text((10, 85), f"量化损失: {result['quantization_loss'].item():.4f}", fill='orange', font=font)
    canvas.paste(codebook_img, (2 * (cell + padding) + padding, title_h + padding))
    draw.text((2 * (cell + padding) + padding + 5, title_h + padding + cell + 5), "VQ 码本", fill='black', font=font)
    
    # 4. Transformer 预测示意
    trans_img = Image.new('RGB', (cell, cell), 'white')
    td = ImageDraw.Draw(trans_img)
    td.text((10, 10), "自回归预测:", fill='black', font=font)
    td.text((10, 35), "token[t-1] → token[t]", fill='blue', font=try_load_font(11))
    td.text((10, 60), "+ 动作条件 (FiLM)", fill='green', font=try_load_font(11))
    td.text((10, 85), "因果掩码防止偷看", fill='red', font=try_load_font(11))
    canvas.paste(trans_img, (3 * (cell + padding) + padding, title_h + padding))
    draw.text((3 * (cell + padding) + padding + 5, title_h + padding + cell + 5), "Transformer", fill='black', font=font)
    
    draw.text((padding, 5), "路线 B: VQ-VAE 压缩 + Transformer 自回归预测", fill='black', font=title_font)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_jepa(output_path):
    """路线 C: Video-JEPA 特征预测"""
    episodes = make_pixelworld_dataset(num_episodes=8, length=10, seed=42)
    
    # 创建 JEPA 模型
    jepa = TinyVideoJEPA(feature_size=32, action_size=5, patch_size=4, num_patches=16)
    optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)
    
    # 准备数据
    histories, actions_list, positions = jepa_batch_from_episodes(episodes, history_length=3)
    
    # 训练几步
    losses = []
    for step in range(20):
        loss, pred, target, features = jepa.loss(histories, actions_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        jepa.update_target(momentum=0.99)
        losses.append(loss.item())
    
    # 提取特征
    with torch.no_grad():
        last_frame = histories[:, -1:]  # [B, 1, C, H, W]
        patches = patchify_video(last_frame, jepa.patch_size)
        features = jepa.online_encoder(patches)
        spread = feature_spread(features)
    
    # 可视化
    cell = 140
    padding = 20
    title_h = 40
    
    img_w = cell * 4 + padding * 5
    img_h = cell + title_h + padding * 2 + 60
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(13)
    title_font = try_load_font(20)
    
    # 1. 历史帧
    frame = (histories[0, -1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    frame_pil = Image.fromarray(frame).resize((cell, cell), Image.NEAREST)
    canvas.paste(frame_pil, (padding, title_h + padding))
    draw.text((padding + 5, title_h + padding + cell + 5), "历史帧", fill='black', font=font)
    
    # 2. 特征可视化
    feat = features[0].numpy()
    # 简化：只显示特征形状和统计信息
    feat_img = Image.new('RGB', (cell, cell), 'white')
    fd = ImageDraw.Draw(feat_img)
    fd.text((10, 10), f"特征形状: {feat.shape}", fill='black', font=font)
    fd.text((10, 35), f"均值: {feat.mean():.4f}", fill='blue', font=font)
    fd.text((10, 60), f"标准差: {feat.std():.4f}", fill='green', font=font)
    fd.text((10, 85), f"最小: {feat.min():.4f}", fill='gray', font=try_load_font(11))
    fd.text((10, 105), f"最大: {feat.max():.4f}", fill='gray', font=try_load_font(11))
    canvas.paste(feat_img, (cell + padding * 2, title_h + padding))
    draw.text((cell + padding * 2 + 5, title_h + padding + cell + 5), "Online 特征", fill='black', font=font)
    
    # 3. 特征分布
    spread_img = Image.new('RGB', (cell, cell), 'white')
    sd = ImageDraw.Draw(spread_img)
    sd.text((10, 10), f"特征维度: {features.shape[-1]}", fill='black', font=font)
    sd.text((10, 35), f"特征 spread: {spread:.4f}", fill='blue', font=font)
    sd.text((10, 60), f"训练损失: {losses[-1]:.4f}", fill='green', font=font)
    sd.text((10, 85), "EMA 跟随 Online", fill='gray', font=try_load_font(11))
    canvas.paste(spread_img, (2 * (cell + padding) + padding, title_h + padding))
    draw.text((2 * (cell + padding) + padding + 5, title_h + padding + cell + 5), "特征统计", fill='black', font=font)
    
    # 4. JEPA 架构
    arch_img = Image.new('RGB', (cell, cell), 'white')
    ad = ImageDraw.Draw(arch_img)
    ad.text((10, 10), "架构:", fill='black', font=font)
    ad.text((10, 35), "Online Encoder → 特征", fill='blue', font=try_load_font(11))
    ad.text((10, 55), "Predictor → 预测特征", fill='green', font=try_load_font(11))
    ad.text((10, 75), "Target Encoder (EMA)", fill='red', font=try_load_font(11))
    ad.text((10, 95), "不重建像素!", fill='purple', font=try_load_font(11))
    canvas.paste(arch_img, (3 * (cell + padding) + padding, title_h + padding))
    draw.text((3 * (cell + padding) + padding + 5, title_h + padding + cell + 5), "JEPA 架构", fill='black', font=font)
    
    draw.text((padding, 5), "路线 C: Video-JEPA 特征预测（不重建像素）", fill='black', font=title_font)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'docs' / 'public' / 'carracing'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_vq_transformer(output_dir / 'bc-vq-transformer.png')
    visualize_jepa(output_dir / 'bc-jepa.png')
    
    print("\nRoute B/C visualizations generated!")
