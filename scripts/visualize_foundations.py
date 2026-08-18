#!/usr/bin/env python3
"""
Foundations 可视化脚本：运行 F1/F2/F3 的实际代码，生成真实可视化。
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hwm.foundations import (
    conv2d_valid, patchify, depth_to_points, transform_points,
    points_to_occupancy, cem_plan_1d, symlog, make_camera_transform,
)


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_sample_image(size=16, seed=42):
    """生成一张简单的 16x16 测试图片：黑色背景 + 红色方块 + 蓝色方块"""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # 红色方块
    r1, c1 = rng.integers(2, 6), rng.integers(2, 6)
    img[r1:r1+3, c1:c1+3] = [220, 40, 40]
    # 蓝色方块
    r2, c2 = rng.integers(9, 13), rng.integers(9, 13)
    img[r2:r2+3, c2:c2+3] = [40, 40, 220]
    return img


def visualize_convolution(output_path):
    """F1 第一步：卷积响应图"""
    img = make_sample_image()
    
    # 三种不同的 3x3 卷积核
    kernels = {
        '边缘检测 (水平)': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
        '边缘检测 (垂直)': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
        '角点检测': np.array([[1, -1, -1], [1, 1, -1], [-1, 1, 1]], dtype=np.float32),
    }
    
    cell = 120
    padding = 30
    title_h = 40
    cols = len(kernels) + 1  # 原图 + 3 个响应图
    img_w = cols * cell + (cols + 1) * padding
    img_h = cell + title_h + padding * 2 + 60
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(16)
    
    # 画原图（放大显示）
    orig_pil = Image.fromarray(img).resize((cell, cell), Image.NEAREST)
    canvas.paste(orig_pil, (padding, title_h + padding))
    draw.text((padding + 10, title_h + padding + cell + 5), "原图 (16x16)", fill='black', font=font)
    
    # 画每个卷积核的响应
    from hwm.foundations import rgb_to_gray
    gray = rgb_to_gray(img)
    
    for i, (name, kernel) in enumerate(kernels.items()):
        response = conv2d_valid(gray, kernel)
        # 归一化到 0-255
        rmin, rmax = response.min(), response.max()
        if rmax - rmin > 1e-6:
            normed = ((response - rmin) / (rmax - rmin) * 255).astype(np.uint8)
        else:
            normed = np.zeros_like(response, dtype=np.uint8)
        
        x_offset = (i + 1) * (cell + padding) + padding
        resp_pil = Image.fromarray(normed).resize((cell, cell), Image.NEAREST)
        canvas.paste(resp_pil, (x_offset, title_h + padding))
        draw.text((x_offset + 5, title_h + padding + cell + 5), name, fill='black', font=font)
    
    # 标题
    title_font = try_load_font(22)
    draw.text((padding, 5), "F1: 3x3 卷积核的响应图（无需训练即可检测边缘）", fill='black', font=title_font)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_patchify(output_path):
    """F1 第二步：ViT patch 切分"""
    img = make_sample_image()
    patch_size = 4
    
    cell = 120
    padding = 30
    title_h = 40
    img_w = cell * 3 + padding * 4
    img_h = cell + title_h + padding * 2 + 60
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    title_font = try_load_font(22)
    
    # 原图
    orig_pil = Image.fromarray(img).resize((cell, cell), Image.NEAREST)
    canvas.paste(orig_pil, (padding, title_h + padding))
    draw.text((padding + 5, title_h + padding + cell + 5), "原图", fill='black', font=font)
    
    # 画网格线
    grid_pil = Image.fromarray(img).resize((cell, cell), Image.NEAREST)
    grid_draw = ImageDraw.Draw(grid_pil)
    scale = cell / 16
    for i in range(0, 17, patch_size):
        x = int(i * scale)
        grid_draw.line([(x, 0), (x, cell)], fill='yellow', width=2)
        grid_draw.line([(0, x), (cell, x)], fill='yellow', width=2)
    canvas.paste(grid_pil, (cell + padding * 2, title_h + padding))
    draw.text((cell + padding * 2 + 5, title_h + padding + cell + 5), f"切成 {patch_size}x{patch_size} patch", fill='black', font=font)
    
    # Token 表
    tokens = patchify(img, patch_size)
    table_x = 2 * (cell + padding) + padding
    draw.text((table_x + 5, title_h + padding), f"得到 {len(tokens)} 个 token", fill='black', font=font)
    draw.text((table_x + 5, title_h + padding + 25), f"每个 token: {tokens.shape[1]} 维", fill='gray', font=font)
    draw.text((table_x + 5, title_h + padding + 50), f"压缩比: {16*16*3}/{len(tokens)*tokens.shape[1]:.1f}x", fill='blue', font=font)
    
    title_font_draw = try_load_font(22)
    draw.text((padding, 5), "F1: ViT patch 切分——把图片变成 token 序列", fill='black', font=title_font_draw)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_depth_to_occupancy(output_path):
    """F2: 深度反投影 → Occupancy"""
    # 创建一个简单的深度图
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[0, :] = 1.0  # 远处
    depth[1, :] = 2.0
    depth[2, :] = 3.0  # 中间有障碍物
    depth[2, 2] = 1.5  # 障碍物
    depth[3, :] = 4.0
    depth[4, :] = 5.0  # 近处
    
    # 反投影
    points_cam = depth_to_points(depth, fx=6, fy=6, cx=2.5, cy=2.5)
    
    # 变换到世界坐标
    cam2world = make_camera_transform(tx=0, ty=1.5, tz=0, yaw=0.3)
    points_world = transform_points(points_cam, cam2world)
    
    # 生成 Occupancy
    occ = points_to_occupancy(points_world, x_range=(-2, 4), z_range=(0, 6), resolution=0.5)
    
    # 可视化
    cell = 140
    padding = 30
    title_h = 40
    cols = 3
    img_w = cols * cell + (cols + 1) * padding
    img_h = cell + title_h + padding * 2 + 60
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    title_font = try_load_font(22)
    
    # 深度图
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    depth_color[:, :, 2] = depth_norm  # 蓝色通道
    depth_pil = Image.fromarray(depth_color).resize((cell, cell), Image.NEAREST)
    canvas.paste(depth_pil, (padding, title_h + padding))
    draw.text((padding + 5, title_h + padding + cell + 5), "深度图 (5x5)", fill='black', font=font)
    
    # 3D 点云（俯视图）
    point_img = Image.new('RGB', (cell, cell), 'white')
    pd = ImageDraw.Draw(point_img)
    for pt in points_world:
        px = int((pt[0] + 2) / 6 * cell)
        py = int((6 - pt[2]) / 6 * cell)
        if 0 <= px < cell and 0 <= py < cell:
            pd.ellipse([px-2, py-2, px+2, py+2], fill='blue')
    canvas.paste(point_img, (cell + padding * 2, title_h + padding))
    draw.text((cell + padding * 2 + 5, title_h + padding + cell + 5), "3D 点云（俯视）", fill='black', font=font)
    
    # Occupancy 网格
    occ_img = Image.new('RGB', (cell, cell), 'white')
    od = ImageDraw.Draw(occ_img)
    gh, gw = occ.shape
    for r in range(gh):
        for c in range(gw):
            if occ[r, c]:
                x1 = int(c / gw * cell)
                y1 = int(r / gh * cell)
                x2 = int((c + 1) / gw * cell)
                y2 = int((r + 1) / gh * cell)
                od.rectangle([x1, y1, x2, y2], fill='red')
    canvas.paste(occ_img, (2 * (cell + padding) + padding, title_h + padding))
    draw.text((2 * (cell + padding) + padding + 5, title_h + padding + cell + 5), f"Occupancy ({occ.sum()} 格)", fill='black', font=font)
    
    draw.text((padding, 5), "F2: 深度图 → 3D 点云 → Occupancy 网格", fill='black', font=title_font)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_cem_search(output_path):
    """F2: CEM 搜索过程"""
    start = 0.0
    target = 3.0
    
    # 运行 CEM
    best_actions, history = cem_plan_1d(start, target, horizon=5, population=200, elite=20, rounds=8)
    
    cell_w = 100
    cell_h = 80
    padding = 30
    title_h = 40
    rounds = len(history)
    
    img_w = rounds * cell_w + (rounds + 1) * padding
    img_h = cell_h + title_h + padding * 2 + 60
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    title_font = try_load_font(22)
    
    # 画每一轮的搜索分布
    rng = np.random.default_rng(0)
    mean = np.zeros(5, dtype=np.float32)
    std = np.ones(5, dtype=np.float32)
    
    for r in range(rounds):
        x_offset = r * (cell_w + padding) + padding
        
        # 采样
        actions = rng.normal(mean, std, size=(100, 5))
        actions = np.clip(actions, -1.0, 1.0)
        finals = start + actions.sum(axis=1)
        
        # 画直方图
        hist_img = Image.new('RGB', (cell_w, cell_h), 'white')
        hd = ImageDraw.Draw(hist_img)
        
        # 简化：画目标线和最终位置分布
        target_x = int((target + 1) / 8 * cell_w)
        hd.line([(target_x, 0), (target_x, cell_h)], fill='green', width=2)
        
        for f in finals:
            fx = int((f + 1) / 8 * cell_w)
            if 0 <= fx < cell_w:
                hd.line([(fx, cell_h // 2 - 10), (fx, cell_h // 2 + 10)], fill='blue', width=1)
        
        mean_final = start + actions.mean(axis=0).sum()
        mean_x = int((mean_final + 1) / 8 * cell_w)
        hd.line([(mean_x, 0), (mean_x, cell_h)], fill='red', width=3)
        
        canvas.paste(hist_img, (x_offset, title_h + padding))
        draw.text((x_offset + 5, title_h + padding + cell_h + 5), f"轮次 {r+1}", fill='black', font=font)
        draw.text((x_offset + 5, title_h + padding + cell_h + 25), f"score={history[r]:.2f}", fill='gray', font=try_load_font(11))
        
        # 更新分布
        scores = -(finals - target) ** 2
        elite_idx = np.argsort(scores)[-20:]
        mean = actions[elite_idx].mean(axis=0)
        std = actions[elite_idx].std(axis=0) + 1e-4
    
    draw.text((padding, 5), "F2: CEM 搜索——从随机采样到精英集中", fill='black', font=title_font)
    
    # 图例
    legend_x = img_w - 200
    draw.text((legend_x, title_h + padding - 20), "绿线=目标  红线=均值  蓝线=样本", fill='black', font=try_load_font(12))
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'docs' / 'public' / 'carracing'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_convolution(output_dir / 'f1-convolution.png')
    visualize_patchify(output_dir / 'f1-patchify.png')
    visualize_depth_to_occupancy(output_dir / 'f2-depth-occupancy.png')
    visualize_cem_search(output_dir / 'f2-cem-search.png')
    
    print("\nAll foundations visualizations generated!")
