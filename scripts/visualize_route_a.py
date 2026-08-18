#!/usr/bin/env python3
"""
Route-A 可视化脚本：运行 A1/A2 的实际代码，生成真实可视化。
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from hwm.neural import PixelEncoder, RSSM, RSSMState
from hwm.data import MovingSquareWorld, make_pixelworld_dataset
from hwm.control import PositionDynamics, fit_position_dynamics, beam_plan


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def visualize_rssm_dataflow(output_path):
    """A1: RSSM 数据流可视化——展示各组件的 shape"""
    episodes = make_pixelworld_dataset(num_episodes=4, length=8, seed=42)

    encoder = PixelEncoder(embed_size=64)
    rssm = RSSM(action_size=5, deter_size=64, stoch_size=16, embed_size=64)

    observations = torch.tensor(np.array([ep.observations for ep in episodes]), dtype=torch.float32)
    actions = torch.tensor(np.array([ep.actions for ep in episodes]), dtype=torch.long)

    # observations: [B, T+1, H, W, C], actions: [B, T]
    # 第 t 个 action 对应 observations[t] -> observations[t+1]
    # RSSM observe 需要 embeds[B,T,E] 与 actions[B,T] 对齐
    obs_next = observations[:, 1:]  # [B, T, H, W, C]
    B, T = obs_next.shape[0], obs_next.shape[1]
    obs_flat = obs_next.reshape(B * T, *obs_next.shape[2:])
    emb_flat = encoder(obs_flat)
    embeddings = emb_flat.reshape(B, T, -1)

    prior, posterior = rssm.observe(embeddings, actions)

    # 可视化
    cell_w = 160
    cell_h = 100
    padding = 20
    title_h = 40

    components = [
        ("观测\n[B,T,H,W,C]", f"{list(observations.shape)}", "blue"),
        ("Encoder\nCNN", f"-> {list(embeddings.shape)}", "green"),
        ("RSSM\nprior/posterior", f"det:{list(prior.deterministic.shape)}\nstoch:{list(prior.stochastic.shape)}", "orange"),
        ("特征\n[deter+stoch]", f"{list(posterior.feature.shape)}", "purple"),
    ]

    cols = len(components)
    img_w = cols * cell_w + (cols + 1) * padding
    img_h = cell_h + title_h + padding * 2 + 80

    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(13)
    title_font = try_load_font(20)

    for i, (name, shape, color) in enumerate(components):
        x = i * (cell_w + padding) + padding
        y = title_h + padding

        draw.rounded_rectangle([x, y, x + cell_w, y + cell_h], radius=8, outline=color, width=3)

        lines = name.split('\n')
        for j, line in enumerate(lines):
            draw.text((x + 10, y + 10 + j * 18), line, fill=color, font=font)

        shape_lines = shape.split('\n')
        for j, line in enumerate(shape_lines):
            draw.text((x + 10, y + 55 + j * 16), line, fill='gray', font=try_load_font(11))

        if i < len(components) - 1:
            arrow_x = x + cell_w + 5
            arrow_y = y + cell_h // 2
            draw.line([(arrow_x, arrow_y), (arrow_x + 10, arrow_y)], fill='black', width=2)
            draw.polygon([(arrow_x + 10, arrow_y - 5), (arrow_x + 15, arrow_y), (arrow_x + 10, arrow_y + 5)], fill='black')

    draw.text((padding, 5), "A1: RSSM 数据流——从像素到隐状态", fill='black', font=title_font)
    draw.text((padding, img_h - 40), "B=4 episodes, T=8 steps, deter=64, stoch=16, feature=80", fill='gray', font=try_load_font(12))

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_position_planning(output_path):
    """A2: 位置模型 + beam search 规划"""
    episodes = make_pixelworld_dataset(num_episodes=8, length=10, seed=42)

    from hwm.foundations import center_of_red
    positions = []
    next_positions = []
    actions_list = []
    for ep in episodes:
        for t in range(len(ep.observations) - 1):
            pos = center_of_red(ep.observations[t])
            npos = center_of_red(ep.observations[t + 1])
            positions.append(pos)
            next_positions.append(npos)
            actions_list.append(ep.actions[t])

    positions = torch.tensor(np.array(positions), dtype=torch.float32)
    next_positions = torch.tensor(np.array(next_positions), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.long)

    model = PositionDynamics(action_size=5, hidden_size=64)
    losses = fit_position_dynamics(model, positions, actions_tensor, next_positions, updates=100)

    start_pos = positions[0]
    target_pos = torch.tensor([10.0, 10.0])
    plan = beam_plan(model, start_pos, goal=target_pos, horizon=4, beam_size=10)

    cell = 140
    padding = 20
    title_h = 40

    img_w = cell * 3 + padding * 4
    img_h = cell + title_h + padding * 2 + 60

    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    title_font = try_load_font(20)

    # 训练损失曲线
    loss_img = Image.new('RGB', (cell, cell), 'white')
    ld = ImageDraw.Draw(loss_img)
    if losses:
        max_loss = max(losses)
        min_loss = min(losses)
        for i in range(len(losses) - 1):
            x1 = int(i / len(losses) * cell)
            y1 = int((1 - (losses[i] - min_loss) / (max_loss - min_loss + 1e-8)) * cell)
            x2 = int((i + 1) / len(losses) * cell)
            y2 = int((1 - (losses[i + 1] - min_loss) / (max_loss - min_loss + 1e-8)) * cell)
            ld.line([(x1, y1), (x2, y2)], fill='blue', width=2)
    canvas.paste(loss_img, (padding, title_h + padding))
    draw.text((padding + 5, title_h + padding + cell + 5), "训练损失", fill='black', font=font)
    draw.text((padding + 5, title_h + padding + cell + 25), f"初始: {losses[0]:.3f} -> 最终: {losses[-1]:.3f}", fill='gray', font=try_load_font(11))

    # 预测 vs 真实
    pred_img = Image.new('RGB', (cell, cell), 'white')
    pd = ImageDraw.Draw(pred_img)
    for i in range(min(20, len(positions))):
        px = int(positions[i, 0].item() / 14 * cell)
        py = int(positions[i, 1].item() / 14 * cell)
        pd.ellipse([px-3, py-3, px+3, py+3], fill='blue')

        with torch.no_grad():
            pred = model(positions[i:i+1], actions_tensor[i:i+1])
        pred_x = int(pred[0, 0].item() / 14 * cell)
        pred_y = int(pred[0, 1].item() / 14 * cell)
        pd.ellipse([pred_x-2, pred_y-2, pred_x+2, pred_y+2], fill='red')

    canvas.paste(pred_img, (cell + padding * 2, title_h + padding))
    draw.text((cell + padding * 2 + 5, title_h + padding + cell + 5), "预测(红) vs 真实(蓝)", fill='black', font=font)

    # 规划路径
    plan_img = Image.new('RGB', (cell, cell), 'white')
    pld = ImageDraw.Draw(plan_img)
    current = start_pos.clone().unsqueeze(0)
    path_points = [current[0].tolist()]
    for action in plan.actions:
        with torch.no_grad():
            next_p = model(current, torch.tensor([action]))
        path_points.append(next_p[0].tolist())
        current = next_p

    for i in range(len(path_points) - 1):
        x1 = int(path_points[i][0] / 14 * cell)
        y1 = int(path_points[i][1] / 14 * cell)
        x2 = int(path_points[i + 1][0] / 14 * cell)
        y2 = int(path_points[i + 1][1] / 14 * cell)
        pld.line([(x1, y1), (x2, y2)], fill='green', width=2)
        pld.ellipse([x1-3, y1-3, x1+3, y1+3], fill='green')

    tx = int(target_pos[0].item() / 14 * cell)
    ty = int(target_pos[1].item() / 14 * cell)
    pld.ellipse([tx-5, ty-5, tx+5, ty+5], outline='red', width=2)

    canvas.paste(plan_img, (2 * (cell + padding) + padding, title_h + padding))
    draw.text((2 * (cell + padding) + padding + 5, title_h + padding + cell + 5), "Beam search 路径", fill='black', font=font)
    draw.text((2 * (cell + padding) + padding + 5, title_h + padding + cell + 25), f"动作: {list(plan.actions)}", fill='gray', font=try_load_font(11))

    draw.text((padding, 5), "A2: 位置模型——训练、预测、规划", fill='black', font=title_font)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'docs' / 'public' / 'carracing'
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_rssm_dataflow(output_dir / 'a1-rssm-dataflow.png')
    visualize_position_planning(output_dir / 'a2-position-planning.png')

    print("\nAll route-a visualizations generated!")
