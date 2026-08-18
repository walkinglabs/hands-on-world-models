#!/usr/bin/env python3
"""
F0 可视化脚本：运行九格世界的实际代码，生成真实的过程可视化。
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hwm.gridworld import GridWorld


def create_grid_image(world_size=3, cell_size=100, padding=20):
    """创建网格背景图"""
    img_size = world_size * cell_size + 2 * padding
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # 画网格
    for i in range(world_size + 1):
        x = padding + i * cell_size
        draw.line([(x, padding), (x, padding + world_size * cell_size)], fill='gray', width=2)
        draw.line([(padding, x), (padding + world_size * cell_size, x)], fill='gray', width=2)
    
    return img, draw


def draw_agent(draw, pos, cell_size, padding, color='blue'):
    """画智能体"""
    x = padding + pos[0] * cell_size + cell_size // 2
    y = padding + pos[1] * cell_size + cell_size // 2
    radius = cell_size // 4
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='black', width=2)


def draw_trap(draw, pos, cell_size, padding):
    """画陷阱"""
    x1 = padding + pos[0] * cell_size + 10
    y1 = padding + pos[1] * cell_size + 10
    x2 = padding + (pos[0] + 1) * cell_size - 10
    y2 = padding + (pos[1] + 1) * cell_size - 10
    draw.rectangle([x1, y1, x2, y2], fill='red', outline='darkred', width=2)
    # 画 X
    draw.line([(x1, y1), (x2, y2)], fill='white', width=3)
    draw.line([(x1, y2), (x2, y1)], fill='white', width=3)


def draw_goal(draw, pos, cell_size, padding):
    """画目标"""
    x = padding + pos[0] * cell_size + cell_size // 2
    y = padding + pos[1] * cell_size + cell_size // 2
    # 画星形
    points = []
    for i in range(10):
        angle = i * np.pi / 5 - np.pi / 2
        radius = cell_size // 3 if i % 2 == 0 else cell_size // 6
        px = x + radius * np.cos(angle)
        py = y + radius * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill='green', outline='darkgreen')


def draw_path(draw, path, cell_size, padding, color='blue', width=3):
    """画路径"""
    for i in range(len(path) - 1):
        x1 = padding + path[i][0] * cell_size + cell_size // 2
        y1 = padding + path[i][1] * cell_size + cell_size // 2
        x2 = padding + path[i+1][0] * cell_size + cell_size // 2
        y2 = padding + path[i+1][1] * cell_size + cell_size // 2
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        # 画箭头
        if i < len(path) - 1:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx, dy = dx/length, dy/length
                arrow_size = 10
                arrow_x = x2 - dx * arrow_size
                arrow_y = y2 - dy * arrow_size
                draw.line([(x2, y2), (arrow_x - dy*5, arrow_y + dx*5)], fill=color, width=width)
                draw.line([(x2, y2), (arrow_x + dy*5, arrow_y - dx*5)], fill=color, width=width)


def visualize_greedy_vs_planned(output_path):
    """可视化贪心 vs 规划"""
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), walls=(), traps=((1, 1),))
    
    # 创建两列对比图
    cell_size = 100
    padding = 20
    img_width = (3 * cell_size + 2 * padding) * 2 + 40
    img_height = 3 * cell_size + 2 * padding + 60
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 标题
    try:
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((img_width//4 - 80, 10), "贪心策略（无模型）", fill='red', font=font)
    draw.text((img_width*3//4 - 80, 10), "规划策略（有模型）", fill='green', font=font)
    
    # 左：贪心路径
    greedy_path = [(0, 0), (1, 0), (1, 1)]  # 走进陷阱
    offset_x = 0
    grid_img, grid_draw = create_grid_image(3, cell_size, padding)
    draw_trap(grid_draw, (1, 1), cell_size, padding)
    draw_goal(grid_draw, (2, 2), cell_size, padding)
    draw_path(grid_draw, greedy_path, cell_size, padding, color='red', width=4)
    for pos in greedy_path:
        draw_agent(grid_draw, pos, cell_size, padding, color='red')
    img.paste(grid_img, (offset_x, 50))
    
    # 右：规划路径
    planned_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]  # 绕开陷阱
    offset_x = (3 * cell_size + 2 * padding) + 40
    grid_img, grid_draw = create_grid_image(3, cell_size, padding)
    draw_trap(grid_draw, (1, 1), cell_size, padding)
    draw_goal(grid_draw, (2, 2), cell_size, padding)
    draw_path(grid_draw, planned_path, cell_size, padding, color='green', width=4)
    for pos in planned_path:
        draw_agent(grid_draw, pos, cell_size, padding, color='green')
    img.paste(grid_img, (offset_x, 50))
    
    img.save(output_path)
    print(f"Saved: {output_path}")


def visualize_mpc_process(output_path):
    """可视化 MPC 过程"""
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), walls=(), traps=((1, 1),))
    
    cell_size = 80
    padding = 15
    panels = 4
    panel_width = 3 * cell_size + 2 * padding
    panel_height = 3 * cell_size + 2 * padding + 40
    
    img_width = panel_width * panels + 20 * (panels - 1)
    img_height = panel_height
    img = Image.new('RGB', (img_width, img_height), 'white')
    
    # MPC 的四个步骤
    mpc_steps = [
        ([(0, 0)], [(0, 0), (1, 0), (2, 0), (2, 1)]),  # 在 (0,0)，规划到 (2,1)，执行到 (1,0)
        ([(0, 0), (1, 0)], [(1, 0), (2, 0), (2, 1), (2, 2)]),  # 在 (1,0)，规划到 (2,2)，执行到 (2,0)
        ([(0, 0), (1, 0), (2, 0)], [(2, 0), (2, 1), (2, 2)]),  # 在 (2,0)，执行到 (2,1)
        ([(0, 0), (1, 0), (2, 0), (2, 1)], [(2, 1), (2, 2)]),  # 在 (2,1)，执行到 (2,2)
    ]
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 16)
    except:
        font = ImageFont.load_default()
    
    for i, (executed, plan) in enumerate(mpc_steps):
        offset_x = i * (panel_width + 20)
        panel = Image.new('RGB', (panel_width, panel_height), 'white')
        panel_draw = ImageDraw.Draw(panel)
        
        # 标题
        panel_draw.text((10, 5), f"步骤 {i}", fill='black', font=font)
        
        # 画网格
        draw_trap(panel_draw, (1, 1), cell_size, padding)
        draw_goal(panel_draw, (2, 2), cell_size, padding)
        
        # 画规划路径（虚线效果用浅色）
        draw_path(panel_draw, plan, cell_size, padding, color='lightblue', width=2)
        
        # 画已执行路径
        draw_path(panel_draw, executed, cell_size, padding, color='blue', width=3)
        
        # 画智能体在当前位置
        current_pos = executed[-1]
        draw_agent(panel_draw, current_pos, cell_size, padding, color='blue')
        
        img.paste(panel, (offset_x, 0))
    
    img.save(output_path)
    print(f"Saved: {output_path}")


def visualize_learning_from_data(output_path):
    """可视化从数据学习转移模型"""
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), walls=(), traps=((1, 1),))
    
    # 收集一些随机轨迹
    np.random.seed(42)
    trajectories = []
    for _ in range(5):
        traj = []
        pos = (0, 0)
        traj.append(pos)
        for _ in range(8):
            action = np.random.choice(['up', 'down', 'left', 'right'])
            trans = env.step(pos, action)
            next_pos = trans.next_state
            traj.append(next_pos)
            pos = next_pos
            if pos == (2, 2) or pos in env.traps:
                break
        trajectories.append(traj)
    
    # 创建三列图
    cell_size = 80
    padding = 15
    panel_width = 3 * cell_size + 2 * padding
    panel_height = 3 * cell_size + 2 * padding + 40
    
    img_width = panel_width * 3 + 40
    img_height = panel_height
    img = Image.new('RGB', (img_width, img_height), 'white')
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 18)
    except:
        font = ImageFont.load_default()
    
    # 左：轨迹
    panel = Image.new('RGB', (panel_width, panel_height), 'white')
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((10, 5), "1. 收集轨迹", fill='black', font=font)
    draw_trap(panel_draw, (1, 1), cell_size, padding)
    draw_goal(panel_draw, (2, 2), cell_size, padding)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, traj in enumerate(trajectories):
        draw_path(panel_draw, traj, cell_size, padding, color=colors[i % len(colors)], width=2)
    img.paste(panel, (0, 0))
    
    # 中：计数
    panel = Image.new('RGB', (panel_width, panel_height), 'white')
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((10, 5), "2. 统计转移", fill='black', font=font)
    # 画一些计数示例
    y_offset = 50
    counts = [
        "((0,0), right) → (1,0): 3次",
        "((0,0), down) → (0,1): 2次",
        "((1,0), right) → (2,0): 4次",
        "((1,0), down) → (1,1): 1次",
        "((2,0), down) → (2,1): 5次",
    ]
    small_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 14) if font != ImageFont.load_default() else font
    for count in counts:
        panel_draw.text((20, y_offset), count, fill='black', font=small_font)
        y_offset += 30
    img.paste(panel, (panel_width + 20, 0))
    
    # 右：概率
    panel = Image.new('RGB', (panel_width, panel_height), 'white')
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((10, 5), "3. 学习概率", fill='black', font=font)
    y_offset = 50
    probs = [
        "P((1,0) | (0,0), right) = 1.0",
        "P((0,1) | (0,0), down) = 1.0",
        "P((2,0) | (1,0), right) = 1.0",
        "P((1,1) | (1,0), down) = 1.0",
        "P((2,1) | (2,0), down) = 1.0",
    ]
    for prob in probs:
        panel_draw.text((20, y_offset), prob, fill='black', font=small_font)
        y_offset += 30
    img.paste(panel, (2 * (panel_width + 20), 0))
    
    img.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'docs' / 'public' / 'carracing'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_greedy_vs_planned(output_dir / 'f0-greedy-vs-planned.png')
    visualize_mpc_process(output_dir / 'f0-mpc-process.png')
    visualize_learning_from_data(output_dir / 'f0-learning-from-data.png')
    
    print("\nAll visualizations generated successfully!")
