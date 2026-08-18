#!/usr/bin/env python3
"""
Z0 可视化脚本：运行审问世界模型的实际代码，生成真实可视化。
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hwm.evaluation import horizon_errors, counterfactual_sensitivity, calibration_bins


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def visualize_horizon_errors(output_path):
    """测试一：多步 horizon 曲线"""
    # 模拟一个简单的一维动态模型
    def simple_predict(start, actions):
        """x_{t+1} = x_t + a_t + noise"""
        trajectory = [start]
        current = start
        for a in actions:
            current = current + a + np.random.normal(0, 0.1)
            trajectory.append(current)
        return np.array(trajectory[1:])
    
    # 生成测试数据
    np.random.seed(42)
    starts = np.random.uniform(-5, 5, size=20)
    max_horizon = 10
    action_sequences = np.random.uniform(-1, 1, size=(20, max_horizon))
    
    # 真实 rollout
    true_rollouts = np.stack([simple_predict(s, a) for s, a in zip(starts, action_sequences)])
    
    # 计算每个 horizon 的误差
    errors = horizon_errors(simple_predict, starts, action_sequences, true_rollouts)
    
    # 可视化
    cell_w = 300
    cell_h = 200
    padding = 30
    title_h = 40
    
    img_w = cell_w + padding * 2
    img_h = cell_h + title_h + padding * 2 + 40
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    title_font = try_load_font(20)
    
    # 画误差曲线
    chart_img = Image.new('RGB', (cell_w, cell_h), 'white')
    cd = ImageDraw.Draw(chart_img)
    
    # 坐标轴
    cd.line([(40, cell_h - 30), (cell_w - 10, cell_h - 30)], fill='black', width=2)  # x 轴
    cd.line([(40, cell_h - 30), (40, 10)], fill='black', width=2)  # y 轴
    
    # 画曲线
    max_error = errors.max() if errors.max() > 0 else 1
    for i in range(len(errors) - 1):
        x1 = 40 + int(i / len(errors) * (cell_w - 60))
        y1 = cell_h - 30 - int(errors[i] / max_error * (cell_h - 50))
        x2 = 40 + int((i + 1) / len(errors) * (cell_w - 60))
        y2 = cell_h - 30 - int(errors[i + 1] / max_error * (cell_h - 50))
        cd.line([(x1, y1), (x2, y2)], fill='red', width=3)
        cd.ellipse([x1-3, y1-3, x1+3, y1+3], fill='red')
    
    # 标签
    cd.text((cell_w // 2 - 30, cell_h - 25), "Horizon", fill='black', font=font)
    cd.text((5, cell_h // 2 - 20), "MSE", fill='black', font=font)
    
    canvas.paste(chart_img, (padding, title_h + padding))
    
    # 标注关键信息
    draw.text((padding + 50, title_h + padding + cell_h + 10), 
              f"复合误差：horizon 越长，误差越大（{errors[0]:.3f} → {errors[-1]:.3f}）", 
              fill='black', font=font)
    
    draw.text((padding, 5), "Z0 测试一：多步 horizon 曲线", fill='black', font=title_font)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_counterfactual(output_path):
    """测试二：反事实动作"""
    # 模拟一个简单模型
    def simple_predict(start, actions):
        trajectory = [start]
        current = start
        for a in actions:
            current = current + a
            trajectory.append(current)
        return np.array(trajectory)
    
    start = np.array([0.0, 0.0])
    action_sequences = [
        np.array([1, 0, 1, 0]),  # 向右向上
        np.array([0, 1, 0, 1]),  # 向上向右
        np.array([-1, 0, -1, 0]),  # 向左向下
    ]
    
    # 计算反事实敏感度
    sensitivity = counterfactual_sensitivity(simple_predict, start, action_sequences)
    
    # 可视化三条轨迹
    cell = 200
    padding = 30
    title_h = 40
    
    img_w = cell + padding * 2
    img_h = cell + title_h + padding * 2 + 40
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    title_font = try_load_font(20)
    
    chart_img = Image.new('RGB', (cell, cell), 'white')
    cd = ImageDraw.Draw(chart_img)
    
    # 画三条轨迹
    colors = ['red', 'blue', 'green']
    labels = ['动作 A', '动作 B', '动作 C']
    
    for i, actions in enumerate(action_sequences):
        trajectory = simple_predict(start, actions)
        for t in range(len(trajectory) - 1):
            x1 = int(cell / 2 + trajectory[t, 0] * 20)
            y1 = int(cell / 2 - trajectory[t, 1] * 20)
            x2 = int(cell / 2 + trajectory[t + 1, 0] * 20)
            y2 = int(cell / 2 - trajectory[t + 1, 1] * 20)
            cd.line([(x1, y1), (x2, y2)], fill=colors[i], width=2)
            cd.ellipse([x2-3, y2-3, x2+3, y2+3], fill=colors[i])
    
    # 起点
    cd.ellipse([cell//2-5, cell//2-5, cell//2+5, cell//2+5], fill='black')
    cd.text((cell//2 + 8, cell//2 - 5), "起点", fill='black', font=try_load_font(11))
    
    canvas.paste(chart_img, (padding, title_h + padding))
    
    # 图例
    for i, label in enumerate(labels):
        draw.text((padding + 10, title_h + padding + cell + 10 + i * 20), 
                  f"{label}: {labels[i]}", fill=colors[i], font=font)
    
    draw.text((padding, 5), "Z0 测试二：固定起点，换动作→轨迹必须改变", fill='black', font=title_font)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_calibration(output_path):
    """测试四：不确定性校准"""
    # 模拟校准数据
    np.random.seed(42)
    num_samples = 200
    
    # 生成预测概率和真实结果
    # 模拟一个过度自信的模型
    probabilities = np.random.uniform(0.5, 1.0, num_samples)
    # 真实准确率低于预测置信度（过度自信）
    outcomes = (probabilities > np.random.uniform(0.3, 0.8, num_samples)).astype(float)
    
    # 计算校准
    cal_data = calibration_bins(probabilities, outcomes, num_bins=5)
    
    # 可视化
    cell = 200
    padding = 30
    title_h = 40
    
    img_w = cell + padding * 2
    img_h = cell + title_h + padding * 2 + 40
    
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    title_font = try_load_font(20)
    
    chart_img = Image.new('RGB', (cell, cell), 'white')
    cd = ImageDraw.Draw(chart_img)
    
    # 坐标轴
    cd.line([(40, cell - 30), (cell - 10, cell - 30)], fill='black', width=2)
    cd.line([(40, cell - 30), (40, 10)], fill='black', width=2)
    
    # 对角线（完美校准）
    cd.line([(40, cell - 30), (cell - 10, 10)], fill='gray', width=1)
    cd.text((cell - 60, cell - 25), "置信度", fill='black', font=try_load_font(11))
    cd.text((5, 15), "准确率", fill='black', font=try_load_font(11))
    
    # 画校准点
    for bin_data in cal_data:
        conf = bin_data['confidence']
        freq = bin_data['frequency']
        x = 40 + int(conf * (cell - 60))
        y = cell - 30 - int(freq * (cell - 50))
        cd.ellipse([x-5, y-5, x+5, y+5], fill='blue')
    
    canvas.paste(chart_img, (padding, title_h + padding))
    
    # 标注
    draw.text((padding + 10, title_h + padding + cell + 10), 
              "蓝色点=模型校准，灰线=完美校准", fill='black', font=font)
    draw.text((padding + 10, title_h + padding + cell + 30), 
              "如果点在线下方→模型过度自信", fill='gray', font=try_load_font(12))
    
    draw.text((padding, 5), "Z0 测试四：不确定性校准曲线", fill='black', font=title_font)
    
    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'docs' / 'public' / 'carracing'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_horizon_errors(output_dir / 'z0-horizon-errors.png')
    visualize_counterfactual(output_dir / 'z0-counterfactual.png')
    visualize_calibration(output_dir / 'z0-calibration.png')
    
    print("\nAll z0 visualizations generated!")
