#!/usr/bin/env python3
"""
第 9 章评测可视化：用与 Notebook 相同的解析 toy 生成讲义配图。
数字必须与 notebooks/09_evaluation/test-a-world-model.ipynb 一致。
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hwm.evaluation import (
    RunManifest,
    RunTimer,
    calibration_bins,
    counterfactual_sensitivity,
    horizon_errors,
    runtime_summary,
)


# 与 F0 / 3.6 配图一致的中性色
RED = "#C0392B"
BLUE = "#2471A3"
GREEN = "#1E8449"
ORANGE = "#D35400"
GRAY = "#7F8C8D"
INK = "#2C3E50"
LIGHT = "#F4F6F7"
GRID = "#D5D8DC"
WHITE = "#FFFFFF"


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def rollout(start, actions, scale=1.0):
    """Notebook 里的一维世界：x ← x + scale * a。"""
    result, state = [], float(start)
    for action in actions:
        state += scale * action
        result.append(state)
    return np.asarray(result, dtype=np.float32)


def notebook_horizon():
    starts = [0.0, 1.0, -1.0]
    action_sequences = [[1.0] * 12, [-1.0] * 12, [1.0, -1.0] * 6]
    truth = np.stack([rollout(s, a, 1.0) for s, a in zip(starts, action_sequences)])
    errors = horizon_errors(lambda s, a: rollout(s, a, 0.9), starts, action_sequences, truth)
    return starts, action_sequences, truth, errors


def notebook_counterfactual():
    actions = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
    sensitivity = counterfactual_sensitivity(lambda s, a: rollout(s, a, 0.9), 0.0, actions)
    trajs = [rollout(0.0, a, 0.9) for a in actions]
    return actions, sensitivity, trajs


def notebook_ood():
    in_distribution = abs(rollout(0, [1], 0.9)[0] - rollout(0, [1], 1.0)[0])
    ood = abs(rollout(0, [3], 0.7)[0] - rollout(0, [3], 1.0)[0])
    return float(in_distribution), float(ood)


def notebook_calibration():
    probabilities = np.array([0.1, 0.2, 0.35, 0.65, 0.8, 0.95], dtype=np.float32)
    outcomes = np.array([0, 0, 1, 0, 1, 1], dtype=np.float32)
    return calibration_bins(probabilities, outcomes, num_bins=3)


def notebook_planner():
    candidates = np.linspace(-4, 4, 33)
    model_scores = -(0.9 * candidates - 3.0) ** 2
    true_scores = -(1.0 * candidates - 3.0) ** 2
    chosen = float(candidates[model_scores.argmax()])
    return candidates, model_scores, true_scores, chosen


def rounded(draw, box, fill=None, outline=INK, width=2, radius=10):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_axes(draw, x0, y0, x1, y1, xlabel="", ylabel="", font=None):
    draw.line([(x0, y1), (x1, y1)], fill=INK, width=2)
    draw.line([(x0, y0), (x0, y1)], fill=INK, width=2)
    if font and xlabel:
        draw.text((x1 - 48, y1 + 6), xlabel, fill=INK, font=font)
    if font and ylabel:
        draw.text((x0 - 8, y0 - 18), ylabel, fill=INK, font=font)


def visualize_interrogation(output_path):
    """六项审问总览：数字来自 Notebook 同一套 toy。"""
    _, _, _, errors = notebook_horizon()
    _, sensitivity, _ = notebook_counterfactual()
    id_err, ood_err = notebook_ood()
    bins = notebook_calibration()
    _, _, _, chosen = notebook_planner()

    pad, title_h = 24, 52
    cell_w, cell_h = 250, 168
    cols, rows = 3, 2
    img_w = cols * cell_w + (cols + 1) * pad
    img_h = title_h + rows * cell_h + (rows + 1) * pad + 8
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(11)

    draw.text((pad, 14), "审问一台世界模型：六项测试", fill=INK, font=title_font)

    cards = [
        (
            0,
            0,
            "① 多步 horizon",
            f"1 / 5 / 12 步 MSE\n{errors[0]:.2f}  →  {errors[4]:.2f}  →  {errors[11]:.2f}",
            "一步还看不出偏差\n十二步已经差一个数量级",
            RED,
        ),
        (
            1,
            0,
            "② 反事实动作",
            f"相对停留的差异\n{sensitivity[0]:.1f}  /  {sensitivity[1]:.1f}  /  {sensitivity[2]:.1f}",
            "同一起点，只换动作\n轨迹必须分开",
            BLUE,
        ),
        (
            2,
            0,
            "③ 单变量 OOD",
            f"训练内 |a|=1：{id_err:.1f}\n训练外 |a|=3：{ood_err:.1f}",
            "模型仍给出数字\n数字不必可靠",
            ORANGE,
        ),
        (
            0,
            1,
            "④ 不确定性校准",
            "\n".join(
                f"[{b['lower']:.2f},{b['upper']:.2f})  "
                f"p={b['confidence']:.2f}  实={b['frequency']:.2f}"
                for b in bins
            ),
            "说 80%，是否真有 80%",
            GREEN,
        ),
        (
            1,
            1,
            "⑤ Planner 漏洞",
            f"训练支持 [-1, 1]\nPlanner 选 a = {chosen:.2f}",
            "规划器会主动钻空子\n选训练里从没见过的动作",
            RED,
        ),
        (
            2,
            1,
            "⑥ Run Manifest",
            "seed / 命令 / 耗时\n硬件 / checkpoint 哈希",
            "没有清单，实验无法复现",
            INK,
        ),
    ]

    for col, row, title, body, foot, color in cards:
        x = pad + col * (cell_w + pad)
        y = title_h + pad + row * (cell_h + pad)
        rounded(draw, [x, y, x + cell_w, y + cell_h], fill=LIGHT, outline=color, width=2)
        draw.text((x + 12, y + 10), title, fill=color, font=font)
        draw.text((x + 12, y + 38), body, fill=INK, font=small)
        draw.text((x + 12, y + cell_h - 40), foot, fill=GRAY, font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_toy_world(output_path):
    """这就是要审问的东西：真实位移 1.0，模型只预测 0.9。"""
    steps = 12
    truth = rollout(0.0, [1.0] * steps, 1.0)
    pred = rollout(0.0, [1.0] * steps, 0.9)

    img_w, img_h = 820, 360
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(14)
    small = try_load_font(12)

    draw.text((24, 12), "被审问的世界：一维位移，模型故意少走一成", fill=INK, font=title_font)

    x0, y0, x1, y1 = 70, 60, 790, 280
    draw_axes(draw, x0, y0, x1, y1, "步数 h", "位置 x", small)

    def px(h, val):
        xx = x0 + int(h / steps * (x1 - x0 - 10))
        yy = y1 - int(val / 13.0 * (y1 - y0 - 10))
        return xx, yy

    for series, color, width in ((truth, BLUE, 3), (pred, RED, 3)):
        pts = [px(i + 1, float(v)) for i, v in enumerate(series)]
        draw.line(pts, fill=color, width=width)
        for p in pts:
            draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)

    draw.line([(x0 + 20, 310), (x0 + 50, 310)], fill=BLUE, width=3)
    draw.text((x0 + 56, 300), "真实世界  x ← x + a", fill=BLUE, font=font)
    draw.line([(x0 + 320, 310), (x0 + 350, 310)], fill=RED, width=3)
    draw.text((x0 + 356, 300), "有偏模型  x ← x + 0.9 a", fill=RED, font=font)
    draw.text((x0 + 20, 332), "第 12 步：真实 12.0，模型 10.8。一步几乎看不出，多步才会拉开。", fill=GRAY, font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_horizon_errors(output_path):
    _, _, _, errors = notebook_horizon()
    img_w, img_h = 720, 400
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)

    draw.text((24, 12), "测试一：多步 horizon 曲线", fill=INK, font=title_font)

    x0, y0, x1, y1 = 70, 56, 680, 310
    draw_axes(draw, x0, y0, x1, y1, "horizon h", "MSE", small)
    max_e = float(errors.max())

    pts = []
    for i, e in enumerate(errors):
        xx = x0 + int((i / (len(errors) - 1)) * (x1 - x0 - 8))
        yy = y1 - int(float(e) / max_e * (y1 - y0 - 12))
        pts.append((xx, yy))
    draw.line(pts, fill=RED, width=3)
    for p in pts:
        draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=RED)

    marks = [(0, "h=1"), (4, "h=5"), (11, "h=12")]
    for idx, label in marks:
        xx, yy = pts[idx]
        draw.text((xx - 10, yy - 22), f"{float(errors[idx]):.2f}", fill=RED, font=small)
        draw.text((xx - 12, y1 + 8), label, fill=GRAY, font=small)

    draw.text(
        (24, 340),
        f"三条轨迹平均：horizon 1 / 5 / 12 的 MSE = "
        f"{float(errors[0]):.2f} / {float(errors[4]):.2f} / {float(errors[11]):.2f}",
        fill=INK,
        font=font,
    )
    draw.text((24, 364), "复合误差：每一步 10% 的位移偏差，滚到第 12 步已经接近 1.0。", fill=GRAY, font=small)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_horizon_process(output_path):
    """步骤图：三条动作各自滚 12 步，误差按序列单独算。"""
    starts, action_sequences, truth, _errors = notebook_horizon()
    pred = np.stack([rollout(s, a, 0.9) for s, a in zip(starts, action_sequences)])
    per_seq = (pred - truth) ** 2

    labels = ["一直向右  a=1", "一直向左  a=-1", "左右交替  a=±1"]
    horizons = [1, 5, 12]
    panel_w, panel_h = 250, 230
    pad, title_h = 20, 48
    img_w = 3 * panel_w + 4 * pad
    img_h = title_h + panel_h + pad + 70
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(11)

    draw.text((pad, 12), "测试一的步骤：同一条动作，模型自己喂自己", fill=INK, font=title_font)

    for i, (lab, t_row, p_row) in enumerate(zip(labels, truth, pred)):
        x = pad + i * (panel_w + pad)
        y = title_h
        rounded(draw, [x, y, x + panel_w, y + panel_h], fill=LIGHT, outline=GRID, width=1)
        draw.text((x + 12, y + 8), lab, fill=INK, font=font)

        ax0, ax1, ay = x + 20, x + panel_w - 20, y + 118
        draw.line([(ax0, ay), (ax1, ay)], fill=INK, width=2)
        lo = min(float(t_row.min()), float(p_row.min()), float(starts[i])) - 1.2
        hi = max(float(t_row.max()), float(p_row.max()), float(starts[i])) + 1.2

        def ax(val, _lo=lo, _hi=hi):
            return ax0 + int((val - _lo) / (_hi - _lo) * (ax1 - ax0))

        start_x = ax(starts[i])
        draw.ellipse([start_x - 5, ay - 5, start_x + 5, ay + 5], fill=INK)
        draw.text((start_x - 12, ay + 10), "起点", fill=INK, font=small)

        tx = ax(float(t_row[-1]))
        px = ax(float(p_row[-1]))
        draw.ellipse([tx - 6, ay - 18, tx + 6, ay - 6], fill=BLUE)
        draw.ellipse([px - 6, ay - 18, px + 6, ay - 6], fill=RED)
        if abs(tx - px) > 18:
            draw.text((tx - 6, ay - 36), "真", fill=BLUE, font=small)
            draw.text((px - 6, ay - 36), "模", fill=RED, font=small)
        else:
            draw.text((min(tx, px) - 18, ay - 36), "真/模", fill=INK, font=small)

        lines = [f"h={h}  本条 MSE={float(per_seq[i, h - 1]):.2f}" for h in horizons]
        draw.text((x + 12, y + 158), "\n".join(lines), fill=GRAY, font=small)

    draw.text(
        (pad, title_h + panel_h + 16),
        "左、中两条同向累加，第 12 步本条 MSE=1.44；右条左右对消，第 12 步碰巧重合。三条平均才是 0.96。",
        fill=GRAY,
        font=font,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_counterfactual(output_path):
    actions, sensitivity, trajs = notebook_counterfactual()
    img_w, img_h = 780, 380
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)

    draw.text((24, 12), "测试二：固定起点，只换动作", fill=INK, font=title_font)

    x0, y0, x1, y1 = 70, 56, 520, 300
    draw_axes(draw, x0, y0, x1, y1, "步数", "位置", small)

    colors = [GRAY, BLUE, RED]
    names = ["停留  [0,0,0]", "向右  [1,1,1]", "向左  [-1,-1,-1]"]
    for traj, color in zip(trajs, colors):
        pts = [(x0, (y0 + y1) // 2)]
        for i, v in enumerate(traj):
            xx = x0 + int((i + 1) / 3 * (x1 - x0 - 8))
            yy = (y0 + y1) // 2 - int(float(v) / 3.2 * ((y1 - y0) / 2 - 8))
            pts.append((xx, yy))
        draw.line(pts, fill=color, width=3)
        for p in pts[1:]:
            draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=color)

    sx, sy = 540, 70
    for i, (name, color, s) in enumerate(zip(names, colors, sensitivity)):
        yy = sy + i * 70
        rounded(draw, [sx, yy, sx + 220, yy + 58], fill=LIGHT, outline=color, width=2)
        draw.text((sx + 12, yy + 8), name, fill=color, font=font)
        draw.text((sx + 12, yy + 30), f"Δ = {float(s):.1f}", fill=INK, font=font)

    draw.text((24, 332), "三条轨迹从 x=0 出发。换动作后未来必须分开；若 Δ 全是 0，模型只在抄惯性。", fill=GRAY, font=small)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_ood(output_path):
    id_err, ood_err = notebook_ood()
    img_w, img_h = 820, 360
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(14)
    small = try_load_font(12)

    draw.text((24, 12), "测试三：只改一个变量——动作幅度", fill=INK, font=title_font)

    panels = [
        ("训练分布内", "a = 1，模型尺度 0.9", id_err, "真实下一步 = 1.0\n模型下一步 = 0.9", BLUE),
        ("训练分布外", "a = 3，模型尺度掉到 0.7", ood_err, "真实下一步 = 3.0\n模型下一步 = 2.1", RED),
    ]
    for i, (title, sub, err, body, color) in enumerate(panels):
        x = 24 + i * 400
        y = 56
        rounded(draw, [x, y, x + 372, y + 230], fill=LIGHT, outline=color, width=2)
        draw.text((x + 16, y + 12), title, fill=color, font=title_font)
        draw.text((x + 16, y + 46), sub, fill=INK, font=font)
        draw.text((x + 16, y + 80), body, fill=GRAY, font=small)
        bar_max = 1.0
        bw = int(err / bar_max * 300)
        draw.rectangle([x + 16, y + 150, x + 16 + 300, y + 178], outline=GRID, width=1)
        draw.rectangle([x + 16, y + 150, x + 16 + bw, y + 178], fill=color)
        draw.text((x + 16, y + 188), f"绝对误差  {err:.1f}", fill=color, font=font)

    draw.text((24, 304), "OOD 条件必须事先写死：这里只把动作从 1 改成 3。误差从 0.1 跳到 0.9。", fill=GRAY, font=small)
    draw.text((24, 328), "模型仍然吐出一个看起来合理的数字——这正是「不懂装懂」。", fill=GRAY, font=small)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_calibration(output_path):
    bins = notebook_calibration()
    img_w, img_h = 720, 420
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)

    draw.text((24, 12), "测试四：置信度分箱以后，还在不在对角线上", fill=INK, font=title_font)

    x0, y0, x1, y1 = 80, 56, 400, 330
    draw_axes(draw, x0, y0, x1, y1, "置信度", "频率", small)
    draw.line([(x0, y1), (x1, y0)], fill=GRID, width=2)

    for b in bins:
        px = x0 + int(b["confidence"] * (x1 - x0))
        py = y1 - int(b["frequency"] * (y1 - y0))
        r = 7
        draw.ellipse([px - r, py - r, px + r, py + r], fill=BLUE, outline=INK)
        draw.text((px + 10, py - 8), f"n={b['count']}", fill=GRAY, font=small)

    sx, sy = 430, 70
    draw.text((sx, sy), "三个箱子（各 2 个样本）", fill=INK, font=font)
    for i, b in enumerate(bins):
        yy = sy + 36 + i * 70
        gap = b["confidence"] - b["frequency"]
        color = ORANGE if abs(gap) > 0.05 else GREEN
        rounded(draw, [sx, yy, sx + 260, yy + 60], fill=LIGHT, outline=color, width=2)
        draw.text(
            (sx + 10, yy + 8),
            f"[{b['lower']:.2f}, {b['upper']:.2f})",
            fill=INK,
            font=small,
        )
        draw.text(
            (sx + 10, yy + 30),
            f"p = {b['confidence']:.2f}    实发 = {b['frequency']:.2f}",
            fill=color,
            font=small,
        )

    ece = float(np.mean([abs(b["confidence"] - b["frequency"]) for b in bins]))
    draw.text((24, 352), f"三个箱子的平均 |p − 频率| ≈ {ece:.2f}。低箱偏高估，高箱略低估。", fill=INK, font=font)
    draw.text((24, 376), "六个样本只能示范算法，不能给校准下结论——这是教学版故意留下的坑。", fill=GRAY, font=small)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_planner(output_path):
    candidates, model_scores, true_scores, chosen = notebook_planner()
    img_w, img_h = 820, 400
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)

    draw.text((24, 12), "测试五：Planner 会主动走到训练支持外面", fill=INK, font=title_font)

    x0, y0, x1, y1 = 70, 56, 790, 300
    draw_axes(draw, x0, y0, x1, y1, "动作 a", "分数", small)

    def mapx(a):
        return x0 + int((a + 4) / 8 * (x1 - x0))

    def mapy(s):
        # scores in [-about 50, 0]
        return y1 - int((s + 50) / 50 * (y1 - y0 - 8))

    # training support band
    band_l, band_r = mapx(-1), mapx(1)
    draw.rectangle([band_l, y0, band_r, y1], fill="#EAF2F8")
    draw.text((band_l + 8, y0 + 8), "训练支持 [-1, 1]", fill=BLUE, font=small)

    mpts = [(mapx(float(a)), mapy(float(s))) for a, s in zip(candidates, model_scores)]
    tpts = [(mapx(float(a)), mapy(float(s))) for a, s in zip(candidates, true_scores)]
    draw.line(tpts, fill=GRAY, width=2)
    draw.line(mpts, fill=RED, width=3)

    cx, cy = mapx(chosen), mapy(float(model_scores.max()))
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=RED, outline=INK)
    draw.text((cx - 70, cy - 24), f"Planner 选 a = {chosen:.2f}", fill=RED, font=font)

    draw.line([(90, 330), (130, 330)], fill=RED, width=3)
    draw.text((136, 322), "模型分数  −(0.9a − 3)²", fill=RED, font=font)
    draw.line([(320, 330), (360, 330)], fill=GRAY, width=2)
    draw.text((366, 322), "真实分数  −(a − 3)²", fill=GRAY, font=font)
    draw.text((24, 356), "模型以为 0.9a ≈ 3，于是把 a 推到 3.25。训练里从未见过 |a|>1。", fill=GRAY, font=small)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_manifest(output_path):
    with RunTimer() as timer:
        _ = sum(range(1000))
    manifest = RunManifest(
        experiment="evaluation-smoke",
        route="evaluation",
        seed=0,
        dataset="analytic-toy",
        split="test",
        command="run test-a-world-model notebook",
        started_at="2026-08-12T00:00:00Z",
        wall_time_seconds=timer.seconds,
        notes="CPU 教学 smoke，不是 24GB 证据",
    )
    summary = runtime_summary()
    payload = {
        "experiment": manifest.experiment,
        "route": manifest.route,
        "seed": manifest.seed,
        "dataset": manifest.dataset,
        "split": manifest.split,
        "command": manifest.command,
        "started_at": manifest.started_at,
        "wall_time_seconds": round(manifest.wall_time_seconds, 8),
        "device": manifest.device,
        "gpu": manifest.gpu,
        "cuda": manifest.cuda,
        "peak_allocated_mb": manifest.peak_allocated_mb,
        "peak_reserved_mb": manifest.peak_reserved_mb,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "notes": manifest.notes,
        "runtime": summary,
    }

    img_w, img_h = 860, 460
    canvas = Image.new("RGB", (img_w, img_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    mono = try_load_font(12)

    draw.text((24, 12), "测试六：一次运行要留下什么", fill=INK, font=title_font)

    left = [
        ("必须写清", BLUE),
        ("experiment / route / seed", INK),
        ("dataset + split", INK),
        ("完整命令", INK),
        ("started_at + wall time", INK),
        ("device / gpu / cuda", INK),
        ("peak allocated 与 reserved", INK),
        ("checkpoint sha256", INK),
        ("", INK),
        ("这次教学运行", ORANGE),
        (f"python {summary['python']}", GRAY),
        (f"gpu = {manifest.gpu}", GRAY),
        ("peak_reserved_mb = None", GRAY),
        ("所以它不是 24GB 证据", ORANGE),
    ]
    x, y = 24, 56
    rounded(draw, [x, y, x + 330, y + 380], fill=LIGHT, outline=BLUE, width=2)
    yy = y + 14
    for text, color in left:
        if not text:
            yy += 10
            continue
        draw.text((x + 16, yy), text, fill=color, font=font)
        yy += 24

    blob = json.dumps(payload, ensure_ascii=False, indent=2)
    lines = blob.splitlines()
    rx, ry = 370, 56
    rounded(draw, [rx, ry, 836, 436], fill="#1C2833", outline=INK, width=1)
    draw.text((rx + 14, ry + 10), "manifest.json", fill="#AED6F1", font=font)
    ty = ry + 36
    for line in lines[:18]:
        draw.text((rx + 14, ty), line[:72], fill="#EAECEE", font=mono)
        ty += 18
        if ty > 410:
            break

    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "docs" / "public" / "carracing"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_interrogation(output_dir / "interrogation.png")
    visualize_toy_world(output_dir / "toy-world.png")
    visualize_horizon_errors(output_dir / "horizon-errors.png")
    visualize_horizon_process(output_dir / "horizon-process.png")
    visualize_counterfactual(output_dir / "counterfactual.png")
    visualize_ood(output_dir / "ood.png")
    visualize_calibration(output_dir / "calibration.png")
    visualize_planner(output_dir / "planner-exploit.png")
    visualize_manifest(output_dir / "run-manifest.png")

    print("\nAll evaluation visualizations generated!")
