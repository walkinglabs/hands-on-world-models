#!/usr/bin/env python3
"""Generate lecture figures for PA1-D/E, 8.5 and 8.6 from live toy runs."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hwm.evaluation import calibration_bins, counterfactual_sensitivity, horizon_errors
from hwm.foundations import (
    depth_to_points,
    make_camera_transform,
    points_to_occupancy,
    transform_points,
)
from hwm.robot import render_tabletop, step_tabletop


OUT = Path(__file__).parent.parent / "docs" / "public" / "carracing"


def font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def caption_box(draw, xy, text, fill, f):
    draw.text(xy, text, fill=fill, font=f)


def save(img, name):
    path = OUT / name
    img.save(path)
    print("saved", path, img.size)


def tabletop():
    state = np.array([0.20, 0.50, 0.85, 0.50, 0.20, 0.85, 0.31, 0.50], dtype=np.float32)
    image = render_tabletop(state, size=32)
    up = np.array(Image.fromarray(image).resize((256, 256), Image.NEAREST))
    canvas = Image.new("RGB", (860, 360), "white")
    canvas.paste(Image.fromarray(up), (24, 52))
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (24, 12), "桌面状态：白=抓手，红/绿=目标，蓝=障碍", "black", font(18))
    lines = [
        "state = [gx, gy, rx, ry, gx2, gy2, ox, oy]",
        "本例: 抓手 (0.20, 0.50)",
        "红色目标 (0.85, 0.50)，障碍挡在正前方 (0.31, 0.50)",
        "直达动作 a=[1,0] 一步就会撞",
        "斜向 a=[0.7,-0.7] 更远，但可能绕开",
        "指令只有两条：移动到红色 / 绿色目标",
    ]
    y = 60
    for line in lines:
        caption_box(draw, (300, y), line, (40, 40, 40), font(15))
        y += 42
    save(canvas, "tabletop.png")


def action_rerank():
    canvas = Image.new("RGB", (860, 340), "white")
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (24, 12), "直达必撞场景：重排前后的真实闭环", "black", font(20))
    bars = [
        ("直达碰撞率", 1.000, (200, 70, 70), "1.000"),
        ("重排后碰撞率", 0.328, (40, 130, 90), "0.328"),
        ("平均进展", -0.036, (70, 90, 180), "-0.036"),
    ]
    x0, y0, w, h = 250, 70, 480, 56
    for i, (name, value, color, label) in enumerate(bars):
        y = y0 + i * 80
        caption_box(draw, (24, y + 16), name, (30, 30, 30), font(18))
        draw.rectangle([x0, y, x0 + w, y + h], outline=(180, 180, 180), width=1)
        width = int(abs(value) * w) if name != "平均进展" else int(0.18 * w)
        draw.rectangle([x0, y, x0 + max(width, 10), y + h], fill=color)
        caption_box(draw, (x0 + 14, y + 16), label, "white", font(18))
    caption_box(
        draw,
        (24, 300),
        "64 个直达必撞场景，collision_weight=4.0。碰撞从 1.000 降到 0.328，平均进展却是 -0.036。",
        (60, 60, 60),
        font(15),
    )
    save(canvas, "action-rerank.png")


def unproject():
    depth = np.full((6, 6), 4.0, dtype=np.float32)
    depth[2:4, 2:4] = 2.0
    points = depth_to_points(depth, fx=6, fy=6, cx=2.5, cy=2.5)
    world = transform_points(points, make_camera_transform(tx=1.0))
    occ = points_to_occupancy(world, (-2, 4), (0, 6), 0.5)
    wrong = transform_points(points, make_camera_transform(tx=1.3))
    wrong_occ = points_to_occupancy(wrong, (-2, 4), (0, 6), 0.5)

    def grid_img(grid, scale=18):
        h, w = grid.shape
        img = Image.new("RGB", (w * scale, h * scale), (245, 245, 245))
        d = ImageDraw.Draw(img)
        for r in range(h):
            for c in range(w):
                color = (40, 40, 40) if grid[r, c] else (230, 230, 230)
                d.rectangle(
                    [c * scale, r * scale, (c + 1) * scale - 1, (r + 1) * scale - 1],
                    fill=color,
                    outline=(210, 210, 210),
                )
        return img

    canvas = Image.new("RGB", (900, 360), "white")
    canvas.paste(grid_img(occ), (30, 70))
    canvas.paste(grid_img(wrong_occ), (330, 70))
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (30, 16), "E1：6×6 深度反投影后的俯视占用", "black", font(20))
    caption_box(draw, (30, 44), "平移 1.0 m，占用 8 格", (40, 40, 40), font(14))
    caption_box(draw, (330, 44), "平移写成 1.3 m，占用 7 格", (40, 40, 40), font(14))
    lines = [
        "36 个点，近/远 z = 2.0 / 4.0",
        "正确平移 [1, 0, 0]",
        "写错 0.3 m 后点云整体平移",
        "占用 IoU 从 1.0 掉到 0.5",
        "几何错了，后面的网络救不回来",
    ]
    y = 80
    for line in lines:
        caption_box(draw, (630, y), line, (40, 40, 40), font(15))
        y += 40
    save(canvas, "unproject.png")


def occupancy():
    canvas = Image.new("RGB", (860, 300), "white")
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (24, 12), "E2b smoke：学到的占用 vs 复制上一帧", "black", font(20))
    items = [("复制上一帧 IoU", 0.277), ("学到的预测 IoU", 0.436)]
    x0, y0, max_w, h = 80, 80, 560, 64
    for i, (name, value) in enumerate(items):
        y = y0 + i * 90
        draw.rectangle([x0, y, x0 + max_w, y + h], outline=(180, 180, 180), width=1)
        w = int(value / 0.6 * max_w)
        color = (180, 120, 40) if i == 0 else (40, 120, 90)
        draw.rectangle([x0, y, x0 + w, y + h], fill=color)
        caption_box(draw, (x0 + 12, y + 18), f"{name}  {value:.3f}", "white", font(18))
    caption_box(
        draw,
        (80, 260),
        "96 个样本、16×16 BEV、历史 3 帧、未来 3 帧。IoU 0.436 超过复制基线，但仍远不到可用。",
        (60, 60, 60),
        font(14),
    )
    save(canvas, "occupancy.png")


def horizon():
    def rollout(start, actions, scale=1.0):
        result, state = [], float(start)
        for action in actions:
            state += scale * action
            result.append(state)
        return np.asarray(result)

    starts = [0.0, 1.0, -1.0]
    action_sequences = [[1.0] * 12, [-1.0] * 12, [1.0, -1.0] * 6]
    truth = np.stack([rollout(s, a, 1.0) for s, a in zip(starts, action_sequences)])
    model = horizon_errors(lambda s, a: rollout(s, a, 0.9), starts, action_sequences, truth)

    w, h = 760, 360
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (20, 10), "0.9 倍位移：一步很准，十二步接近 1", "black", font(20))
    ox, oy, cw, ch = 70, 50, 520, 240
    draw.rectangle([ox, oy, ox + cw, oy + ch], outline="black", width=2)
    ymax = 1.0
    def xy(i, val):
        x = ox + int(i / 11 * cw)
        y = oy + ch - int(min(val, ymax) / ymax * ch)
        return x, y

    pts = [xy(i, float(v)) for i, v in enumerate(model)]
    draw.line(pts, fill=(200, 50, 50), width=3)
    for p in pts:
        draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(200, 50, 50))
    caption_box(draw, (610, 80), "H=1   0.01", (40, 40, 40), font(16))
    caption_box(draw, (610, 116), "H=5   0.17", (40, 40, 40), font(16))
    caption_box(draw, (610, 152), "H=12  0.96", (40, 40, 40), font(16))
    caption_box(draw, (ox, 310), "误差按 horizon 单调上升。复制起点在第 12 步会到几十，图里不画，免得把这根曲线压扁。", (50, 50, 50), font(14))
    save(canvas, "z85-horizon.png")


def counterfactual():
    def rollout(start, actions, scale=0.9):
        result, state = [], float(start)
        for action in actions:
            state += scale * action
            result.append(state)
        return np.asarray(result)

    seqs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
    labels = ["停留", "持续 +1", "持续 -1"]
    colors = [(40, 40, 40), (30, 110, 200), (30, 140, 80)]
    sens = counterfactual_sensitivity(lambda s, a: rollout(s, a), 0.0, seqs)

    w, h = 760, 340
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (20, 10), "同一起点，只换动作", "black", font(20))
    ox, oy, cw, ch = 60, 50, 520, 220
    draw.rectangle([ox, oy, ox + cw, oy + ch], outline="black", width=2)
    for seq, color in zip(seqs, colors):
        pred = rollout(0.0, seq)
        xs = [0] + list(range(1, 4))
        ys = [0.0] + list(pred)
        pts = []
        for t, val in zip(xs, ys):
            x = ox + int(t / 3 * cw)
            y = oy + ch // 2 - int(val / 3.0 * (ch // 2 - 12))
            pts.append((x, y))
        draw.line(pts, fill=color, width=3)
        for p in pts:
            draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=color)
    caption_box(draw, (600, 70), f"{labels[0]}  Δ= {sens[0]:.1f}", colors[0], font(15))
    caption_box(draw, (600, 110), f"{labels[1]}  Δ= {sens[1]:.1f}", colors[1], font(15))
    caption_box(draw, (600, 150), f"{labels[2]}  Δ= {sens[2]:.1f}", colors[2], font(15))
    caption_box(draw, (60, 290), "相对停留的平均绝对差是 1.8。如果三条线重合，动作条件就没学到。", (50, 50, 50), font(14))
    save(canvas, "z85-counterfactual.png")


def calibration_planner():
    probabilities = np.array([0.1, 0.2, 0.35, 0.65, 0.8, 0.95])
    outcomes = np.array([0, 0, 1, 0, 1, 1])
    bins = calibration_bins(probabilities, outcomes, num_bins=3)

    w, h = 900, 360
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (20, 10), "校准分箱与 Planner 漏洞", "black", font(20))

    ox, oy, s = 50, 50, 220
    draw.rectangle([ox, oy, ox + s, oy + s], outline="black", width=2)
    draw.line([ox, oy + s, ox + s, oy], fill=(170, 170, 170), width=2)
    for item in bins:
        x = ox + int(item["confidence"] * s)
        y = oy + s - int(item["frequency"] * s)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(40, 90, 180))
    caption_box(draw, (50, 290), "低箱 0.15 vs 0.00；中箱 0.50 vs 0.50；高箱 0.88 vs 1.00", (50, 50, 50), font(13))

    candidates = np.linspace(-4, 4, 33)
    scores = -((0.9 * candidates - 3.0) ** 2)
    chosen = float(candidates[scores.argmax()])
    px, py, pw, ph = 360, 70, 500, 200
    draw.rectangle([px, py, px + pw, py + ph], outline="black", width=2)
    draw.rectangle([px + int(((-1) + 4) / 8 * pw), py, px + int((1 + 4) / 8 * pw), py + ph], fill=(235, 245, 235))
    pts = []
    smin, smax = float(scores.min()), float(scores.max())
    for a, sc in zip(candidates, scores):
        x = px + int((a + 4) / 8 * pw)
        y = py + ph - int((sc - smin) / (smax - smin) * ph)
        pts.append((x, y))
    draw.line(pts, fill=(200, 60, 40), width=3)
    cx = px + int((chosen + 4) / 8 * pw)
    draw.line([cx, py, cx, py + ph], fill=(20, 20, 20), width=2)
    caption_box(draw, (360, 280), f"训练支持 [-1, 1]（浅绿），Planner 选中 a={chosen:.2f}", (50, 50, 50), font(14))
    save(canvas, "z85-calibration-planner.png")


def pa2_falsify():
    canvas = Image.new("RGB", (880, 320), "white")
    draw = ImageDraw.Draw(canvas)
    caption_box(draw, (20, 12), "跑实验之前先写下四种结果怎么判", "black", font(20))
    boxes = [
        ((30, 60), "目标场景升、其它不变", "支持你选的解释"),
        ((460, 60), "所有场景一起升一点", "可能只是参数/训练更久"),
        ((30, 180), "目标场景几乎不动", "解释错了，换另一条"),
        ((460, 180), "目标场景更差", "改动有害，负结果也算完成"),
    ]
    for (x, y), title, body in boxes:
        draw.rectangle([x, y, x + 390, y + 100], outline=(160, 160, 160), width=2)
        caption_box(draw, (x + 14, y + 16), title, (20, 20, 20), font(16))
        caption_box(draw, (x + 14, y + 52), body, (70, 70, 70), font(15))
    save(canvas, "z86-falsify.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tabletop()
    action_rerank()
    unproject()
    occupancy()
    horizon()
    counterfactual()
    calibration_planner()
    pa2_falsify()
