#!/usr/bin/env python3
"""
路线 D/E 可视化：用 notebook 同配置跑出真实桌面、几何与占用图。
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hwm.foundations import (
    depth_to_points,
    make_camera_transform,
    points_to_occupancy,
    transform_points,
)
from hwm.robot import (
    INSTRUCTIONS,
    TabletopOutcomeModel,
    TinyVLA,
    evaluate_reranker,
    evaluate_vla,
    make_outcome_dataset,
    make_tabletop_dataset,
    outcome_loss,
    render_tabletop,
    rerank_actions,
    step_tabletop,
)
from hwm.spatial import (
    TinyDynamicField,
    TinyNeuralField,
    TinyOccupancyPredictor,
    make_colored_sphere_samples,
    make_moving_occupancy_dataset,
    make_moving_sphere_samples,
    occupancy_iou,
)


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _upsample(array, scale=8):
    image = np.asarray(array)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return Image.fromarray(image).resize(
        (image.shape[1] * scale, image.shape[0] * scale), Image.NEAREST
    )


def _grid_to_rgb(grid, occupied=(220, 70, 50), empty=(245, 247, 250)):
    grid = np.asarray(grid)
    rgb = np.zeros(grid.shape + (3,), dtype=np.uint8)
    rgb[:] = empty
    mask = grid > 0.5
    rgb[mask] = occupied
    return rgb


def visualize_tabletop(output_path):
    """这就是世界：一张 32×32 的桌子，白点是手，红/绿是目标，蓝是障碍。"""
    torch.manual_seed(0)
    data = make_tabletop_dataset(num_samples=160, chunk_size=3, seed=0)
    picks = [0, 3, 7, 12, 20]
    cell = 128
    padding = 18
    title_h = 42
    caption_h = 36
    cols = len(picks)
    img_w = cols * cell + (cols + 1) * padding
    img_h = title_h + cell + caption_h + padding * 2 + 28
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "D1：这张桌子上有一只手、两个目标和一块障碍", fill="black", font=title_font)

    for index, sample in enumerate(picks):
        state = data["states"][sample].numpy()
        instruction = int(data["instructions"][sample])
        frame = render_tabletop(state, size=32)
        x = padding + index * (cell + padding)
        y = title_h + padding
        canvas.paste(_upsample(frame, scale=4), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#888888")
        draw.text((x + 4, y + cell + 6), INSTRUCTIONS[instruction], fill="black", font=font)

    draw.text(
        (padding, img_h - 24),
        "白=抓手，红=红色目标，绿=绿色目标，蓝=障碍。指令只有两句，图片是 32×32。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_vla_closedloop(output_path):
    """D1：监督 loss 下降之后，闭环成功率仍然只有 0.188。"""
    torch.manual_seed(0)
    data = make_tabletop_dataset(num_samples=160, chunk_size=3, seed=0)
    state_policy = torch.nn.Sequential(
        torch.nn.Linear(8 + 2, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
        torch.nn.Tanh(),
    )
    instruction_onehot = torch.nn.functional.one_hot(data["instructions"], 2).float()
    state_input = torch.cat((data["states"], instruction_onehot), dim=-1)
    target = data["action_chunks"][:, 0]
    opt = torch.optim.Adam(state_policy.parameters(), lr=3e-3)
    for _ in range(50):
        opt.zero_grad()
        prediction = state_policy(state_input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        opt.step()
    model = TinyVLA(chunk_size=3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        chunks = model(data["images"], data["instructions"], data["states"])
        loss = torch.nn.functional.mse_loss(chunks, data["action_chunks"])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    test_data = make_tabletop_dataset(32, chunk_size=3, seed=17)
    metrics = evaluate_vla(
        model, test_data["states"], test_data["instructions"], max_steps=12
    )
    run = metrics["runs"][0]
    path = run["trajectory"]
    start_state = test_data["states"][0].numpy().copy()
    instruction = int(test_data["instructions"][0])
    target = start_state[2:4] if instruction == 0 else start_state[4:6]

    width, height = 760, 300
    padding = 28
    title_h = 40
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "D1：loss 降了，手却还在桌子上打转", fill="black", font=title_font)

    board = 200
    origin = (padding, title_h + 8)

    def to_xy(point):
        return (
            origin[0] + int(point[0] * (board - 1)),
            origin[1] + int(point[1] * (board - 1)),
        )

    draw.rectangle(
        [origin[0], origin[1], origin[0] + board, origin[1] + board],
        outline="#d1d5db",
        fill="#f8fafc",
    )
    for obj, color, radius in (
        (start_state[2:4], "#dc2626", 7),
        (start_state[4:6], "#16a34a", 7),
        (start_state[6:8], "#2563eb", 10),
    ):
        x, y = to_xy(obj)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    points = [to_xy(p) for p in path]
    if len(points) > 1:
        draw.line(points, fill="#c2410c", width=3)
    for point in points:
        draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill="#c2410c")
    draw.ellipse(
        [points[0][0] - 5, points[0][1] - 5, points[0][0] + 5, points[0][1] + 5],
        outline="black",
        width=2,
    )

    tx = padding + board + 36
    lines = [
        (f"chunk loss  {losses[0]:.3f} → {losses[-1]:.3f}", "#1d4ed8"),
        (f"成功率      {metrics['success_rate']:.3f}", "#c2410c"),
        (f"平均碰撞    {metrics['mean_collisions']:.3f}", "#c2410c"),
        (
            f"距离        {metrics['initial_distance']:.3f} → {metrics['final_distance']:.3f}",
            "#374151",
        ),
        (f"本条指令    {INSTRUCTIONS[instruction]}", "#374151"),
        (f"目标坐标    ({target[0]:.2f}, {target[1]:.2f})", "#6b7280"),
    ]
    for index, (text, color) in enumerate(lines):
        draw.text((tx, title_h + 16 + index * 28), text, fill=color, font=font)
    draw.text(
        (padding, height - 28),
        "左：seed=17 第一条测试轨迹。红线是抓手，红/绿圆是目标，蓝圆是障碍。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_vla_checker(output_path):
    """D2：直达会撞的场景里，四个候选和真实碰撞。"""
    torch.manual_seed(1)
    data = make_outcome_dataset(400, seed=1)
    model = TabletopOutcomeModel()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(80):
        opt.zero_grad()
        loss, _, _ = outcome_loss(
            model, data["states"], data["actions"], data["next_states"], data["collisions"]
        )
        loss.backward()
        opt.step()

    state = torch.tensor([0.20, 0.50, 0.85, 0.50, 0.20, 0.85, 0.31, 0.50])
    candidates = torch.tensor([[1.0, 0.0], [0.7, -0.7], [0.7, 0.7], [-1.0, 0.0]])
    names = ["直达", "斜上", "斜下", "后退"]
    chosen, scores = rerank_actions(model, state, instruction=0, candidates=candidates)
    true_hits = [step_tabletop(state.numpy(), action.numpy())[1] for action in candidates]
    with torch.no_grad():
        _, logits = model(state[None].expand(len(candidates), -1), candidates)
        probs = torch.sigmoid(logits).tolist()

    cell = 220
    padding = 20
    title_h = 40
    img_w = cell + 420 + padding * 3
    img_h = cell + title_h + 78
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "D2：障碍挡在正前方时，模型仍可能选会撞的那一步", fill="black", font=title_font)

    origin = (padding, title_h + 8)
    board = cell
    draw.rectangle(
        [origin[0], origin[1], origin[0] + board, origin[1] + board],
        outline="#d1d5db",
        fill="#111827",
    )
    frame = render_tabletop(state.numpy(), size=32)
    canvas.paste(_upsample(frame, scale=cell // 32), origin)

    def to_xy(point):
        return (
            origin[0] + int(np.clip(point[0], 0, 1) * (board - 1)),
            origin[1] + int(np.clip(point[1], 0, 1) * (board - 1)),
        )

    start = to_xy(state[:2].numpy())
    for action, hit in zip(candidates.numpy(), true_hits):
        nxt, _ = step_tabletop(state.numpy(), action)
        end = to_xy(nxt[:2])
        color = "#ef4444" if hit else "#22c55e"
        draw.line([start, end], fill=color, width=3)
        draw.ellipse([end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4], fill=color)

    tx = origin[0] + board + 24
    draw.text((tx, title_h + 12), "候选          真实碰撞   预测概率   分数", fill="#6b7280", font=small)
    for index, name in enumerate(names):
        mark = "← 选中" if index == chosen else ""
        color = "#c2410c" if true_hits[index] else "#047857"
        text = (
            f"{name:6s}   {str(true_hits[index]):5s}     "
            f"{probs[index]:.3f}      {float(scores[index]):+.3f}  {mark}"
        )
        draw.text((tx, title_h + 40 + index * 28), text, fill=color, font=font)

    draw.text(
        (padding, img_h - 28),
        f"选中候选 {int(chosen)}。这一次真实仍碰撞——单点样例不能当安全证据。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_checker_batch(output_path):
    """D2：64 个必撞场景，直达 1.000，重排 0.328，进展却是负的。"""
    torch.manual_seed(1)
    data = make_outcome_dataset(400, seed=1)
    model = TabletopOutcomeModel()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss, _, _ = outcome_loss(
            model, data["states"], data["actions"], data["next_states"], data["collisions"]
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    safety = evaluate_reranker(model, num_cases=64, seed=23, collision_weight=4.0)

    width, height = 720, 280
    padding = 48
    title_h = 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "D2：碰撞降了，离目标却更远了", fill="black", font=title_font)

    bars = [
        ("直达碰撞", safety["direct_collision_rate"], "#b91c1c"),
        ("重排碰撞", safety["reranked_collision_rate"], "#047857"),
    ]
    base_y = title_h + 36
    bar_h = 28
    max_w = 360
    for index, (name, value, color) in enumerate(bars):
        y = base_y + index * 56
        draw.text((padding, y), name, fill="black", font=font)
        draw.rectangle(
            [padding + 110, y, padding + 110 + int(value * max_w), y + bar_h],
            fill=color,
        )
        draw.text((padding + 120 + int(value * max_w), y + 4), f"{value:.3f}", fill=color, font=font)

    draw.text(
        (padding, height - 52),
        f"outcome loss {losses[0]:.3f} → {losses[-1]:.3f}    "
        f"重排平均进展 {safety['reranked_mean_progress']:+.3f}",
        fill="gray",
        font=small,
    )
    draw.text(
        (padding, height - 28),
        "64 个场景都把障碍放在直达路线上。安全地停住，并不等于会绕行。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def _e1_geometry():
    depth = np.full((6, 6), 4.0, dtype=np.float32)
    depth[2:4, 2:4] = 2.0
    points_camera = depth_to_points(depth, fx=6, fy=6, cx=2.5, cy=2.5)
    points_world = transform_points(points_camera, make_camera_transform(tx=1.0))
    occupancy = points_to_occupancy(points_world, (-2, 4), (0, 6), 0.5)
    wrong_world = transform_points(points_camera, make_camera_transform(tx=1.3))
    wrong_occ = points_to_occupancy(wrong_world, (-2, 4), (0, 6), 0.5)
    shift = points_world.mean(0) - points_camera.mean(0)
    calibration_error = float(np.linalg.norm(wrong_world - points_world, axis=1).mean())
    return depth, points_camera, occupancy, wrong_occ, shift, calibration_error


def visualize_depth_occupancy(output_path):
    """E1：6×6 深度 → 36 个点 → 12×12 Occupancy，占用 8 格。"""
    depth, points, occupancy, _, shift, _ = _e1_geometry()
    cell = 160
    padding = 22
    title_h = 40
    cols = 3
    img_w = cols * cell + (cols + 1) * padding
    img_h = title_h + cell + 78
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "E1：深度像素怎样变成俯视占用", fill="black", font=title_font)

    depth_vis = ((4.0 - depth) / 2.0 * 220 + 30).astype(np.uint8)
    depth_rgb = np.stack([depth_vis, depth_vis, np.full_like(depth_vis, 40)], axis=-1)
    canvas.paste(_upsample(depth_rgb, scale=cell // 6), (padding, title_h + 8))
    draw.text((padding, title_h + cell + 16), "深度 6×6", fill="black", font=font)
    draw.text((padding, title_h + cell + 36), "中间 2×2 是 2 m", fill="gray", font=small)

    cloud = Image.new("RGB", (cell, cell), "#f8fafc")
    cd = ImageDraw.Draw(cloud)
    xs, ys = points[:, 0], points[:, 1]
    for x, y, depth_z in zip(xs, ys, points[:, 2]):
        px = int((x - xs.min()) / (xs.max() - xs.min() + 1e-6) * (cell - 20) + 10)
        py = int((y - ys.min()) / (ys.max() - ys.min() + 1e-6) * (cell - 20) + 10)
        radius = 7 if depth_z < 3 else 4
        color = "#1d4ed8" if depth_z < 3 else "#93c5fd"
        cd.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)
    canvas.paste(cloud, (padding * 2 + cell, title_h + 8))
    draw.text((padding * 2 + cell, title_h + cell + 16), "相机坐标 36 点", fill="black", font=font)
    draw.text((padding * 2 + cell, title_h + cell + 36), "大点 z=2，小点 z=4", fill="gray", font=small)

    occ_rgb = _grid_to_rgb(occupancy)
    canvas.paste(_upsample(occ_rgb, scale=cell // 12), (padding * 3 + cell * 2, title_h + 8))
    draw.text(
        (padding * 3 + cell * 2, title_h + cell + 16),
        f"Occupancy {int(occupancy.sum())}/144",
        fill="black",
        font=font,
    )
    draw.text(
        (padding * 3 + cell * 2, title_h + cell + 36),
        f"平移 {shift[0]:.1f} m 后落入格子",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_calibration(output_path):
    """E1：把平移写成 1.3 m，点云整体偏 0.3 m，占用错 5 格。"""
    _, _, occupancy, wrong_occ, _, error = _e1_geometry()
    cell = 180
    padding = 24
    title_h = 40
    img_w = cell * 3 + padding * 4
    img_h = title_h + cell + 72
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "E1：外参写错 0.3 米，占用格子就对不齐", fill="black", font=title_font)

    xor = (occupancy != wrong_occ).astype(np.uint8)
    panels = [
        (occupancy, "tx = 1.0 m", f"占用 {int(occupancy.sum())} 格", (220, 70, 50)),
        (wrong_occ, "tx = 1.3 m", f"占用 {int(wrong_occ.sum())} 格", (37, 99, 235)),
        (xor, "对不齐的格子", f"相差 {int(xor.sum())} 格", (180, 83, 9)),
    ]
    for index, (grid, title, note, color) in enumerate(panels):
        x = padding + index * (cell + padding)
        y = title_h + 8
        canvas.paste(_upsample(_grid_to_rgb(grid, occupied=color), scale=cell // 12), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#d1d5db")
        draw.text((x, y + cell + 8), title, fill="black", font=font)
        draw.text((x, y + cell + 28), note, fill="gray", font=small)
    draw.text(
        (padding, img_h - 22),
        f"每个点的平均位移 {error:.3f} m。网络可以记住固定偏差，却修不好错误几何。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_4d_field(output_path):
    """E2a：同一坐标只换时间和动作，密度会变。"""
    torch.manual_seed(0)
    coordinates, density, color = make_colored_sphere_samples(640, seed=0)
    field = TinyNeuralField()
    opt = torch.optim.Adam(field.parameters(), lr=5e-3)
    static_losses = []
    for _ in range(80):
        opt.zero_grad()
        pred_d, pred_c = field(coordinates)
        loss = torch.nn.functional.mse_loss(pred_d, density) + torch.nn.functional.mse_loss(
            pred_c, color
        )
        loss.backward()
        opt.step()
        static_losses.append(float(loss.detach()))

    coordinates_4d, times, actions, density_4d, color_4d = make_moving_sphere_samples(1024, seed=1)
    dynamic = TinyDynamicField()
    opt = torch.optim.Adam(dynamic.parameters(), lr=5e-3)
    dynamic_losses = []
    for _ in range(100):
        pred_d, pred_c = dynamic(coordinates_4d, times, actions)
        loss = torch.nn.functional.binary_cross_entropy(
            pred_d, density_4d
        ) + torch.nn.functional.mse_loss(pred_c, color_4d)
        opt.zero_grad()
        loss.backward()
        opt.step()
        dynamic_losses.append(float(loss.detach()))

    query = torch.tensor([[0.35, 0.0, 0.0]]).expand(5, -1)
    names = ["停", "上", "下", "左", "右"]
    with torch.no_grad():
        early, _ = dynamic(query, torch.zeros(5), torch.arange(5))
        late, _ = dynamic(query, torch.ones(5), torch.arange(5))
    early = [float(x) for x in early]
    late = [float(x) for x in late]

    width, height = 760, 300
    padding = 36
    title_h = 40
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "E2a：固定 (0.35, 0, 0)，只改时间和动作", fill="black", font=title_font)

    bar_w = 22
    gap = 18
    base_x = padding + 40
    base_y = title_h + 170
    max_h = 120
    ymax = max(early + late + [0.2])
    for index, name in enumerate(names):
        x = base_x + index * (bar_w * 2 + gap)
        h0 = int(early[index] / ymax * max_h)
        h1 = int(late[index] / ymax * max_h)
        draw.rectangle([x, base_y - h0, x + bar_w - 2, base_y], fill="#93c5fd")
        draw.rectangle([x + bar_w, base_y - h1, x + 2 * bar_w - 2, base_y], fill="#1d4ed8")
        draw.text((x, base_y + 8), name, fill="black", font=small)
        draw.text((x - 2, base_y - h0 - 16), f"{early[index]:.2f}", fill="#6b7280", font=small)

    draw.text((padding + 430, title_h + 24), "浅= t=0，深= t=1", fill="#1d4ed8", font=font)
    draw.text(
        (padding + 430, title_h + 56),
        f"静态场  {static_losses[0]:.3f} → {static_losses[-1]:.3f}",
        fill="#374151",
        font=font,
    )
    draw.text(
        (padding + 430, title_h + 84),
        f"动态场  {dynamic_losses[0]:.3f} → {dynamic_losses[-1]:.3f}",
        fill="#374151",
        font=font,
    )
    draw.text(
        (padding + 430, title_h + 112),
        f"t 差异均值  {abs(np.mean(np.array(early) - np.array(late))):.4f}".replace(
            "t 差异均值", "t 差异均值"
        ),
        fill="#374151",
        font=font,
    )
    draw.text(
        (padding, height - 28),
        "这是坐标查询，不是多视角重建。密度变了，只说明时间和动作进了接口。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_occupancy_future(output_path):
    """E2b：过去三帧加动作，预测未来三帧；horizon 越远 IoU 越低。"""
    torch.manual_seed(1)
    history, actions, future = make_moving_occupancy_dataset(96, seed=1)
    model = TinyOccupancyPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    positive_weight = torch.tensor(18.0)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        logits = model(history, actions)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, future, pos_weight=positive_weight
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        logits = model(history, actions)
        pred = (torch.sigmoid(logits) > 0.5).float()
        ious = [float(occupancy_iou(logits[:, t], future[:, t])) for t in range(3)]
        total_iou = float(occupancy_iou(logits, future))
        same_history = history[:1].expand(5, -1, -1, -1)
        counterfactual = torch.sigmoid(model(same_history, torch.arange(5)))
        diffs = [
            float((counterfactual[0] - counterfactual[i]).abs().mean()) for i in range(1, 5)
        ]

    sample = 0
    names = ["stay", "up", "down", "left", "right"]
    cell = 72
    padding = 16
    title_h = 40
    rows = 3
    cols = 3
    label_w = 92
    img_w = label_w + cols * cell + (cols + 1) * padding + 240
    img_h = title_h + rows * (cell + 18) + 64
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text(
        (padding, 8),
        f"E2b：动作 {names[int(actions[sample])]} 条件下的未来占用",
        fill="black",
        font=title_font,
    )

    row_specs = [
        ("过去 3 帧", history[sample].numpy(), "#2563eb"),
        ("真实未来", future[sample].numpy(), "#16a34a"),
        ("预测占用", pred[sample].numpy(), "#c2410c"),
    ]
    for row, (name, frames, _color) in enumerate(row_specs):
        y = title_h + row * (cell + 18)
        draw.text((padding, y + cell // 2 - 8), name, fill="black", font=font)
        for col in range(3):
            x = label_w + padding + col * (cell + 8)
            canvas.paste(_upsample(_grid_to_rgb(frames[col]), scale=cell // 16), (x, y))
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#d1d5db")
            if row == 0:
                draw.text((x + 4, y + cell + 2), f"t-{2 - col}", fill="gray", font=small)

    tx = label_w + cols * (cell + 8) + 36
    draw.text((tx, title_h + 8), f"loss  {losses[0]:.3f} → {losses[-1]:.3f}", fill="#1d4ed8", font=font)
    draw.text((tx, title_h + 36), f"总 IoU  {total_iou:.3f}", fill="#047857", font=font)
    for index, value in enumerate(ious):
        draw.text((tx, title_h + 64 + index * 24), f"horizon {index + 1}  {value:.3f}", fill="#374151", font=font)
    draw.text((tx, title_h + 148), "换动作后的差异", fill="#6b7280", font=small)
    draw.text((tx, title_h + 168), str([round(x, 3) for x in diffs]), fill="#374151", font=small)
    draw.text(
        (padding, img_h - 24),
        "空格子远多于占用，所以训练加了 pos_weight=18。IoU 是离线占用，不是闭环驾驶。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_spatial_world(output_path):
    """路线 E 总览：同一小方块从历史走到动作条件的未来。"""
    histories, actions, futures = make_moving_occupancy_dataset(
        num_samples=64, size=16, past=3, future=3, seed=42
    )
    names = ["stay", "up", "down", "left", "right"]
    cell = 140
    padding = 20
    title_h = 40
    cols = 4
    img_w = cols * cell + (cols + 1) * padding
    img_h = title_h + cell + 72
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "E：先把空间算对，再问未来占用会去哪", fill="black", font=title_font)

    panels = [
        (histories[0, 0].numpy(), "过去第 1 帧"),
        (histories[0, -1].numpy(), "过去第 3 帧"),
        (futures[0, 0].numpy(), "未来第 1 帧"),
        (futures[0, -1].numpy(), "未来第 3 帧"),
    ]
    for index, (grid, title) in enumerate(panels):
        x = padding + index * (cell + padding)
        y = title_h + 8
        canvas.paste(_upsample(_grid_to_rgb(grid), scale=cell // 16), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#d1d5db")
        draw.text((x, y + cell + 8), title, fill="black", font=font)
    draw.text(
        (padding, img_h - 24),
        f"动作 = {names[int(actions[0])]}。方块按动作平移，占用跟着走——这就是后面要预测的东西。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "docs" / "public" / "carracing"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_tabletop(output_dir / "de-tabletop.png")
    visualize_vla_closedloop(output_dir / "de-vla-closedloop.png")
    visualize_vla_checker(output_dir / "de-vla-checker.png")
    visualize_checker_batch(output_dir / "de-checker-batch.png")
    visualize_depth_occupancy(output_dir / "de-depth-occupancy.png")
    visualize_calibration(output_dir / "de-calibration.png")
    visualize_4d_field(output_dir / "de-4d-query.png")
    visualize_occupancy_future(output_dir / "de-occupancy-future.png")
    visualize_spatial_world(output_dir / "de-spatial-world.png")

    print("\nRoute D/E visualizations generated!")
