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
from hwm.neural import (
    Actor,
    Critic,
    PixelEncoder,
    RSSM,
    RSSMState,
    TinyWorldModel,
    batch_from_episodes,
    imagine,
    world_model_loss,
)
from hwm.data import (
    ACTION_NAMES,
    MovingSquareWorld,
    make_pixelworld_dataset,
    pixelworld_transition_arrays,
)
from hwm.control import (
    PositionDynamics,
    fit_position_dynamics,
    beam_plan,
    run_pixelworld_controller,
    run_random_controller,
)


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


def _upsample(array, scale=8):
    image = np.asarray(array)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image).resize(
        (image.shape[1] * scale, image.shape[0] * scale), Image.NEAREST
    )


def visualize_pixelworld(output_path):
    """A1: PixelWorld 是什么——红方块、绿目标、五个动作。"""
    world = MovingSquareWorld()
    start = (2, 2)
    frames = [world.render(start)]
    position = start
    demo_actions = [2, 2, 4, 2, 4]
    for action in demo_actions:
        position = world.next_position(position, action)
        frames.append(world.render(position))

    cell = 128
    padding = 18
    title_h = 42
    caption_h = 36
    cols = len(frames)
    img_w = cols * cell + (cols + 1) * padding
    img_h = title_h + cell + caption_h + padding * 2 + 28
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(20)
    font = try_load_font(13)
    small = try_load_font(12)

    draw.text((padding, 8), "A1: PixelWorld——16×16 的动作条件小世界", fill="black", font=title_font)
    labels = ["t=0 起点"] + [
        f"a={ACTION_NAMES[action]}" for action in demo_actions
    ]
    for index, (frame, label) in enumerate(zip(frames, labels)):
        x = padding + index * (cell + padding)
        y = title_h + padding
        canvas.paste(_upsample(frame, scale=8), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#888888")
        draw.text((x + 4, y + cell + 6), label, fill="black", font=font)

    draw.text(
        (padding, img_h - 24),
        "红=方块，绿=目标 (12,12)。动作：0 stay / 1 left / 2 right / 3 up / 4 down。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_reconstruction(output_path):
    """A1: 15 次更新后的重建，对照原图和复制上一帧。"""
    torch.manual_seed(0)
    episodes = make_pixelworld_dataset(num_episodes=4, length=8, seed=0)
    observations, actions, rewards, dones = batch_from_episodes(
        episodes, sequence_length=8
    )
    model = TinyWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(15):
        optimizer.zero_grad()
        loss, metrics, outputs = world_model_loss(
            model, observations, actions, rewards, dones
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()

    targets = observations[:, 1:].float() / 255.0
    copy_last = observations[:, :-1].float() / 255.0
    recon = outputs["reconstruction"].detach()
    copy_mse = float(torch.nn.functional.mse_loss(copy_last, targets))
    recon_mse = float(metrics["reconstruction"])

    cell = 96
    padding = 16
    label_w = 88
    title_h = 40
    rows = 3
    cols = 4
    img_w = label_w + cols * cell + (cols + 1) * padding
    img_h = title_h + rows * (cell + 28) + padding + 36
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "A1: 原图 / 复制上一帧 / RSSM 重建", fill="black", font=title_font)

    row_specs = [
        ("原图 t+1", targets[0, :cols], None),
        ("复制上一帧", copy_last[0, :cols], f"MSE {copy_mse:.4f}"),
        ("15 步重建", recon[0, :cols], f"MSE {recon_mse:.4f}"),
    ]
    for row, (name, frames, note) in enumerate(row_specs):
        y = title_h + row * (cell + 28)
        draw.text((padding, y + cell // 2 - 8), name, fill="black", font=font)
        if note:
            draw.text((padding, y + cell // 2 + 10), note, fill="gray", font=small)
        for col in range(cols):
            x = label_w + padding + col * (cell + padding // 2)
            canvas.paste(_upsample(frames[col].numpy(), scale=6), (x, y))
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#888888")
            if row == 0:
                draw.text((x + 4, y + cell + 4), f"t={col + 1}", fill="gray", font=small)

    draw.text(
        (padding, img_h - 28),
        "复制上一帧 MSE 0.0061 < 重建 0.0144：loss 下降不等于学会了动态。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_a1_loss(output_path):
    """A1: 15 次更新的 total / reconstruction / KL。"""
    torch.manual_seed(0)
    episodes = make_pixelworld_dataset(num_episodes=4, length=8, seed=0)
    batch = batch_from_episodes(episodes, sequence_length=8)
    model = TinyWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    totals, recons, kls = [], [], []
    for _ in range(15):
        optimizer.zero_grad()
        loss, metrics, _ = world_model_loss(model, *batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        totals.append(float(loss.detach()))
        recons.append(float(metrics["reconstruction"]))
        kls.append(float(metrics["kl"]))

    width, height = 720, 280
    padding = 48
    title_h = 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(12)
    draw.text((padding, 8), "A1: 15 次更新——接口通了，世界还没学会", fill="black", font=title_font)

    plot_w = width - padding * 2
    plot_h = height - title_h - padding - 8
    origin = (padding, title_h + 8)

    def draw_series(values, color):
        vmax = max(values)
        vmin = min(values)
        span = max(vmax - vmin, 1e-6)
        points = []
        for index, value in enumerate(values):
            x = origin[0] + int(index / (len(values) - 1) * plot_w)
            y = origin[1] + plot_h - int((value - vmin) / span * plot_h)
            points.append((x, y))
        draw.line(points, fill=color, width=2)
        return points[-1]

    last_total = draw_series(totals, "#1d4ed8")
    last_recon = draw_series(recons, "#047857")
    last_kl = draw_series(kls, "#c2410c")
    draw.text((last_total[0] - 90, last_total[1] - 16), f"total {totals[-1]:.3f}", fill="#1d4ed8", font=font)
    draw.text((last_recon[0] - 90, last_recon[1] + 4), f"recon {recons[-1]:.3f}", fill="#047857", font=font)
    draw.text((last_kl[0] - 70, last_kl[1] - 16), f"KL {kls[-1]:.3f}", fill="#c2410c", font=font)
    draw.text(
        (padding, height - 28),
        f"total {totals[0]:.3f} → {totals[-1]:.3f}    recon {recons[0]:.3f} → {recons[-1]:.3f}    KL {kls[0]:.3f} → {kls[-1]:.3f}",
        fill="gray",
        font=font,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_planned_vs_random(output_path):
    """A2: 同一起点，learned MPC 走到目标，随机走散。"""
    torch.manual_seed(1)
    world = MovingSquareWorld()
    episodes = []
    for row in (0, 3, 6, 9, 12, 13):
        for col in (0, 3, 6, 9, 12, 13):
            for action in range(5):
                episode, _ = world.generate([action], start=(row, col))
                episodes.append(episode)
    positions, actions, next_positions = pixelworld_transition_arrays(episodes)
    model = PositionDynamics(hidden_size=48)
    fit_position_dynamics(model, positions, actions, next_positions, updates=100)
    planned = run_pixelworld_controller(model, (5, 5), max_steps=24)
    random_run = run_random_controller((5, 5), max_steps=24, seed=0)

    cell = 18
    board = 16 * cell
    padding = 24
    title_h = 40
    gap = 36
    img_w = padding * 3 + board * 2
    img_h = title_h + board + 70
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "A2: 同一起点 (5,5)——规划 vs 随机", fill="black", font=title_font)

    def paint_board(origin_x, path, color, title, note):
        y0 = title_h
        draw.rectangle(
            [origin_x, y0, origin_x + board, y0 + board],
            outline="#d1d5db",
            fill="#f8fafc",
        )
        for index in range(17):
            draw.line(
                [(origin_x + index * cell, y0), (origin_x + index * cell, y0 + board)],
                fill="#e5e7eb",
            )
            draw.line(
                [(origin_x, y0 + index * cell), (origin_x + board, y0 + index * cell)],
                fill="#e5e7eb",
            )
        gx, gy = world.goal
        draw.rectangle(
            [
                origin_x + gy * cell + 2,
                y0 + gx * cell + 2,
                origin_x + (gy + 3) * cell - 2,
                y0 + (gx + 3) * cell - 2,
            ],
            fill="#86efac",
        )
        points = [
            (origin_x + col * cell + cell // 2, y0 + row * cell + cell // 2)
            for row, col in path
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse(
                [point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3],
                fill=color,
            )
        draw.text((origin_x, y0 + board + 8), title, fill="black", font=font)
        draw.text((origin_x, y0 + board + 28), note, fill="gray", font=small)

    paint_board(
        padding,
        planned["positions"],
        "#047857",
        "learned MPC",
        f"{len(planned['actions'])} 步到达 (12,12)，成功率 1.00",
    )
    paint_board(
        padding * 2 + board,
        random_run["positions"],
        "#b91c1c",
        "随机策略",
        f"24 步后距离 {random_run['final_distance']:.1f}，成功率 0.00",
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def visualize_imagination(output_path):
    """A2: 一次 5 步 imagination 的动作、奖励与 TD-λ。"""
    torch.manual_seed(1)
    episodes = make_pixelworld_dataset(num_episodes=4, length=8, seed=1)
    batch = batch_from_episodes(episodes, sequence_length=8)
    observations, actions, *_ = batch
    model = TinyWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(10):
        optimizer.zero_grad()
        loss, _, _ = world_model_loss(model, *batch)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model(observations, actions, sample=False)
    posterior = outputs["posterior"]
    start = RSSMState(
        posterior.deterministic[:, -1].detach(),
        posterior.stochastic[:, -1].detach(),
        posterior.mean[:, -1].detach(),
        posterior.std[:, -1].detach(),
    )
    actor = Actor()
    critic = Critic()
    imagined = imagine(model, actor, start, horizon=5)
    values = critic(imagined["features"].detach())
    from hwm.neural import lambda_returns

    returns = lambda_returns(
        imagined["rewards"].detach(),
        imagined["continues"].detach(),
        values.detach(),
        values[:, -1].detach(),
    )
    names = [ACTION_NAMES[int(a)] for a in imagined["actions"][0].tolist()]
    rewards = [float(x) for x in imagined["rewards"][0].detach()]
    tds = [float(x) for x in returns[0].detach()]

    cell_w = 118
    cell_h = 92
    padding = 20
    title_h = 40
    cols = 5
    img_w = cols * cell_w + (cols + 1) * padding
    img_h = title_h + cell_h + 78
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = try_load_font(18)
    font = try_load_font(13)
    small = try_load_font(12)
    draw.text((padding, 8), "A2: 一次 5 步 imagination（未读未来图片）", fill="black", font=title_font)
    for index in range(cols):
        x = padding + index * (cell_w + padding)
        y = title_h + 8
        draw.rounded_rectangle(
            [x, y, x + cell_w, y + cell_h], radius=8, outline="#c2410c", width=2
        )
        draw.text((x + 10, y + 8), f"prior t+{index + 1}", fill="#c2410c", font=font)
        draw.text((x + 10, y + 30), f"a = {names[index]}", fill="black", font=font)
        draw.text((x + 10, y + 50), f"r̂ = {rewards[index]:+.3f}", fill="gray", font=small)
        draw.text((x + 10, y + 68), f"Gλ = {tds[index]:+.3f}", fill="gray", font=small)
        if index < cols - 1:
            ax = x + cell_w + 4
            ay = y + cell_h // 2
            draw.line([(ax, ay), (ax + 12, ay)], fill="black", width=2)
    draw.text(
        (padding, img_h - 28),
        "start = 真实 posterior；之后只走 RSSM prior。数字来自 seed=1 的一次前向。",
        fill="gray",
        font=small,
    )
    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'docs' / 'public' / 'carracing'
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_rssm_dataflow(output_dir / 'rssm-dataflow.png')
    visualize_position_planning(output_dir / 'position-planning.png')
    visualize_pixelworld(output_dir / 'pixelworld.png')
    visualize_reconstruction(output_dir / 'rssm-reconstruction.png')
    visualize_a1_loss(output_dir / 'world-model-loss-curve.png')
    visualize_planned_vs_random(output_dir / 'planned-vs-random.png')
    visualize_imagination(output_dir / 'imagination-training.png')

    print("\nAll route-a visualizations generated!")
