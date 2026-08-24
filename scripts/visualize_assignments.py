#!/usr/bin/env python3
"""
Assignments 可视化脚本：为 pa0, pa1-a, pa1-b, pa1-c, pa2 生成真实可视化。
所有图片通过运行 hwm 模块的实际代码生成。
"""

import sys
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from hwm.data import MovingSquareWorld, make_pixelworld_dataset
from hwm.neural import PixelEncoder, RSSM, RSSMState, TinyWorldModel, Actor, Critic, imagine, batch_from_episodes, world_model_loss, lambda_returns
from hwm.control import PositionDynamics, fit_position_dynamics, beam_plan


def try_load_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size)
    except Exception:
        return ImageFont.load_default()


def draw_loss_curve(draw, x0, y0, w, h, losses, color='blue', label='', title=''):
    if not losses:
        return
    max_l, min_l = max(losses), min(losses)
    rng = max_l - min_l + 1e-8
    for i in range(len(losses) - 1):
        x1 = x0 + int(i / len(losses) * w)
        y1 = y0 + int((1 - (losses[i] - min_l) / rng) * h)
        x2 = x0 + int((i + 1) / len(losses) * w)
        y2 = y0 + int((1 - (losses[i + 1] - min_l) / rng) * h)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
    if title:
        draw.text((x0 + 5, y0 - 18), title, fill='black', font=try_load_font(13))
    if label:
        draw.text((x0 + 5, y0 + h + 4), label, fill='gray', font=try_load_font(11))


# ──── PA0: 世界模型循环 ────
def visualize_pa0_loop(output_path):
    from hwm.gridworld import GridWorld, EmpiricalDynamics, lookahead, mpc_episode, ACTIONS

    world = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                      walls=((2, 2),), traps=((1, 3),))
    rng = random.Random(42)
    action_names = list(ACTIONS.keys())

    all_transitions = []
    trajectories = []
    for _ in range(15):
        state = world.start
        traj = [state]
        for _ in range(12):
            action = rng.choice(action_names)
            trans = world.step(state, action, rng)
            all_transitions.append(trans)
            traj.append(trans.next_state)
            state = trans.next_state
            if trans.done:
                break
        trajectories.append(traj)

    model = EmpiricalDynamics()
    model.fit(all_transitions)

    mpc_transitions, mpc_plans = mpc_episode(world, model, depth=3, max_steps=15, seed=42)
    mpc_path = [world.start] + [t.next_state for t in mpc_transitions]

    # 可视化
    cell = 150
    pad = 20
    title_h = 40
    n_panels = 4
    img_w = n_panels * cell + (n_panels + 1) * pad
    img_h = cell + title_h + pad * 2 + 50

    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(13)
    title_font = try_load_font(18)
    draw.text((pad, 5), "PA0: 世界模型学习循环——数据 → 学习 → 规划 → 执行", fill='black', font=title_font)

    grid_size = 5

    # Panel 1: 收集轨迹
    p1 = Image.new('RGB', (cell, cell), 'white')
    d1 = ImageDraw.Draw(p1)
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for ep_idx in range(min(5, len(trajectories))):
        pts = trajectories[ep_idx]
        for t in range(len(pts) - 1):
            r1, c1 = pts[t]
            r2, c2 = pts[t + 1]
            x1, y1 = int(c1 / grid_size * cell), int(r1 / grid_size * cell)
            x2, y2 = int(c2 / grid_size * cell), int(r2 / grid_size * cell)
            d1.line([(x1, y1), (x2, y2)], fill=colors[ep_idx % 5], width=2)
    canvas.paste(p1, (pad, title_h + pad))
    draw.text((pad + 5, title_h + cell + pad + 5), "收集轨迹", fill='black', font=font)
    draw.text((pad + 5, title_h + cell + pad + 22), f"{len(all_transitions)} transitions", fill='gray', font=try_load_font(11))

    # Panel 2: 学习动态
    p2 = Image.new('RGB', (cell, cell), 'white')
    d2 = ImageDraw.Draw(p2)
    covered = sum(1 for k in model.counts if model.counts[k])
    total = max(len(model.counts), 1)
    d2.text((10, 20), f"状态-动作对: {total}", fill='black', font=font)
    d2.text((10, 45), f"已覆盖: {covered}", fill='green', font=font)
    d2.text((10, 70), f"未覆盖: {total - covered}", fill='red', font=font)
    coverage = covered / total * 100
    d2.text((10, 100), f"覆盖率: {coverage:.0f}%", fill='black', font=try_load_font(15))
    canvas.paste(p2, (cell + 2 * pad, title_h + pad))
    draw.text((cell + 2 * pad + 5, title_h + cell + pad + 5), "学习动态", fill='black', font=font)

    # Panel 3: MPC 路径
    p3 = Image.new('RGB', (cell, cell), 'white')
    d3 = ImageDraw.Draw(p3)
    for i in range(grid_size + 1):
        d3.line([(int(i / grid_size * cell), 0), (int(i / grid_size * cell), cell)], fill='#eee', width=1)
        d3.line([(0, int(i / grid_size * cell)), (cell, int(i / grid_size * cell))], fill='#eee', width=1)
    for wr, wc in world.walls:
        x, y = int(wc / grid_size * cell), int(wr / grid_size * cell)
        d3.rectangle([x, y, x + int(cell / grid_size), y + int(cell / grid_size)], fill='#333')
    for tr, tc in world.traps:
        x, y = int(tc / grid_size * cell), int(tr / grid_size * cell)
        d3.rectangle([x, y, x + int(cell / grid_size), y + int(cell / grid_size)], fill='red')
    for i in range(len(mpc_path) - 1):
        r1, c1 = mpc_path[i]
        r2, c2 = mpc_path[i + 1]
        x1, y1 = int((c1 + 0.5) / grid_size * cell), int((r1 + 0.5) / grid_size * cell)
        x2, y2 = int((c2 + 0.5) / grid_size * cell), int((r2 + 0.5) / grid_size * cell)
        d3.line([(x1, y1), (x2, y2)], fill='blue', width=3)
        d3.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], fill='blue')
    gr, gc = world.goal
    gx, gy = int((gc + 0.5) / grid_size * cell), int((gr + 0.5) / grid_size * cell)
    d3.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline='green', width=2)
    canvas.paste(p3, (2 * (cell + pad) + pad, title_h + pad))
    draw.text((2 * (cell + pad) + pad + 5, title_h + cell + pad + 5), "MPC 路径", fill='black', font=font)
    reached = any(t.next_state == world.goal for t in mpc_transitions)
    draw.text((2 * (cell + pad) + pad + 5, title_h + cell + pad + 22),
              f"步数: {len(mpc_transitions)}, 到达: {'Y' if reached else 'N'}", fill='gray', font=try_load_font(11))

    # Panel 4: 执行与修正
    p4 = Image.new('RGB', (cell, cell), 'white')
    d4 = ImageDraw.Draw(p4)
    d4.text((10, 15), "MPC 闭环执行:", fill='black', font=font)
    d4.text((10, 40), f"规划步数: {len(mpc_transitions)}", fill='blue', font=font)
    d4.text((10, 65), f"到达目标: {'Yes' if reached else 'No'}", fill='green' if reached else 'red', font=font)
    total_reward = sum(t.reward for t in mpc_transitions)
    d4.text((10, 90), f"总回报: {total_reward:.1f}", fill='black', font=font)
    d4.text((10, 115), "预测 vs 真实:", fill='black', font=font)
    d4.text((10, 135), "偏差 = 修正信号", fill='orange', font=font)
    canvas.paste(p4, (3 * (cell + pad) + pad, title_h + pad))
    draw.text((3 * (cell + pad) + pad + 5, title_h + cell + pad + 5), "执行与修正", fill='black', font=font)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


# ──── PA1-A: Dreamer-lite 训练循环 ────
def visualize_pa1a_cycle(output_path):
    episodes = make_pixelworld_dataset(num_episodes=8, length=10, seed=42)
    observations, actions, rewards, dones = batch_from_episodes(episodes, sequence_length=8)

    model = TinyWorldModel(action_size=5, embed_size=64, deter_size=64, stoch_size=16)
    actor = Actor(feature_size=80, action_size=5)
    critic = Critic(feature_size=80)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rssm_losses = []
    for step in range(20):
        optimizer.zero_grad()
        total, metrics, outputs = world_model_loss(model, observations, actions, rewards, dones)
        total.backward()
        optimizer.step()
        rssm_losses.append(float(metrics['reconstruction']))

    # imagination (需要梯度给 Actor)
    B = observations.shape[0]
    embeds = model.encoder(observations[:, 1:].reshape(-1, 16, 16, 3)).reshape(B, -1, 64)
    with torch.no_grad():
        priors, posteriors = model.rssm.observe(embeds, actions)
        start_state = RSSMState(
            posteriors.deterministic[:, -1],
            posteriors.stochastic[:, -1],
            posteriors.mean[:, -1],
            posteriors.std[:, -1],
        )
    # model 冻结但 actor 需要梯度
    for p in model.parameters():
        p.requires_grad_(False)
    imag = imagine(model, actor, start_state, horizon=5)
    for p in model.parameters():
        p.requires_grad_(True)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-3)
    values = critic(imag['features'])
    returns_target = lambda_returns(imag['rewards'], imag['continues'], values.detach(), values[:, -1].detach())
    actor_loss = -(imag['log_probs'] * returns_target.detach()).mean()
    critic_loss = torch.nn.functional.mse_loss(values, returns_target.detach())
    opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()
    opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()

    # 可视化
    cell_w, cell_h, pad, title_h = 160, 130, 20, 40
    n_cols = 4
    img_w = n_cols * cell_w + (n_cols + 1) * pad
    img_h = cell_h + title_h + pad * 2 + 60

    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(13)
    draw.text((pad, 5), "PA1-A: Dreamer-lite 训练循环——真实数据流", fill='black', font=try_load_font(18))

    # P1: 数据
    p1 = Image.new('RGB', (cell_w, cell_h), 'white')
    d1 = ImageDraw.Draw(p1)
    for i in range(4):
        frame = episodes[i].observations[0]
        img = Image.fromarray(frame.astype(np.uint8)).resize((28, 28))
        p1.paste(img, (i * 32 + 5, 10))
    d1.text((5, 45), f"8 eps x 10 steps", fill='black', font=font)
    d1.text((5, 65), f"obs: {list(observations.shape)}", fill='gray', font=try_load_font(11))
    d1.text((5, 82), f"act: {list(actions.shape)}", fill='gray', font=try_load_font(11))
    canvas.paste(p1, (pad, title_h + pad))
    draw.text((pad + 5, title_h + cell_h + pad + 5), "数据收集", fill='black', font=font)

    # P2: RSSM loss
    p2 = Image.new('RGB', (cell_w, cell_h), 'white')
    d2 = ImageDraw.Draw(p2)
    draw_loss_curve(d2, 5, 5, cell_w - 10, cell_h - 30, rssm_losses, 'blue',
                    label=f"{rssm_losses[0]:.3f} -> {rssm_losses[-1]:.3f}")
    canvas.paste(p2, (cell_w + 2 * pad, title_h + pad))
    draw.text((cell_w + 2 * pad + 5, title_h + cell_h + pad + 5), "RSSM 训练", fill='black', font=font)

    # P3: Imagination
    p3 = Image.new('RGB', (cell_w, cell_h), 'white')
    d3 = ImageDraw.Draw(p3)
    d3.text((5, 10), "Imagination:", fill='black', font=font)
    d3.text((5, 30), f"feat: {list(imag['features'].shape)}", fill='gray', font=try_load_font(11))
    d3.text((5, 48), f"rew: {list(imag['rewards'].shape)}", fill='gray', font=try_load_font(11))
    d3.text((5, 66), f"horizon: 5", fill='gray', font=try_load_font(11))
    d3.text((5, 84), f"start: posterior", fill='green', font=try_load_font(11))
    d3.text((5, 102), f"rollout: prior", fill='orange', font=try_load_font(11))
    canvas.paste(p3, (2 * (cell_w + pad) + pad, title_h + pad))
    draw.text((2 * (cell_w + pad) + pad + 5, title_h + cell_h + pad + 5), "Imagination", fill='black', font=font)

    # P4: Actor-Critic
    p4 = Image.new('RGB', (cell_w, cell_h), 'white')
    d4 = ImageDraw.Draw(p4)
    d4.text((5, 10), "Actor-Critic:", fill='black', font=font)
    d4.text((5, 30), f"actor: {float(actor_loss.item()):.4f}", fill='blue', font=try_load_font(11))
    d4.text((5, 48), f"critic: {float(critic_loss.item()):.4f}", fill='red', font=try_load_font(11))
    d4.text((5, 66), f"ret mean: {float(returns_target.mean()):.4f}", fill='gray', font=try_load_font(11))
    d4.text((5, 84), f"ret std: {float(returns_target.std()):.4f}", fill='gray', font=try_load_font(11))
    d4.text((5, 102), f"gamma=0.99, lambda=0.95", fill='gray', font=try_load_font(11))
    canvas.paste(p4, (3 * (cell_w + pad) + pad, title_h + pad))
    draw.text((3 * (cell_w + pad) + pad + 5, title_h + cell_h + pad + 5), "Actor-Critic", fill='black', font=font)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


# ──── PA1-B: 可控制视频 ────
def visualize_pa1b_controllable(output_path):
    from hwm.video import TinyVQVAE

    episodes = make_pixelworld_dataset(num_episodes=8, length=10, seed=42)
    vqvae = TinyVQVAE(codebook_size=16, embedding_size=8)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=1e-3)

    frames = []
    for ep in episodes:
        for obs in ep.observations:
            frames.append(obs)
    frames_tensor = torch.tensor(np.array(frames[:64]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    recon_losses = []
    for step in range(30):
        optimizer.zero_grad()
        result = vqvae(frames_tensor)
        loss = result['loss']
        loss.backward()
        optimizer.step()
        recon_losses.append(float(result['reconstruction_loss']))

    with torch.no_grad():
        result = vqvae(frames_tensor[:8])
        originals = frames_tensor[:8].permute(0, 2, 3, 1).numpy()
        recons = result['reconstruction'].permute(0, 2, 3, 1).numpy()
    tokens = result['tokens']
    unique_tokens = len(torch.unique(tokens))

    pad, title_h = 20, 40
    top_w, top_h = 8 * 40 + 20, 80
    top = Image.new('RGB', (top_w, top_h), 'white')
    td = ImageDraw.Draw(top)
    for i in range(8):
        orig = (originals[i] * 255).clip(0, 255).astype(np.uint8)
        recon = (recons[i] * 255).clip(0, 255).astype(np.uint8)
        top.paste(Image.fromarray(orig).resize((32, 32)), (i * 40 + 10, 5))
        top.paste(Image.fromarray(recon).resize((32, 32)), (i * 40 + 10, 42))
    td.text((5, top_h - 15), "上:原始 下:重建", fill='gray', font=try_load_font(11))

    bot_w, bot_h = 600, 180
    bot = Image.new('RGB', (bot_w, bot_h), 'white')
    bd = ImageDraw.Draw(bot)
    draw_loss_curve(bd, 10, 10, 180, 120, recon_losses, 'blue', title='VQ-VAE 训练损失')
    bd.text((10, 140), f"MSE: {recon_losses[0]:.3f} -> {recon_losses[-1]:.3f}", fill='gray', font=try_load_font(11))
    bd.text((220, 10), f"码本: 16", fill='black', font=try_load_font(14))
    bd.text((220, 35), f"使用: {unique_tokens}/16", fill='blue' if unique_tokens > 8 else 'red', font=try_load_font(14))
    bd.text((220, 60), f"token shape: {list(tokens.shape)}", fill='gray', font=try_load_font(11))
    bd.text((220, 90), "动作反事实:", fill='black', font=try_load_font(14))
    bd.text((220, 115), "同一起点, 5 种动作 -> 5 种未来", fill='gray', font=try_load_font(11))
    action_colors = ['gray', 'blue', 'green', 'red', 'orange']
    action_labels = ['S', 'L', 'R', 'U', 'D']
    for i in range(5):
        x = 220 + i * 30
        bd.rectangle([x, 140, x + 24, 164], outline=action_colors[i], width=2)
        bd.text((x + 6, 145), action_labels[i], fill=action_colors[i], font=try_load_font(12))

    img_w = max(top_w, bot_w) + pad * 2
    img_h = top_h + bot_h + title_h + pad * 3 + 20
    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 5), "PA1-B: 可控制视频世界模型——VQ-VAE + 动作条件", fill='black', font=try_load_font(18))
    canvas.paste(top, (pad, title_h + pad))
    canvas.paste(bot, (pad, title_h + pad + top_h + pad))
    canvas.save(output_path)
    print(f"Saved: {output_path}")


# ──── PA1-C: JEPA 特征质量 ────
def visualize_pa1c_jepa(output_path):
    from hwm.jepa import TinyVideoJEPA, feature_spread, jepa_batch_from_episodes, patchify_video

    episodes = make_pixelworld_dataset(num_episodes=8, length=10, seed=42)
    histories, actions_batch, positions = jepa_batch_from_episodes(episodes, history_length=3)

    feature_size = 16
    patch_size = 4
    num_patches = (histories.shape[-2] // patch_size) * (histories.shape[-1] // patch_size)

    jepa = TinyVideoJEPA(feature_size=feature_size, action_size=5,
                         patch_size=patch_size, num_patches=num_patches)
    optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)
    losses, spreads = [], []

    for step in range(30):
        optimizer.zero_grad()
        loss, pred, target, features = jepa.loss(histories, actions_batch)
        loss.backward()
        optimizer.step()
        jepa.update_target(momentum=0.99)
        losses.append(float(loss))
        with torch.no_grad():
            spreads.append(float(feature_spread(features)))

    pad, title_h, cell = 20, 40, 200
    img_w = 3 * cell + 4 * pad
    img_h = cell + title_h + pad * 2 + 50

    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 5), "PA1-C: JEPA 特征质量评估——坍缩诊断 + 特征分析", fill='black', font=try_load_font(18))

    p1 = Image.new('RGB', (cell, cell), 'white')
    d1 = ImageDraw.Draw(p1)
    draw_loss_curve(d1, 10, 15, cell - 20, cell - 50, losses, 'blue', title='Feature Loss')
    d1.text((10, cell - 30), f"{losses[0]:.3f} -> {losses[-1]:.3f}", fill='gray', font=try_load_font(11))
    canvas.paste(p1, (pad, title_h + pad))
    draw.text((pad + 5, title_h + cell + pad + 5), "特征预测损失", fill='black', font=try_load_font(13))

    p2 = Image.new('RGB', (cell, cell), 'white')
    d2 = ImageDraw.Draw(p2)
    draw_loss_curve(d2, 10, 15, cell - 20, cell - 50, spreads, 'green', title='Feature Spread')
    d2.text((10, cell - 30), f"spread: {spreads[-1]:.3f}", fill='gray', font=try_load_font(11))
    status = "OK: 未坍缩" if spreads[-1] > 0.1 else "WARN: 可能坍缩!"
    d2.text((10, cell - 48), status, fill='green' if spreads[-1] > 0.1 else 'red', font=try_load_font(12))
    canvas.paste(p2, (cell + 2 * pad, title_h + pad))
    draw.text((cell + 2 * pad + 5, title_h + cell + pad + 5), "坍缩诊断", fill='black', font=try_load_font(13))

    p3 = Image.new('RGB', (cell, cell), 'white')
    d3 = ImageDraw.Draw(p3)
    d3.text((10, 10), "特征统计:", fill='black', font=try_load_font(14))
    with torch.no_grad():
        d3.text((10, 35), f"mean: {features.mean():.4f}", fill='gray', font=try_load_font(12))
        d3.text((10, 55), f"std: {features.std():.4f}", fill='gray', font=try_load_font(12))
        d3.text((10, 75), f"min: {features.min():.4f}", fill='gray', font=try_load_font(12))
        d3.text((10, 95), f"max: {features.max():.4f}", fill='gray', font=try_load_font(12))
    d3.text((10, 120), f"shape: {list(features.shape)}", fill='gray', font=try_load_font(11))
    d3.text((10, 145), "不重建像素，", fill='black', font=try_load_font(13))
    d3.text((10, 165), "只在特征空间评估", fill='black', font=try_load_font(13))
    canvas.paste(p3, (2 * (cell + pad) + pad, title_h + pad))
    draw.text((2 * (cell + pad) + pad + 5, title_h + cell + pad + 5), "特征分析", fill='black', font=try_load_font(13))

    canvas.save(output_path)
    print(f"Saved: {output_path}")


# ──── PA2: 研究循环 ────
def visualize_pa2_research(output_path):
    from hwm.evaluation import horizon_errors

    episodes = make_pixelworld_dataset(num_episodes=8, length=10, seed=42)
    from hwm.foundations import center_of_red
    positions, next_positions, actions_list = [], [], []
    for ep in episodes:
        for t in range(len(ep.observations) - 1):
            positions.append(center_of_red(ep.observations[t]))
            next_positions.append(center_of_red(ep.observations[t + 1]))
            actions_list.append(ep.actions[t])

    pos_t = torch.tensor(np.array(positions), dtype=torch.float32)
    npos_t = torch.tensor(np.array(next_positions), dtype=torch.float32)
    act_t = torch.tensor(np.array(actions_list), dtype=torch.long)

    model_small = PositionDynamics(action_size=5, hidden_size=16)
    losses_small = fit_position_dynamics(model_small, pos_t, act_t, npos_t, updates=80)

    model_big = PositionDynamics(action_size=5, hidden_size=64)
    losses_big = fit_position_dynamics(model_big, pos_t, act_t, npos_t, updates=80)

    def predict_fn(m):
        def fn(pos, acts):
            p = pos.unsqueeze(0)
            trajectory = []
            for a in acts:
                p = m(p, torch.tensor([a]))
                trajectory.append(p[0].detach().numpy())
            return np.stack(trajectory)
        return fn

    starts = pos_t[:5]
    action_seqs = [act_t[i:i+3].numpy() for i in range(5)]
    true_rollouts = [npos_t[i:i+3].numpy() for i in range(5)]

    errors_small = horizon_errors(predict_fn(model_small), starts, action_seqs, true_rollouts)
    errors_big = horizon_errors(predict_fn(model_big), starts, action_seqs, true_rollouts)

    pad, title_h, cell = 20, 40, 160
    n_panels = 4
    img_w = n_panels * cell + (n_panels + 1) * pad
    img_h = cell + title_h + pad * 2 + 50

    canvas = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 5), "PA2: 研究循环——从失败到改进", fill='black', font=try_load_font(18))

    p1 = Image.new('RGB', (cell, cell), 'white')
    d1 = ImageDraw.Draw(p1)
    draw_loss_curve(d1, 10, 15, cell - 20, cell - 50, losses_small, 'red', title='原始模型 loss')
    d1.text((10, cell - 30), f"最终: {losses_small[-1]:.3f}", fill='gray', font=try_load_font(11))
    canvas.paste(p1, (pad, title_h + pad))
    draw.text((pad + 5, title_h + cell + pad + 5), "稳定失败", fill='black', font=try_load_font(13))

    p2 = Image.new('RGB', (cell, cell), 'white')
    d2 = ImageDraw.Draw(p2)
    d2.text((10, 10), "竞争解释:", fill='black', font=try_load_font(14))
    d2.text((10, 35), "H1: 容量不足", fill='blue', font=try_load_font(13))
    d2.text((10, 55), "-> 增大 hidden", fill='blue', font=try_load_font(13))
    d2.text((10, 85), "H2: 数据不足", fill='orange', font=try_load_font(13))
    d2.text((10, 105), "-> 增加 episodes", fill='orange', font=try_load_font(13))
    d2.text((10, 135), "证伪: 若 H1 对", fill='gray', font=try_load_font(11))
    d2.text((10, 150), "增大模型应有效", fill='gray', font=try_load_font(11))
    canvas.paste(p2, (cell + 2 * pad, title_h + pad))
    draw.text((cell + 2 * pad + 5, title_h + cell + pad + 5), "两种解释", fill='black', font=try_load_font(13))

    p3 = Image.new('RGB', (cell, cell), 'white')
    d3 = ImageDraw.Draw(p3)
    draw_loss_curve(d3, 10, 15, cell - 20, cell - 50, losses_big, 'green', title='改进模型 loss')
    d3.text((10, cell - 30), f"最终: {losses_big[-1]:.3f}", fill='gray', font=try_load_font(11))
    improvement = (losses_small[-1] - losses_big[-1]) / losses_small[-1] * 100
    d3.text((10, cell - 48), f"改善: {improvement:.1f}%", fill='green', font=try_load_font(12))
    canvas.paste(p3, (2 * (cell + pad) + pad, title_h + pad))
    draw.text((2 * (cell + pad) + pad + 5, title_h + cell + pad + 5), "最小改动", fill='black', font=try_load_font(13))

    p4 = Image.new('RGB', (cell, cell), 'white')
    d4 = ImageDraw.Draw(p4)
    d4.text((10, 5), "多步误差对照:", fill='black', font=try_load_font(13))
    all_err = list(errors_small) + list(errors_big)
    max_err, min_err = max(all_err), min(all_err)
    err_range = max_err - min_err + 1e-8
    n_s = len(errors_small)
    for i in range(n_s - 1):
        x1 = 10 + int(i / n_s * (cell - 20))
        y1 = 30 + int((1 - (errors_small[i] - min_err) / err_range) * (cell - 60))
        x2 = 10 + int((i + 1) / n_s * (cell - 20))
        y2 = 30 + int((1 - (errors_small[i + 1] - min_err) / err_range) * (cell - 60))
        d4.line([(x1, y1), (x2, y2)], fill='red', width=2)
    n_b = len(errors_big)
    for i in range(n_b - 1):
        x1 = 10 + int(i / n_b * (cell - 20))
        y1 = 30 + int((1 - (errors_big[i] - min_err) / err_range) * (cell - 60))
        x2 = 10 + int((i + 1) / n_b * (cell - 20))
        y2 = 30 + int((1 - (errors_big[i + 1] - min_err) / err_range) * (cell - 60))
        d4.line([(x1, y1), (x2, y2)], fill='green', width=2)
    d4.text((10, cell - 35), "红:原始 绿:改进", fill='gray', font=try_load_font(11))
    d4.text((10, cell - 18), f"horizon 1->{n_s}", fill='gray', font=try_load_font(11))
    canvas.paste(p4, (3 * (cell + pad) + pad, title_h + pad))
    draw.text((3 * (cell + pad) + pad + 5, title_h + cell + pad + 5), "公平对照", fill='black', font=try_load_font(13))

    canvas.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'docs' / 'public' / 'carracing'
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_pa0_loop(output_dir / 'world-loop.png')
    visualize_pa1a_cycle(output_dir / 'dreamer-lite.png')
    visualize_pa1b_controllable(output_dir / 'controllable-video.png')
    visualize_pa1c_jepa(output_dir / 'jepa-quality.png')
    visualize_pa2_research(output_dir / 'research-cycle.png')

    print("\nAll assignment visualizations generated!")
