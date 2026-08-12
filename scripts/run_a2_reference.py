"""运行路线 A 的 CPU 参考闭环，并保存可复现证据。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from hwm.control import PositionDynamics, evaluate_controllers, fit_position_dynamics
from hwm.data import MovingSquareWorld, pixelworld_transition_arrays
from hwm.evaluation import RunManifest, sha256_file


def build_training_episodes():
    world = MovingSquareWorld()
    episodes = []
    for row in (0, 3, 6, 9, 12, 13):
        for col in (0, 3, 6, 9, 12, 13):
            for action in range(5):
                episode, _ = world.generate([action], start=(row, col))
                episodes.append(episode)
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/a2-reference"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    episodes = build_training_episodes()
    positions, actions, next_positions = pixelworld_transition_arrays(episodes)
    model = PositionDynamics(hidden_size=48)
    losses = fit_position_dynamics(
        model,
        positions,
        actions,
        next_positions,
        updates=args.updates,
    )
    metrics = evaluate_controllers(
        model,
        starts=[(1, 1), (2, 8), (8, 2), (5, 5)],
        max_steps=24,
        random_seeds=20,
    )

    checkpoint = args.output / "position-dynamics.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "planned_success_rate": metrics["planned_success_rate"],
        "planned_final_distance": metrics["planned_final_distance"],
        "random_success_rate": metrics["random_success_rate"],
        "random_final_distance": metrics["random_final_distance"],
        "training_transitions": int(len(actions)),
        "test_starts": [[1, 1], [2, 8], [8, 2], [5, 5]],
    }
    result_path = args.output / "metrics.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = RunManifest(
        experiment="a2-position-dynamics-reference",
        route="A",
        seed=args.seed,
        dataset="pixelworld-v1-generated-grid",
        split="fixed-train-grid-and-four-held-out-starts",
        command=(
            "python scripts/run_a2_reference.py "
            f"--output {args.output} --seed {args.seed} --updates {args.updates}"
        ),
        started_at=started_at,
        wall_time_seconds=time.perf_counter() - start_time,
        device="cpu",
        gpu="not-applicable",
        cuda="not-applicable",
        checkpoint_sha256=sha256_file(checkpoint),
        notes=(
            f"Python {platform.python_version()}, PyTorch {torch.__version__}; "
            "toy interpretable-state baseline, not Dreamer-lite"
        ),
    )
    manifest.save(args.output / "manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
