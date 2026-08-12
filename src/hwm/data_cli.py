"""课程数据的统一查看与项目内生成入口。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data" / "registry.json"


def load_registry(path=REGISTRY):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_pixelworld(output, seed, num_samples):
    from .data import make_pixelworld_dataset

    episodes = make_pixelworld_dataset(
        num_episodes=num_samples,
        length=18,
        seed=seed,
    )
    np.savez_compressed(
        output,
        observations=np.stack([episode.observations for episode in episodes]),
        actions=np.stack([episode.actions for episode in episodes]),
        rewards=np.stack([episode.rewards for episode in episodes]),
        dones=np.stack([episode.dones for episode in episodes]),
        episode_ids=np.asarray([episode.episode_id for episode in episodes]),
    )


def generate_lineworld(output, seed, num_samples):
    from .gridworld import LineWorld

    environment = LineWorld()

    def random_policy(_state, rng):
        return rng.choice(environment.actions)

    transitions, episode_ids = environment.collect(
        random_policy,
        episodes=num_samples,
        seed=seed,
    )
    np.savez_compressed(
        output,
        states=np.asarray([item.state for item in transitions]),
        actions=np.asarray([item.action for item in transitions]),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        next_states=np.asarray([item.next_state for item in transitions]),
        dones=np.asarray([item.done for item in transitions], dtype=bool),
        episode_ids=np.asarray(episode_ids),
    )


def generate_tabletop(output, seed, num_samples):
    from .robot import make_tabletop_dataset

    data = make_tabletop_dataset(num_samples=num_samples, seed=seed)
    arrays = {name: value.detach().cpu().numpy() for name, value in data.items()}
    np.savez_compressed(output, **arrays)


def generate_occupancy(output, seed, num_samples):
    from .spatial import make_moving_occupancy_dataset

    history, actions, future = make_moving_occupancy_dataset(
        num_samples=num_samples,
        seed=seed,
    )
    np.savez_compressed(
        output,
        history=history.numpy(),
        actions=actions.numpy(),
        future=future.numpy(),
    )


GENERATORS = {
    "lineworld": generate_lineworld,
    "pixelworld": generate_pixelworld,
    "tabletop": generate_tabletop,
    "occupancy": generate_occupancy,
}


def artifact_metadata(path, dataset, seed, num_samples):
    return {
        "dataset": dataset,
        "seed": seed,
        "num_samples": num_samples,
        "artifact": path.name,
        "sha256": sha256(path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="动手学世界模型数据工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="列出数据状态")
    generate = subparsers.add_parser("generate", help="生成项目内 T1 数据")
    generate.add_argument("dataset", choices=sorted(GENERATORS))
    generate.add_argument("--output", type=Path, default=Path("artifacts/data"))
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args(argv)

    if args.command == "list":
        for item in load_registry()["datasets"]:
            routes = ",".join(item["routes"])
            print(f"{item['id']:24s} {item['tier']:2s} {item['status']:21s} {routes}")
        return 0

    output_directory = args.output.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact = output_directory / f"{args.dataset}-seed{args.seed}.npz"
    GENERATORS[args.dataset](artifact, args.seed, args.num_samples)
    metadata = artifact_metadata(
        artifact,
        args.dataset,
        args.seed,
        args.num_samples,
    )
    metadata_path = artifact.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
