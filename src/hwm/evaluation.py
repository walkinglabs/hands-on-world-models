"""跨路线共用的评价与运行证据工具。"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Optional

import numpy as np


def horizon_errors(predict, starts, action_sequences, true_rollouts):
    """返回每个 horizon 的平均误差，不把多步藏进一个平均数。"""
    true_rollouts = np.asarray(true_rollouts, dtype=np.float32)
    predictions = np.stack(
        [predict(start, actions) for start, actions in zip(starts, action_sequences)]
    ).astype(np.float32)
    if predictions.shape != true_rollouts.shape:
        raise ValueError("预测与真实 rollout shape 不一致")
    axes = tuple(range(2, predictions.ndim))
    squared = (predictions - true_rollouts) ** 2
    per_step = squared.mean(axis=axes) if axes else squared
    return per_step.mean(axis=0)


def counterfactual_sensitivity(predict, start, action_sequences):
    """固定起点，只换动作；结果完全相同是危险信号。"""
    outputs = np.stack([predict(start, actions) for actions in action_sequences])
    reference = outputs[0]
    axes = tuple(range(1, outputs.ndim))
    return np.mean(np.abs(outputs - reference), axis=axes)


def calibration_bins(probabilities, outcomes, num_bins=5):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    outcomes = np.asarray(outcomes, dtype=np.float32)
    edges = np.linspace(0, 1, num_bins + 1)
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        include = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if include.any():
            result.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(include.sum()),
                    "confidence": float(probabilities[include].mean()),
                    "frequency": float(outcomes[include].mean()),
                }
            )
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RunManifest:
    experiment: str
    route: str
    seed: int
    dataset: str
    split: str
    command: str
    started_at: str
    wall_time_seconds: float
    device: str = "cpu"
    gpu: str = "not-recorded"
    cuda: str = "not-recorded"
    peak_allocated_mb: Optional[float] = None
    peak_reserved_mb: Optional[float] = None
    checkpoint_sha256: Optional[str] = None
    notes: str = ""

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class RunTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds = time.perf_counter() - self.start


def runtime_summary():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
