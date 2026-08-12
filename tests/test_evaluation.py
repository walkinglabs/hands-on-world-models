from pathlib import Path
import tempfile
import unittest

import numpy as np

from hwm.evaluation import (
    RunManifest,
    calibration_bins,
    counterfactual_sensitivity,
    horizon_errors,
    sha256_file,
)


class EvaluationTest(unittest.TestCase):
    def test_horizon_and_counterfactual(self):
        def predict(start, actions):
            return np.asarray([start + sum(actions[: index + 1]) for index in range(len(actions))])

        starts = [0, 1]
        actions = [[1, 1, 1], [-1, -1, -1]]
        truth = np.asarray([[1, 2, 3], [0, -1, -2]])
        self.assertTrue(np.allclose(horizon_errors(predict, starts, actions, truth), 0))
        sensitivity = counterfactual_sensitivity(predict, 0, [[0, 0], [1, 1]])
        self.assertGreater(sensitivity[1], 0)

    def test_calibration_and_manifest(self):
        bins = calibration_bins([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], 2)
        self.assertEqual(len(bins), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = RunManifest(
                experiment="smoke",
                route="Z",
                seed=0,
                dataset="toy",
                split="test",
                command="python smoke.py",
                started_at="2026-08-12T00:00:00Z",
                wall_time_seconds=1.0,
            )
            manifest.save(path)
            self.assertTrue(path.exists())
            self.assertEqual(len(sha256_file(path)), 64)


if __name__ == "__main__":
    unittest.main()
