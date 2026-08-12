from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hwm.data_cli import load_registry, main


class DataCLITest(unittest.TestCase):
    def test_registry_has_honest_status(self):
        registry = load_registry()
        ids = {item["id"] for item in registry["datasets"]}
        self.assertIn("pixelworld-v1", ids)
        self.assertIn("nuscenes-mini", ids)
        statuses = {item["status"] for item in registry["datasets"]}
        self.assertIn("artifact-ready", statuses)
        self.assertIn("source-known", statuses)

    def test_generate_pixelworld_with_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "generate",
                        "pixelworld",
                        "--output",
                        directory,
                        "--seed",
                        "7",
                        "--num-samples",
                        "3",
                    ]
                )
            self.assertEqual(code, 0)
            artifact = Path(directory) / "pixelworld-seed7.npz"
            metadata = Path(directory) / "pixelworld-seed7.json"
            self.assertTrue(artifact.exists() and metadata.exists())
            with np.load(artifact) as data:
                self.assertEqual(data["observations"].shape[0], 3)

    def test_generate_lineworld(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "generate",
                        "lineworld",
                        "--output",
                        directory,
                        "--seed",
                        "2",
                        "--num-samples",
                        "4",
                    ]
                )
            self.assertEqual(code, 0)
            with np.load(Path(directory) / "lineworld-seed2.npz") as data:
                self.assertEqual(len(np.unique(data["episode_ids"])), 4)

    def test_generate_tabletop_has_language_and_time_contract(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Tabletop artifact 需要 PyTorch")
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                main(
                    [
                        "generate",
                        "tabletop",
                        "--output",
                        directory,
                        "--seed",
                        "3",
                        "--num-samples",
                        "4",
                    ]
                )
            with np.load(Path(directory) / "tabletop-seed3.npz") as data:
                self.assertEqual(data["instruction_texts"].shape, (4,))
                self.assertEqual(data["time_index"].tolist(), [0, 0, 0, 0])
                self.assertTrue(np.all(data["control_frequency_hz"] == 10))

    def test_generate_moving_sphere(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Moving sphere artifact 需要 PyTorch")
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                main(
                    [
                        "generate",
                        "moving-sphere",
                        "--output",
                        directory,
                        "--seed",
                        "4",
                        "--num-samples",
                        "16",
                    ]
                )
            artifact = Path(directory) / "moving-sphere-seed4.npz"
            metadata = Path(directory) / "moving-sphere-seed4.json"
            with np.load(artifact) as data:
                self.assertEqual(data["coordinates"].shape, (16, 3))
                self.assertEqual(data["density"].shape, (16, 1))
            self.assertIn("generator_sha256", metadata.read_text())


if __name__ == "__main__":
    unittest.main()
