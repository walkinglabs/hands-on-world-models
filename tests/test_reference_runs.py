import json
from pathlib import Path
import unittest

from hwm.evaluation import sha256_file


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "runs" / "reference" / "position-dynamics"


class ReferenceRunTest(unittest.TestCase):
    def test_reference_evidence_is_complete_and_self_consistent(self):
        metrics = json.loads((REFERENCE / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = REFERENCE / "position-dynamics.pt"

        self.assertGreater(
            metrics["planned_success_rate"], metrics["random_success_rate"]
        )
        self.assertLess(
            metrics["planned_final_distance"], metrics["random_final_distance"]
        )
        self.assertEqual(manifest["checkpoint_sha256"], sha256_file(checkpoint))
        self.assertIn("not Dreamer-lite", manifest["notes"])
        self.assertIn("--output runs/reference/position-dynamics", manifest["command"])


if __name__ == "__main__":
    unittest.main()
