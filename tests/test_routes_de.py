import importlib.util
import unittest

import numpy as np

from hwm.foundations import depth_to_points, points_to_occupancy
from hwm.spatial import make_camera_transform, transform_points


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SpatialGeometryTest(unittest.TestCase):
    def test_camera_transform_and_occupancy(self):
        depth = np.ones((4, 4), dtype=np.float32) * 2
        points = depth_to_points(depth, 4, 4, 1.5, 1.5)
        transform = make_camera_transform(tx=1.0)
        world = transform_points(points, transform)
        self.assertTrue(np.allclose(world[:, 0], points[:, 0] + 1))
        grid = points_to_occupancy(world, (-1, 3), (0, 4), 0.5)
        self.assertGreater(grid.sum(), 0)


@unittest.skipUnless(TORCH_AVAILABLE, "安装 requirements-neural.txt 后运行路线 D/E")
class RouteDENeuralTest(unittest.TestCase):
    def test_vla_and_outcome_model(self):
        import torch

        from hwm.robot import TinyVLA, TabletopOutcomeModel, make_tabletop_dataset, outcome_loss

        torch.manual_seed(0)
        data = make_tabletop_dataset(16, seed=0)
        policy = TinyVLA()
        chunks = policy(data["images"], data["instructions"], data["states"])
        self.assertEqual(tuple(chunks.shape), (16, 3, 2))
        loss = torch.nn.functional.mse_loss(chunks, data["action_chunks"])
        loss.backward()
        self.assertIsNotNone(policy.vision[0].weight.grad)
        outcome = TabletopOutcomeModel()
        loss, _, _ = outcome_loss(
            outcome,
            data["states"],
            data["action_chunks"][:, 0],
            data["next_states"],
            data["collisions"],
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_spatial_models(self):
        import torch

        from hwm.spatial import (
            TinyNeuralField,
            TinyOccupancyPredictor,
            make_colored_sphere_samples,
            make_moving_occupancy_dataset,
        )

        history, actions, future = make_moving_occupancy_dataset(8, seed=0)
        predictor = TinyOccupancyPredictor()
        logits = predictor(history, actions)
        self.assertEqual(logits.shape, future.shape)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, future).backward()
        coordinates, density, color = make_colored_sphere_samples(32)
        field = TinyNeuralField()
        predicted_density, predicted_color = field(coordinates)
        self.assertEqual(predicted_density.shape, density.shape)
        self.assertEqual(predicted_color.shape, color.shape)


if __name__ == "__main__":
    unittest.main()
