import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "安装 requirements-neural.txt 后运行控制测试")
class PixelControlTest(unittest.TestCase):
    def test_learned_dynamics_improves_real_action(self):
        from hwm.control import (
            PositionDynamics,
            evaluate_controllers,
            fit_position_dynamics,
        )
        from hwm.data import MovingSquareWorld, pixelworld_transition_arrays

        torch.manual_seed(0)
        world = MovingSquareWorld()
        episodes = []
        for row in (0, 3, 6, 9, 12, 13):
            for col in (0, 3, 6, 9, 12, 13):
                for action in range(5):
                    episode, _ = world.generate([action], start=(row, col))
                    episodes.append(episode)
        positions, actions, next_positions = pixelworld_transition_arrays(episodes)
        model = PositionDynamics(hidden_size=48)
        losses = fit_position_dynamics(
            model,
            positions,
            actions,
            next_positions,
            updates=100,
        )
        metrics = evaluate_controllers(
            model,
            starts=[(1, 1), (2, 8), (8, 2), (5, 5)],
            max_steps=24,
            random_seeds=5,
        )
        self.assertLess(losses[-1], losses[0] * 0.25)
        self.assertGreater(metrics["planned_success_rate"], 0.5)
        self.assertLess(
            metrics["planned_final_distance"],
            metrics["random_final_distance"],
        )


if __name__ == "__main__":
    unittest.main()
