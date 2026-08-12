import unittest

import numpy as np

from hwm.data import (
    MovingSquareWorld,
    ReplayBuffer,
    locate_red_square,
    make_pixelworld_dataset,
    pixelworld_transition_arrays,
)
from hwm.foundations import (
    block_average_decode,
    block_average_encode,
    cem_plan_1d,
    clip_by_norm,
    conv2d_valid,
    depth_to_points,
    patchify,
    points_to_occupancy,
    remember_velocity,
    reconstruction_mse,
    symlog,
    symexp,
    unpatchify,
)


class DataTest(unittest.TestCase):
    def test_episode_stops_when_goal_is_reached(self):
        world = MovingSquareWorld(goal=(2, 3))
        episode, positions = world.generate([2, 2, 2], start=(2, 2))
        self.assertEqual(len(episode.actions), 1)
        self.assertTrue(episode.dones[-1])
        self.assertEqual(positions[-1], (2, 3))

    def test_pixel_episode_has_aligned_time(self):
        world = MovingSquareWorld()
        episode, positions = world.generate([2, 2, 4], start=(2, 2))
        self.assertEqual(episode.observations.shape, (4, 16, 16, 3))
        self.assertEqual(episode.actions.shape, (3,))
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[-1], (3, 4))

    def test_replay_sequences_never_cross_episode(self):
        buffer = ReplayBuffer()
        for episode in make_pixelworld_dataset(num_episodes=5, length=8):
            buffer.add(episode)
        samples = buffer.sample(batch_size=20, sequence_length=4, seed=3)
        self.assertTrue(all(item["observations"].shape[0] == 5 for item in samples))
        self.assertTrue(all(item["actions"].shape[0] == 4 for item in samples))
        self.assertTrue(all("pixelworld" in item["episode_id"] for item in samples))

    def test_pixel_positions_can_be_read_from_images(self):
        world = MovingSquareWorld()
        episode, positions = world.generate([2, 4, 1], start=(3, 5))
        recovered = [
            tuple(locate_red_square(image).astype(int))
            for image in episode.observations
        ]
        self.assertEqual(recovered, positions)
        current, actions, following = pixelworld_transition_arrays([episode])
        self.assertEqual(current.shape, (3, 2))
        self.assertEqual(actions.tolist(), [2, 4, 1])
        self.assertEqual(tuple(following[-1].astype(int)), positions[-1])


class FoundationComponentTest(unittest.TestCase):
    def test_convolution_detects_a_vertical_edge(self):
        image = np.zeros((5, 5), dtype=np.float32)
        image[:, 3:] = 1.0
        kernel = np.array([[-1, 0, 1]] * 3, dtype=np.float32)
        output = conv2d_valid(image, kernel)
        self.assertGreater(output.max(), 0)

    def test_patch_round_trip_is_exact(self):
        image = np.arange(8 * 8 * 3).reshape(8, 8, 3)
        tokens = patchify(image, 4)
        recovered = unpatchify(tokens, image.shape, 4)
        np.testing.assert_array_equal(image, recovered)

    def test_history_recovers_velocity_direction(self):
        world = MovingSquareWorld()
        episode, _ = world.generate([2, 2], start=(2, 2))
        state = remember_velocity(episode.observations)
        self.assertEqual(state.shape, (3, 4))
        self.assertGreater(state[-1, 3], 0)

    def test_compression_trades_detail_for_size(self):
        world = MovingSquareWorld()
        image = world.render((3, 5))
        latent = block_average_encode(image, block_size=4)
        recovered = block_average_decode(latent, block_size=4)
        self.assertEqual(latent.shape, (4, 4, 3))
        self.assertGreater(reconstruction_mse(image, recovered), 0)

    def test_depth_points_form_occupancy(self):
        depth = np.full((3, 3), 2.0, dtype=np.float32)
        points = depth_to_points(depth, fx=2.0, fy=2.0, cx=1.0, cy=1.0)
        grid = points_to_occupancy(points, (-2, 2), (0, 4), 1.0)
        self.assertEqual(points.shape, (9, 3))
        self.assertGreater(grid.sum(), 0)

    def test_symlog_round_trip_and_clipping(self):
        values = np.array([-100000.0, -1.0, 0.0, 1.0, 100000.0])
        np.testing.assert_allclose(symexp(symlog(values)), values, rtol=1e-5)
        clipped = clip_by_norm(np.array([30.0, 40.0]), max_norm=5.0)
        self.assertAlmostEqual(np.linalg.norm(clipped), 5.0, places=5)

    def test_cem_moves_towards_target(self):
        actions, history = cem_plan_1d(start=0.0, target=3.0, horizon=5)
        final_position = actions.sum()
        self.assertLess(abs(final_position - 3.0), 0.25)
        self.assertGreaterEqual(history[-1], history[0])


if __name__ == "__main__":
    unittest.main()
