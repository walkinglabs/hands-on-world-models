import random
import unittest

from hwm.gridworld import (
    EmpiricalDynamics,
    GridWorld,
    Transition,
    lookahead,
    mpc_episode,
    rollout,
)


class GridWorldTest(unittest.TestCase):
    def setUp(self):
        self.world = GridWorld()

    def test_wall_and_boundary_keep_the_state(self):
        self.assertEqual(self.world.next_state((0, 0), "up"), (0, 0))
        self.assertEqual(self.world.next_state((1, 0), "right"), (1, 0))

    def test_greedy_action_enters_the_trap(self):
        action = self.world.greedy_action(self.world.start)
        transition = self.world.transition(self.world.start, action)
        self.assertEqual(action, "right")
        self.assertEqual(transition.next_state, (0, 1))
        self.assertEqual(transition.reward, -10.0)
        self.assertTrue(transition.done)

    def test_rollout_reaches_the_goal(self):
        actions = ("down", "down", "right", "right", "up", "up")
        transitions = rollout(self.world, self.world.start, actions)
        self.assertEqual(transitions[-1].next_state, self.world.goal)
        self.assertEqual(sum(item.reward for item in transitions), 5.0)

    def test_planning_depth_changes_the_result(self):
        shallow, _ = mpc_episode(self.world, self.world, depth=1, max_steps=12)
        deep, plans = mpc_episode(self.world, self.world, depth=6, max_steps=12)

        self.assertNotEqual(shallow[-1].next_state, self.world.goal)
        self.assertEqual(deep[-1].next_state, self.world.goal)
        self.assertEqual(len(deep), 6)
        self.assertEqual(plans[0].evaluated_sequences, 4**6)

    def test_slip_is_reproducible_with_a_seed(self):
        slippery = GridWorld(slip_probability=0.5)
        first = slippery.step((2, 0), "right", random.Random(1))
        second = slippery.step((2, 0), "right", random.Random(1))
        self.assertEqual(first, second)
        self.assertEqual(first.next_state, (2, 0))

    def test_empirical_model_keeps_multiple_outcomes(self):
        samples = [
            Transition((2, 0), "right", -1.0, (2, 1), False),
            Transition((2, 0), "right", -1.0, (2, 1), False),
            Transition((2, 0), "right", -1.0, (2, 0), False),
        ]
        model = EmpiricalDynamics().fit(samples)
        distribution = model.distribution((2, 0), "right")

        self.assertAlmostEqual(distribution[(2, 1)], 2 / 3)
        self.assertAlmostEqual(distribution[(2, 0)], 1 / 3)

    def test_lookahead_rejects_zero_depth(self):
        with self.assertRaises(ValueError):
            lookahead(self.world, self.world.start, depth=0)


if __name__ == "__main__":
    unittest.main()
