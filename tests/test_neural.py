import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "安装 requirements-neural.txt 后运行神经测试")
class NeuralWorldModelTest(unittest.TestCase):
    def setUp(self):
        from hwm.data import make_pixelworld_dataset
        from hwm.neural import TinyWorldModel, batch_from_episodes

        torch.manual_seed(0)
        episodes = make_pixelworld_dataset(num_episodes=4, length=8, seed=0)
        self.batch = batch_from_episodes(episodes, sequence_length=8)
        self.model = TinyWorldModel()

    def test_world_model_shapes_and_gradient(self):
        from hwm.neural import world_model_loss

        observations, actions, rewards, dones = self.batch
        loss, metrics, outputs = world_model_loss(
            self.model, observations, actions, rewards, dones
        )
        loss.backward()
        self.assertEqual(outputs["reconstruction"].shape, (4, 8, 16, 16, 3))
        self.assertEqual(outputs["feature"].shape, (4, 8, 80))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.model.encoder.network[0].weight.grad)
        self.assertGreaterEqual(float(metrics["kl"]), 0.0)

    def test_world_model_loss_falls_on_tiny_batch(self):
        from hwm.neural import world_model_loss

        observations, actions, rewards, dones = self.batch
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            loss, _, _ = world_model_loss(
                self.model, observations, actions, rewards, dones
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])

    def test_imagination_and_lambda_return(self):
        from hwm.neural import Actor, Critic, imagine, lambda_returns

        observations, actions, _, _ = self.batch
        outputs = self.model(observations, actions, sample=False)
        posterior = outputs["posterior"]
        start = type(posterior)(
            posterior.deterministic[:, -1],
            posterior.stochastic[:, -1],
            posterior.mean[:, -1],
            posterior.std[:, -1],
        )
        actor = Actor()
        critic = Critic()
        imagined = imagine(self.model, actor, start, horizon=5)
        values = critic(imagined["features"])
        returns = lambda_returns(
            imagined["rewards"],
            imagined["continues"],
            values,
            values[:, -1].detach(),
        )
        self.assertEqual(imagined["actions"].shape, (4, 5))
        self.assertEqual(returns.shape, (4, 5))
        self.assertTrue(torch.isfinite(returns).all())


if __name__ == "__main__":
    unittest.main()
