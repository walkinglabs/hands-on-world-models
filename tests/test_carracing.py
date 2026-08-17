"""scripts/run_carracing.py 的单元测试：VAE/MDN-RNN/CMA-ES 纯逻辑，不碰真实环境。"""

import unittest

import numpy as np
import torch

from scripts.run_carracing import (
    ConvVAE,
    LinearController,
    MDNRNN,
    MinimalCMAES,
    resize_frame,
)


class TestConvVAE(unittest.TestCase):
    def test_roundtrip_shape(self):
        model = ConvVAE(z_size=32)
        frames = torch.zeros(4, 64, 64, 3)
        mean, logvar = model.encode(frames)
        self.assertEqual(mean.shape, (4, 32))
        z = model.reparameterize(mean, logvar)
        reconstruction = model.decode(z)
        self.assertEqual(reconstruction.shape, (4, 3, 64, 64))

    def test_loss_decreases_one_step(self):
        model = ConvVAE(z_size=32)
        frames = torch.rand(8, 64, 64, 3)
        before = model.loss(frames)[0].item()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            total, _, _ = model.loss(frames)
            total.backward()
            optimizer.step()
        after = model.loss(frames)[0].item()
        self.assertLess(after, before)


class TestMDNRNN(unittest.TestCase):
    def test_sample_shapes(self):
        model = MDNRNN(z_size=32, hidden_size=64, components=5)
        z = torch.zeros(1, 32)
        action = torch.zeros(1, 3)
        hidden = torch.zeros(1, 64)
        z_next, reward, hidden_next = model.sample(z, action, hidden)
        self.assertEqual(z_next.shape, (1, 32))
        self.assertEqual(reward.shape, (1, 1))
        self.assertEqual(hidden_next.shape, (1, 64))

    def test_loss_is_scalar_and_finite(self):
        model = MDNRNN(z_size=32, hidden_size=64, components=5)
        z = torch.randn(16, 32)
        action = torch.randn(16, 3)
        target_z = torch.randn(16, 32)
        target_r = torch.randn(16)
        loss = model.loss(z, action, target_z, target_r).mean()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dim(), 0)


class TestController(unittest.TestCase):
    def test_parameter_count_matches_paper(self):
        # 原文 867 = (z 32 + h 256) * 3 动作 + 3 偏置
        controller = LinearController(32 + 256)
        self.assertEqual(controller.parameters.size, 867)

    def test_act_clips_to_unit_box(self):
        controller = LinearController(32 + 256)
        features = np.zeros((1, 288))
        action = controller.act(features, np.zeros(867))
        self.assertEqual(action.shape, (1, 3))
        self.assertTrue(np.all(np.abs(action) <= 1.0))


class TestMinimalCMAES(unittest.TestCase):
    def test_tell_updates_mean(self):
        optimizer = MinimalCMAES(dimension=10, population=8, seed=0)
        samples = optimizer.ask()
        fitness = np.random.RandomState(1).randn(8)
        mean_before = optimizer.mean.copy()
        optimizer.tell(samples, fitness)
        self.assertFalse(np.allclose(mean_before, optimizer.mean))

    def test_better_individual_dominates(self):
        optimizer = MinimalCMAES(dimension=4, population=4, seed=0)
        samples = optimizer.ask()
        # 让第一个样本最优：新均值应更接近第一个样本
        fitness = np.array([-10.0, 0.0, 0.0, 0.0])
        optimizer.tell(samples, fitness)
        self.assertLess(
            np.linalg.norm(optimizer.mean - samples[0]),
            np.linalg.norm(optimizer.mean - samples[1]),
        )


class TestResizeFrame(unittest.TestCase):
    def test_96_to_64(self):
        observation = np.zeros((96, 96, 3), dtype=np.uint8)
        resized = resize_frame(observation)
        self.assertEqual(resized.shape, (64, 64, 3))
        self.assertIsInstance(resized, np.ndarray)


if __name__ == "__main__":
    unittest.main()
