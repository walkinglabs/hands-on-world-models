import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "安装 requirements-neural.txt 后运行路线 B/C")
class RouteBCNeuralTest(unittest.TestCase):
    def test_vq_video_shapes_gradient_and_action(self):
        import torch

        from hwm.data import make_pixelworld_dataset
        from hwm.video import ActionTokenTransformer, TinyVQVAE, video_batch_from_episodes

        torch.manual_seed(0)
        episodes = make_pixelworld_dataset(2, 4, seed=0)
        current, actions, following = video_batch_from_episodes(episodes)
        tokenizer = TinyVQVAE(codebook_size=16, embedding_size=8)
        output = tokenizer(current)
        output["loss"].backward()
        self.assertEqual(tuple(output["tokens"].shape[1:]), (4, 4))
        self.assertIsNotNone(tokenizer.encoder[0].weight.grad)

        with torch.no_grad():
            current_tokens = tokenizer.encode_tokens(current).flatten(1)
            next_tokens = tokenizer.encode_tokens(following).flatten(1)
        model = ActionTokenTransformer(codebook_size=16, model_size=32)
        loss = model.loss(current_tokens, actions, next_tokens)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        same = current_tokens[:1].expand(2, -1)
        logits = model(same, torch.tensor([1, 2]))
        difference = (logits[0] - logits[1]).abs().mean().detach()
        self.assertGreater(float(difference), 0.0)

    def test_jepa_shapes_ema_probe_and_action(self):
        import torch

        from hwm.data import make_pixelworld_dataset
        from hwm.jepa import (
            TinyVideoJEPA,
            feature_spread,
            fit_linear_probe,
            jepa_batch_from_episodes,
        )

        torch.manual_seed(0)
        episodes = make_pixelworld_dataset(3, 5, seed=1)
        video, actions, positions = jepa_batch_from_episodes(episodes)
        model = TinyVideoJEPA(feature_size=16)
        loss, prediction, target, features = model.loss(video, actions)
        loss.backward()
        self.assertEqual(prediction.shape, target.shape)
        self.assertGreater(float(feature_spread(features).detach()), 0.0)
        before = next(model.target_encoder.parameters()).clone()
        with torch.no_grad():
            next(model.online_encoder.parameters()).add_(0.1)
        model.update_target(momentum=0.5)
        after = next(model.target_encoder.parameters())
        self.assertFalse(torch.equal(before, after))
        pooled = target.mean(dim=1)
        estimate = fit_linear_probe(pooled, positions)
        self.assertEqual(estimate.shape, positions.shape)
        same_video = video[:1].expand(2, -1, -1, -1, -1)
        prediction, _, _ = model(same_video, torch.tensor([1, 2]))
        difference = (prediction[0] - prediction[1]).abs().mean().detach()
        self.assertGreater(float(difference), 0.0)


if __name__ == "__main__":
    unittest.main()
