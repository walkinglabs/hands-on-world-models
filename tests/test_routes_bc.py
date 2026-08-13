import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "安装 requirements-neural.txt 后运行路线 B/C")
class RouteBCNeuralTest(unittest.TestCase):
    def test_vq_video_shapes_gradient_and_action(self):
        import torch

        from hwm.data import make_pixelworld_dataset
        from hwm.video import (
            ActionTokenTransformer,
            TinyVQVAE,
            add_independent_frame_noise,
            diffusion_prediction_target,
            foreground_weighted_mse,
            motion_direction_accuracy,
            psnr,
            red_centers,
            rollout_token_model,
            video_batch_from_episodes,
        )

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
        no_action = ActionTokenTransformer(
            codebook_size=16, model_size=32, action_injection="none"
        )
        same_logits = no_action(same, torch.tensor([1, 2]))
        self.assertTrue(torch.equal(same_logits[0], same_logits[1]))
        film = ActionTokenTransformer(
            codebook_size=16, model_size=32, action_injection="film"
        )
        self.assertEqual(film(same, torch.tensor([1, 2])).shape, logits.shape)
        self.assertTrue(torch.isfinite(red_centers(current)).all())
        self.assertTrue(torch.isfinite(foreground_weighted_mse(current, following)))
        perfect_direction = motion_direction_accuracy(current, following, following)
        self.assertAlmostEqual(float(perfect_direction), 1.0)
        self.assertGreater(float(psnr(following, following)), 100.0)

        clip = following[:6].reshape(2, 3, *following.shape[1:])
        levels = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.4, 0.8]])
        noisy, noise = add_independent_frame_noise(clip, levels)
        self.assertEqual(noisy.shape, clip.shape)
        for target in ("x", "epsilon", "v"):
            self.assertEqual(
                diffusion_prediction_target(clip, noise, levels, target).shape,
                clip.shape,
            )

        frames = rollout_token_model(
            model,
            tokenizer,
            current_tokens[:1],
            [torch.tensor([2]), torch.tensor([2])],
            token_shape=(4, 4),
        )
        self.assertEqual(frames.shape[0], 3)

    def test_jepa_shapes_ema_probe_and_action(self):
        import torch

        from hwm.data import make_pixelworld_dataset
        from hwm.jepa import (
            TinyVideoJEPA,
            apply_linear_probe,
            feature_spread,
            fit_linear_probe,
            fit_linear_probe_weights,
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
        weights = fit_linear_probe_weights(pooled[:-1], positions[:-1])
        held_out = apply_linear_probe(pooled[-1:], weights)
        self.assertEqual(held_out.shape, (1, 2))
        same_video = video[:1].expand(2, -1, -1, -1, -1)
        prediction, _, _ = model(same_video, torch.tensor([1, 2]))
        difference = (prediction[0] - prediction[1]).abs().mean().detach()
        self.assertGreater(float(difference), 0.0)


if __name__ == "__main__":
    unittest.main()
