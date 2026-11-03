import unittest

import numpy as np

from examples.toy_reproduction import (
    Config,
    apply_patch,
    evaluate_patch,
    make_scenes,
    optimize_patch,
)


class ToyReproductionTests(unittest.TestCase):
    def test_patch_is_composited_in_bottom_left_corner(self):
        images = np.zeros((2, 3, 12, 12), dtype=np.float64)
        patch = np.ones((3, 4, 4), dtype=np.float64)
        output = apply_patch(images, patch)
        np.testing.assert_allclose(output[:, :, -4:, :4], 1.0)
        np.testing.assert_allclose(output[:, :, :-4, :], 0.0)

    def test_cot_optimization_hijacks_held_out_layouts(self):
        config = Config(steps=20, batch_size=16, eval_scenes=64)
        train_rng = np.random.default_rng(config.seed)
        eval_rng = np.random.default_rng(config.seed + 1)
        _, train_positions = make_scenes(config, 25, train_rng)
        _, eval_positions = make_scenes(config, config.eval_scenes, eval_rng)
        patch, _ = optimize_patch("cot", config, train_positions, np.random.default_rng(config.seed + 2))
        metrics = evaluate_patch(patch, config, eval_positions, config.seed + 3)
        self.assertGreaterEqual(metrics["cot_asr"], 0.95)
        self.assertGreaterEqual(metrics["action_asr"], 0.95)


if __name__ == "__main__":
    unittest.main()
