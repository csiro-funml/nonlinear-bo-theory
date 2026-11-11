import unittest

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from ts_variance_decay_experiment import (
    MLP,
    Config,
    direct_ridge_variance,
    make_parameter_kernel,
    matern_variance,
    resolve_seeds,
)


class Checks(unittest.TestCase):
    def test_regularizer_has_gradient(self):
        m = MLP(1, 4, 1)
        theta = parameters_to_vector(m.parameters())
        anchor = theta.detach().clone()
        kernel = make_parameter_kernel("rbf", 2.0, 1.5, 1.0)
        loss = -kernel(theta, anchor)
        loss.backward()
        self.assertIsNotNone(next(m.parameters()).grad)

    def test_supported_regularizer_kernels_have_gradients(self):
        theta = np.array([0.2, -0.4])
        for kernel in ("linear", "rbf", "matern", "rational_quadratic"):
            values = torch.tensor(theta, dtype=torch.float32, requires_grad=True)
            anchor = torch.zeros_like(values)
            kernel_function = make_parameter_kernel(kernel, 2.0, 1.5, 1.0)
            loss = -kernel_function(values, anchor)
            loss.backward()
            self.assertIsNotNone(values.grad)

        linear = make_parameter_kernel("linear", 2.0, 1.5, 1.0)
        self.assertAlmostEqual(linear(torch.tensor(theta), torch.zeros(2)).item(), 0.1)

    def test_duplicate_conditioning(self):
        K = np.array([[1, 0.5, 0.5], [0.5, 1, 1], [0.5, 1, 1.0]])
        v = direct_ridge_variance(K, np.array([0, 1, 2]), 1, 0.5)
        self.assertGreaterEqual(v, -1e-10)
        self.assertLess(v, 1)

    def test_gp_empty_and_duplicate(self):
        c = Config()
        g = np.linspace(-1, 1, 5)[:, None]
        self.assertAlmostEqual(
            matern_variance(c, np.empty((0, 1)), g[2:3])[0], 1, places=5
        )
        v = matern_variance(c, np.array([[0.0], [0.0]]), g[2:3])[0]
        self.assertGreaterEqual(v, 0)

    def test_master_seed_derives_reproducible_streams(self):
        first = Config(seed=7, num_runs=3)
        second = Config(seed=7, num_runs=3)
        resolve_seeds(first)
        resolve_seeds(second)
        self.assertEqual(first.seeds, second.seeds)
        self.assertEqual(first.initial_seed, second.initial_seed)
        self.assertEqual(first.teacher_seed, second.teacher_seed)
        self.assertEqual(first.noise_seed, second.noise_seed)

    def test_explicit_seeds_are_preserved(self):
        config = Config(
            seed=7,
            num_runs=3,
            seeds=[11, 12],
            teacher_seed=13,
            initial_seed=14,
            noise_seed=15,
        )
        resolve_seeds(config)
        self.assertEqual(config.seeds, [11, 12])
        self.assertEqual(config.teacher_seed, 13)
        self.assertEqual(config.initial_seed, 14)
        self.assertEqual(config.noise_seed, 15)


if __name__ == "__main__":
    unittest.main()
