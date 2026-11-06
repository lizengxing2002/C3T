import unittest

import numpy as np

from c3t.metrics import evaluate_pair, nrmse, psnr, rmse, ssim, vif


def gradient(size=64, seed=0, scale=0.75):
    """Smooth signed ramp: adding 10 never clips, so closed forms stay exact."""
    rng = np.random.default_rng(seed)
    base = np.add.outer(np.linspace(0, 200 * scale, size), np.linspace(0, 40 * scale, size))
    return np.clip(base + rng.normal(0, 3, (size, size)), 0, 255).astype(np.uint8)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.reference = gradient()

    def test_identical_inputs_are_perfect(self):
        result = evaluate_pair(self.reference, self.reference)
        self.assertAlmostEqual(result['SSIM'], 1.0, places=10)
        self.assertEqual(result['RMSE'], 0.0)
        self.assertEqual(result['NRMSE'], 0.0)
        self.assertAlmostEqual(result['VIF'], 1.0, places=8)
        self.assertTrue(np.isinf(result['PSNR']))

    def test_noise_degrades_every_metric_monotonically(self):
        rng = np.random.default_rng(7)
        previous = None
        for sigma in (3.0, 10.0, 25.0):
            noisy = np.clip(
                self.reference.astype(float) + rng.normal(0, sigma, self.reference.shape), 0, 255
            ).astype(np.uint8)
            current = evaluate_pair(self.reference, noisy)
            if previous is not None:
                self.assertLess(current['PSNR'], previous['PSNR'])
                self.assertLess(current['SSIM'], previous['SSIM'])
                self.assertLess(current['VIF'], previous['VIF'])
                self.assertGreater(current['RMSE'], previous['RMSE'])
            previous = current

    def test_rmse_matches_closed_form(self):
        shifted = np.clip(self.reference.astype(int) + 10, 0, 255).astype(np.uint8)
        self.assertGreaterEqual(
            int(shifted.max()),
            int(self.reference.max()) + 10,
            'reference must not clip under the shift',
        )
        self.assertAlmostEqual(rmse(self.reference, shifted), 10.0, places=9)
        self.assertAlmostEqual(
            nrmse(self.reference, shifted),
            10.0 / float(int(self.reference.max()) - int(self.reference.min())),
            places=9,
        )

    def test_vif_is_bounded_and_rewards_agreement(self):
        unrelated = gradient(seed=31)[::-1, ::-1]
        score = vif(self.reference, unrelated)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.9)
        self.assertGreater(vif(self.reference, self.reference.copy()), 0.999)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_pair(self.reference, self.reference[:-1])

    def test_small_images_rejected_by_vif(self):
        tiny = np.zeros((9, 9), dtype=np.uint8)
        with self.assertRaises(ValueError):
            vif(tiny, tiny)

    def test_constant_reference_rejected_by_nrmse(self):
        flat = np.full((32, 32), 7, dtype=np.uint8)
        with self.assertRaises(ValueError):
            nrmse(flat, flat)

    def test_psnr_zero_for_maximum_error(self):
        self.assertAlmostEqual(
            psnr(np.zeros((16, 16), np.uint8), np.full((16, 16), 255, np.uint8)), 0.0, places=6
        )

    def test_ssim_is_bounded(self):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 256, self.reference.shape, dtype=np.uint8)
        value = ssim(self.reference, noise)
        self.assertLessEqual(value, 1.0)
        self.assertGreaterEqual(value, -1.0)


if __name__ == '__main__':
    unittest.main()
