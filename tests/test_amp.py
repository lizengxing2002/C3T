import unittest
import torch
from c3t.losses import ssim


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA-only mixed-precision regression')
class AMPNumericsTests(unittest.TestCase):
    def test_ssim_matches_fp32_under_autocast(self):
        torch.manual_seed(9)
        x = torch.rand(2, 1, 32, 32, device='cuda', requires_grad=True)
        y = (x.detach() * 0.9 + 0.04).clamp(0, 1)
        expected = ssim(x, y)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            actual = ssim(x, y)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertLess(abs(float(actual - expected)), 1e-7)
        (1 - actual).backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
