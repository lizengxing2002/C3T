import inspect, unittest
import torch
from c3t.model import C3T, ModelConfig, PrecisionHodge, haar, ihaar, inverse_diagonal_variance
from c3t.losses import objective

torch.set_num_threads(2)


class MathematicalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_haar_roundtrip(self):
        for level in [0, 1, 2, 3]:
            x = torch.randn(2, 3, 32, 48, dtype=torch.double)
            self.assertLess((ihaar(haar(x, level), level) - x).abs().max().item(), 1e-12)

    def test_haar_parseval(self):
        x = torch.randn(2, 1, 32, 32, dtype=torch.double)
        self.assertAlmostEqual(x.square().sum().item(), haar(x, 2).square().sum().item(), places=9)

    def test_projection_curl_and_idempotence(self):
        d = torch.randn(2, 3, 4, 3, 3, dtype=torch.double)
        lv = torch.randn_like(d)
        m = PrecisionHodge()
        u, p, v, w = m(d, lv)
        self.assertLess((p[:, 0] + p[:, 2] - p[:, 1]).abs().max().item(), 1e-12)
        self.assertLess((m(p, lv)[1] - p).abs().max().item(), 1e-12)
        self.assertTrue(bool((v > 0).all()))

    def test_weighted_normal_equations(self):
        d = torch.randn(2, 3, 4, 3, 3, dtype=torch.double)
        lv = torch.randn_like(d)
        _, p, _, w = PrecisionHodge()(d, lv)
        r = w * (d - p)
        self.assertLess((r[:, 0] - r[:, 2]).abs().max().item(), 1e-11)
        self.assertLess((r[:, 1] + r[:, 2]).abs().max().item(), 1e-11)

    def test_pythagorean_error_nonincrease(self):
        z = torch.randn(2, 3, 4, 3, 3, dtype=torch.double)
        truth = torch.stack((z[:, 1] - z[:, 0], z[:, 2] - z[:, 0], z[:, 2] - z[:, 1]), 1)
        d = truth + torch.randn_like(truth)
        lv = torch.randn_like(d)
        _, p, _, w = PrecisionHodge()(d, lv)
        before = (w * (d - truth).square()).sum()
        after = (w * (p - truth).square()).sum()
        removed = (w * (d - p).square()).sum()
        self.assertLess(abs((before - after - removed).item()), 1e-9)

    def test_gradcheck_projector(self):
        d = torch.randn(1, 3, 1, 2, 2, dtype=torch.double, requires_grad=True)
        lv = torch.zeros_like(d)
        self.assertTrue(torch.autograd.gradcheck(lambda x: PrecisionHodge()(x, lv)[1], (d,)))

    def test_diagonal_variance(self):
        v = torch.ones(1, 16, 2, 2)
        self.assertTrue(torch.equal(inverse_diagonal_variance(v, 2), torch.ones(1, 1, 8, 8)))

    def test_signed_wavelets_allow_contrast_reversal(self):
        x = torch.rand(1, 1, 32, 32)
        self.assertLess((ihaar(-haar(x, 2), 2) + x).abs().max().item(), 1e-6)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_no_target_input(self):
        self.assertEqual(list(inspect.signature(C3T.forward).parameters), ['self', 'x', 'source'])

    def test_shape_identity_and_fixed_anchor_path(self):
        net = C3T(ModelConfig(width=8))
        torch.nn.init.normal_(net.proposal.weight, std=0.03)
        x = torch.rand(3, 1, 32, 48)
        s = torch.arange(3)
        o = net(x, s)
        self.assertEqual(o['predictions'].shape, (3, 3, 32, 48))
        self.assertLess((C3T.select(o, s) - x).abs().max().item(), 1e-7)
        self.assertLess(
            (o['edges'][:, 0] + o['edges'][:, 2] - o['edges'][:, 1]).abs().max().item(), 1e-5
        )

    def test_all_variants_backward(self):
        for cfg in [
            ModelConfig(width=8),
            ModelConfig(width=8, project=False),
            ModelConfig(width=8, precision_weighted=False),
            ModelConfig(width=8, wavelet_levels=0, signed_evidence=False),
        ]:
            net = C3T(cfg)
            x = torch.rand(2, 1, 32, 32)
            s = torch.tensor([0, 2])
            t = torch.tensor([1, 0])
            y = 0.7 * x + 0.1
            loss, _ = objective(net(x, s), x, y, t)
            loss.backward()
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_toy_fit_reduces_pixel_error(self):
        net = C3T(ModelConfig(width=8, global_context=False))
        opt = torch.optim.Adam(net.parameters(), lr=0.002)
        x = torch.rand(2, 1, 32, 32)
        s = torch.tensor([0, 2])
        t = torch.tensor([1, 0])
        y = 0.6 * x + 0.1
        errors = []
        for _ in range(20):
            opt.zero_grad()
            o = net(x, s)
            loss = (C3T.select(o, t) - y).square().mean()
            errors.append(loss.item())
            loss.backward()
            opt.step()
        self.assertLess(errors[-1], errors[0] * 0.6)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            C3T(ModelConfig(width=8))(torch.zeros(1, 2, 32, 32), torch.tensor([0]))


if __name__ == '__main__':
    unittest.main()
