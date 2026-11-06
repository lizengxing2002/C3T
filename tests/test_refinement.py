import unittest

import torch

from c3t.losses import objective
from c3t.model import C3T, ModelConfig
from c3t.refinement import GraphConsistentRefinement, node_to_edges


class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_node_edges_zero_curl(self):
        e = node_to_edges(torch.randn(2, 3, 32, 32))
        self.assertLess(float((e[:, 0] + e[:, 2] - e[:, 1]).abs().max()), 1e-6)

    def test_initially_matches_original_exactly(self):
        torch.manual_seed(5)
        a = C3T(ModelConfig(width=16))
        torch.manual_seed(5)
        b = C3T(ModelConfig(width=16, potential_refinement=True))
        for n, p in a.named_parameters():
            self.assertTrue(torch.equal(p, dict(b.named_parameters())[n]))
        x = torch.rand(2, 1, 32, 32)
        s = torch.tensor([0, 2])
        self.assertTrue(torch.equal(a(x, s)['predictions'], b(x, s)['predictions']))

    def test_nonzero_refinement_preserves_fixed_anchor_algebra(self):
        m = C3T(ModelConfig(width=16, potential_refinement=True))
        torch.nn.init.normal_(m.refinement.output.weight, std=0.1)
        x = torch.rand(3, 1, 32, 32)
        s = torch.tensor([0, 1, 2])
        o = m(x, s)
        e = o['edges']
        self.assertLess(float((e[:, 0] + e[:, 2] - e[:, 1]).abs().max()), 2e-6)
        self.assertTrue(torch.equal(C3T.select(o, s), x))
        self.assertGreater(float(o['refinement_nodes'].abs().sum()), 0)

    def test_finite_supervised_backward(self):
        m = C3T(ModelConfig(width=16, potential_refinement=True))
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand_like(x)
        s = torch.tensor([0, 1])
        t = torch.tensor([2, 0])
        loss, _ = objective(m(x, s), x, y, t)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(m.refinement.output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_confidence_bounded_and_detached(self):
        m = GraphConsistentRefinement(16)
        v = torch.rand(2, 3, 32, 32, requires_grad=True)
        _, _, g = m(
            torch.rand(2, 16, 32, 32), torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32), v
        )
        self.assertTrue(bool(((g > 0) & (g <= 1)).all()))
        self.assertFalse(g.requires_grad)

    def test_invalid_potentials_rejected(self):
        with self.assertRaises(ValueError):
            node_to_edges(torch.rand(2, 2, 32, 32))


if __name__ == '__main__':
    unittest.main()
