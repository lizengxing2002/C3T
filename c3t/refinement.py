"""Graph-consistent multiscale potential refinement (GCMR).
A minor source-only refinement, not a new Hodge or Laplacian-pyramid theorem.
Predicting node potentials rather than independent target residuals preserves
fixed-anchor graph consistency. Gates use detached *estimated* confidence,
not a claim of calibrated clinical uncertainty.
"""

import torch
from torch import nn
import torch.nn.functional as F


def smooth(x, k=5):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode='reflect'), k, stride=1)


def node_to_edges(nodes):
    if nodes.ndim != 4 or nodes.shape[1] != 3:
        raise ValueError('Expected B,3,H,W modality potentials')
    return torch.stack(
        (nodes[:, 1] - nodes[:, 0], nodes[:, 2] - nodes[:, 0], nodes[:, 2] - nodes[:, 1]), 1
    )


class GraphConsistentRefinement(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.stem = nn.Conv2d(width + 9, width, 1)
        self.branches = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=d, dilation=d, groups=width) for d in (1, 2, 4)]
        )
        self.mix = nn.Sequential(
            nn.Conv2d(3 * width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(width, 6, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, source, edge_images, edge_variance):
        # Retain all full-resolution decoder features; no additional target image input.
        sx = source.float()
        v = edge_variance.float().detach().clamp_min(1e-6)
        anatomy = torch.cat((sx, sx - smooth(sx, 3), sx - smooth(sx, 7)), 1)
        h = F.gelu(self.stem(torch.cat((features, anatomy, edge_images, v.log().clamp(-8, 3)), 1)))
        h = self.mix(torch.cat([F.gelu(layer(h)) for layer in self.branches], 1))
        low, detail = self.output(h).float().chunk(2, 1)
        confidence = 1 / (1 + v.mean(1, keepdim=True).sqrt())
        # A bounded smooth contrast correction plus confidence-gated fine detail.
        nodes = 0.2 * smooth(torch.tanh(low), 5) + 0.1 * confidence * (
            torch.tanh(detail) - smooth(torch.tanh(detail), 5)
        )
        return node_to_edges(nodes), nodes, confidence
