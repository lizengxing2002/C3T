"""C3T: source-anchored conservative cross-contrast transport.

An anchor is ONE observed image and its modality. The network estimates edge
residual evidence on the T1/T2/FLAIR graph, projects it onto graph gradients,
and reconstructs all contrasts relative to the observed anchor. No target is
accepted by forward(). Algebraic path consistency applies to a fixed anchor,
unclipped floating-point outputs; it is not an invertibility/anatomy theorem.
"""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .refinement import GraphConsistentRefinement

MODALITIES = ('T1', 'T2', 'FLAIR')
EDGES = ((0, 1), (0, 2), (1, 2))


@dataclass
class ModelConfig:
    width: int = 32
    wavelet_levels: int = 2
    project: bool = True
    precision_weighted: bool = True
    signed_evidence: bool = True
    global_context: bool = True
    potential_refinement: bool = False

    def __post_init__(self):
        if self.width < 8 or self.width % 8:
            raise ValueError('width must be a positive multiple of 8')
        if not 0 <= self.wavelet_levels <= 3:
            raise ValueError('wavelet_levels must be 0..3')


def haar(x: Tensor, levels: int = 1) -> Tensor:
    """Orthonormal full wavelet packet; retains every coefficient, no FFT phase assumption."""
    for _ in range(levels):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError('Haar requires even spatial dimensions')
        a, b, c, d = x[..., 0::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        x = torch.cat((a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d), dim=1) * 0.5
    return x


def ihaar(x: Tensor, levels: int = 1) -> Tensor:
    for _ in range(levels):
        if x.shape[1] % 4:
            raise ValueError('Invalid wavelet channel count')
        l, h, v, d = x.chunk(4, dim=1)
        a, b, c, e = (
            (l + h + v + d) * 0.5,
            (l - h + v - d) * 0.5,
            (l + h - v - d) * 0.5,
            (l - h - v + d) * 0.5,
        )
        # Differentiable spatial interleaving.
        top = torch.stack((a, b), dim=-1).flatten(-2)
        bottom = torch.stack((c, e), dim=-1).flatten(-2)
        x = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    return x


def inverse_diagonal_variance(v: Tensor, levels: int) -> Tensor:
    """Marginal pixel variance under the explicit independent-wavelet-error approximation."""
    for _ in range(levels):
        a, b, c, d = v.chunk(4, dim=1)
        v = ((a + b + c + d) * 0.25).repeat_interleave(2, -2).repeat_interleave(2, -1)
    return v


class PrecisionHodge(nn.Module):
    """Pixel/subband-wise weighted graph least squares with gauge u_T1=0.

    Incidence (gauge column removed) is [[1,0],[0,1],[-1,1]].
    The positive determinant uses a sum of products (avoids cancellation).
    P_W d is the W-orthogonal projection; this standard algebra is not claimed
    as newly invented. The learnable evidence and MRI anchoring are the proposal.
    """

    def forward(self, d: Tensor, logvar: Tensor, weighted: bool = True):
        if d.shape != logvar.shape or d.shape[1] != 3:
            raise ValueError('Expected B,3,K,H,W evidence and logvar')
        # FP32 solve during AMP; retain float64 for gradcheck/math validation.
        dt = torch.float64 if d.dtype == torch.float64 else torch.float32
        d = d.to(dt)
        lv = logvar.to(dt).clamp(-6, 3)
        w = (-lv).exp().detach() if weighted else torch.ones_like(d)
        w0, w1, w2 = w.unbind(1)
        d0, d1, d2 = d.unbind(1)
        A = w0 + w2
        C = w1 + w2
        det = w0 * w1 + w0 * w2 + w1 * w2
        r1 = w0 * d0 - w2 * d2
        r2 = w1 * d1 + w2 * d2
        u1 = (C * r1 + w2 * r2) / det
        u2 = (w2 * r1 + A * r2) / det
        potentials = torch.stack((torch.zeros_like(u1), u1, u2), dim=1)
        projected = torch.stack((u1, u2, u2 - u1), dim=1)
        # P diag(exp(logvar)) P^T diagonal, valid also for unweighted ablation.
        # Rows of projection P (computed without a per-pixel matrix allocation).
        p00 = C * w0 / det
        p01 = w2 * w1 / det
        p02 = -w1 * w2 / det
        p10 = w2 * w0 / det
        p11 = A * w1 / det
        p12 = w0 * w2 / det
        var = lv.exp()
        v0, v1, v2 = var.unbind(1)
        ev0 = p00.square() * v0 + p01.square() * v1 + p02.square() * v2
        ev1 = p10.square() * v0 + p11.square() * v1 + p12.square() * v2
        ev2 = (p10 - p00).square() * v0 + (p11 - p01).square() * v1 + (p12 - p02).square() * v2
        edgevar = torch.stack((ev0, ev1, ev2), dim=1)
        return potentials, projected, edgevar, w


class ContrastBlock(nn.Module):
    def __init__(self, c: int, emb: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.film = nn.Linear(emb, 2 * c)
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.mix = nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.GELU(), nn.Conv2d(2 * c, c, 1))
        self.scale = nn.Parameter(torch.full((1, c, 1, 1), 0.1))

    def forward(self, x, e):
        s, b = self.film(e).chunk(2, dim=1)
        y = self.norm(x) * (1 + 0.1 * s[:, :, None, None]) + 0.1 * b[:, :, None, None]
        return x + self.scale * self.mix(self.dw(y))


class Context(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, 4, batch_first=True)
        self.pos = nn.Parameter(torch.randn(1, 64, c) * 0.01)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        tok = F.adaptive_avg_pool2d(x, (8, 8)).flatten(2).transpose(1, 2)
        q = self.norm(tok) + self.pos
        y = self.attn(q, q, q, need_weights=False)[0]
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], 8, 8)
        return x + self.scale * F.interpolate(
            y, size=x.shape[-2:], mode='bilinear', align_corners=False
        )


class Backbone(nn.Module):
    def __init__(self, w, context):
        super().__init__()
        self.embedding = nn.Embedding(3, 64)
        self.stem = nn.Conv2d(1, w, 3, padding=1)
        self.enc = nn.ModuleList(
            [nn.ModuleList([ContrastBlock(w * 2**i, 64) for _ in range(2)]) for i in range(4)]
        )
        self.down = nn.ModuleList(
            [nn.Conv2d(w * 2**i, w * 2 ** (i + 1), 4, 2, 1) for i in range(3)]
        )
        self.context = Context(w * 8) if context else nn.Identity()
        self.merge = nn.ModuleList(
            [nn.Conv2d(w * 2 ** (i + 1) + w * 2**i, w * 2**i, 1) for i in [2, 1, 0]]
        )
        self.dec = nn.ModuleList(
            [nn.ModuleList([ContrastBlock(w * 2**i, 64) for _ in range(2)]) for i in [2, 1, 0]]
        )

    def forward(self, x, source):
        e = self.embedding(source)
        x = self.stem(x)
        skips = []
        for i, blocks in enumerate(self.enc):
            for block in blocks:
                x = block(x, e)
            if i < 3:
                skips.append(x)
                x = self.down[i](x)
        x = self.context(x)
        anchor = x
        for merge, blocks, skip in zip(self.merge, self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = merge(torch.cat((x, skip), dim=1))
            for block in blocks:
                x = block(x, e)
        return x, anchor


class C3T(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        k = 4**c.wavelet_levels
        self.backbone = Backbone(c.width, c.global_context)
        self.proposal = nn.Conv2d(c.width, 3 * k, 3, padding=1)
        self.logvar = nn.Conv2d(c.width, 3 * k, 3, padding=1)
        self.evidence = nn.Conv2d(c.width, 3 * k, 3, padding=1) if c.signed_evidence else None
        self.projector = PrecisionHodge()
        nn.init.zeros_(self.proposal.weight)
        nn.init.zeros_(self.proposal.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -3)
        if self.evidence is not None:
            nn.init.zeros_(self.evidence.weight)
            nn.init.zeros_(self.evidence.bias)
        self.register_buffer('edge_lookup', torch.tensor([[0, 0, 1], [0, 0, 2], [1, 2, 0]]))
        self.refinement = GraphConsistentRefinement(c.width) if c.potential_refinement else None

    def forward(self, x: Tensor, source: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('Input must be B,1,H,W single observed contrast')
        if source.shape != (x.shape[0],) or source.dtype != torch.long:
            raise ValueError('source must be B int64 modality IDs')
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError('H,W must be multiples of 8; do not silently resize')
        if bool(((source < 0) | (source > 2)).any()):
            raise ValueError('Unknown modality')
        c = self.config
        b = x.shape[0]
        k = 4**c.wavelet_levels
        f, anchor = self.backbone(x, source)
        full_features = f
        wx = haar(x.float(), c.wavelet_levels)
        f = F.interpolate(f, size=wx.shape[-2:], mode='bilinear', align_corners=False)
        shape = (b, 3, k, *wx.shape[-2:])
        d = self.proposal(f).float().reshape(shape)
        lv = self.logvar(f).float().reshape(shape).clamp(-6, 3)
        if self.evidence is not None:
            gain = 2 * torch.tanh(self.evidence(f).float().reshape(shape))
            d = d + gain * wx[:, None]
        _, projected, ev, w = self.projector(d, lv, c.precision_weighted)
        refinement_nodes = None
        refinement_confidence = None
        if self.refinement is not None:
            coarse = projected if c.project else d
            coarse_var = ev if c.project else lv.exp()
            coarse_images = ihaar(coarse.flatten(0, 1), c.wavelet_levels).reshape(
                b, 3, *x.shape[-2:]
            )
            pixel_var = inverse_diagonal_variance(
                coarse_var.flatten(0, 1), c.wavelet_levels
            ).reshape(b, 3, *x.shape[-2:])
            correction, refinement_nodes, refinement_confidence = self.refinement(
                full_features, x, coarse_images, pixel_var
            )
            correction_w = haar(correction.flatten(0, 1)[:, None], c.wavelet_levels).reshape_as(d)
            # The correction is a graph gradient, so P_W leaves it untouched:
            # P_W(d + B*r) = P_W(d) + B*r and the covariance is unchanged.
            d = d + correction_w
            if c.project:
                projected = projected + correction_w
        effective = projected if c.project else d
        effective_var = ev if c.project else lv.exp()
        edge_images = ihaar(effective.flatten(0, 1), c.wavelet_levels).reshape(b, 3, *x.shape[-2:])
        raw_images = ihaar(d.flatten(0, 1), c.wavelet_levels).reshape(b, 3, *x.shape[-2:])
        vi = inverse_diagonal_variance(effective_var.flatten(0, 1), c.wavelet_levels).reshape(
            b, 3, *x.shape[-2:]
        )
        raw_vi = inverse_diagonal_variance(lv.exp().flatten(0, 1), c.wavelet_levels).reshape(
            b, 3, *x.shape[-2:]
        )
        pred = []
        raw_pred = []
        variance = []
        raw_variance = []
        batch = torch.arange(b, device=x.device)
        for target in range(3):
            tid = torch.full_like(source, target)
            idx = self.edge_lookup[source, tid]
            sign = torch.sign(tid - source).float()[:, None, None]
            pred.append(x[:, 0] + sign * edge_images[batch, idx])
            raw_pred.append(x[:, 0] + sign * raw_images[batch, idx])
            variance.append(vi[batch, idx] * sign.square())
            raw_variance.append(raw_vi[batch, idx] * sign.square())
        return {
            'predictions': torch.stack(pred, 1),
            'raw_predictions': torch.stack(raw_pred, 1),
            'variance': torch.stack(variance, 1),
            'raw_variance': torch.stack(raw_variance, 1),
            'edges': edge_images,
            'raw_edges': raw_images,
            'wavelet_edges': effective,
            'raw_wavelet_edges': d,
            'weights': w,
            'logvar': lv,
            'anchor_features': anchor,
            'refinement_nodes': refinement_nodes,
            'refinement_confidence': refinement_confidence,
        }

    @staticmethod
    def select(result: Dict[str, Tensor], target: Tensor, key='predictions') -> Tensor:
        batch = torch.arange(target.shape[0], device=target.device)
        return result[key][batch, target][:, None]
