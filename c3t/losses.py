"""Training objective for C3T.

``objective`` returns the weighted total and its individual terms. ``ssim`` and
``gradient_loss`` are also used directly by callers that only need one term.
"""

import torch
import torch.nn.functional as F

from .model import C3T, haar


def ssim(x, y):
    # Casting tensors alone does not defeat autocast in convolution. Small local
    # variances require FP32 to avoid cancellation/quantized SSIM gradients.
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _ssim_fp32(x.float(), y.float())


def _ssim_fp32(x, y):
    # Differentiable local 11x11 Gaussian SSIM, data_range=1.
    x = x.float()
    y = y.float()
    v = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-v.square() / (2 * 1.5**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None]
    mx = F.conv2d(x, k)
    my = F.conv2d(y, k)
    vx = (F.conv2d(x * x, k) - mx * mx).clamp_min(0)
    vy = (F.conv2d(y * y, k) - my * my).clamp_min(0)
    xy = F.conv2d(x * y, k) - mx * my
    return (
        ((2 * mx * my + 0.01**2) * (2 * xy + 0.03**2))
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


def gradient_loss(x, y):
    return F.l1_loss(x[..., 1:, :] - x[..., :-1, :], y[..., 1:, :] - y[..., :-1, :]) + F.l1_loss(
        x[..., :, 1:] - x[..., :, :-1], y[..., :, 1:] - y[..., :, :-1]
    )


def objective(result, x, y, target, weights=None):
    w = {
        'l1': 1.0,
        'mse': 0.5,
        'ssim': 0.25,
        'gradient': 0.1,
        'wavelet': 0.05,
        'raw': 0.1,
        'nll': 0.005,
        'curl': 0.01,
    }
    if weights:
        w.update(weights)
    p = C3T.select(result, target)
    raw = C3T.select(result, target, 'raw_predictions')
    var = C3T.select(result, target, 'raw_variance').clamp_min(1e-5)
    # Target information is used only for supervised losses, never network conditioning.
    mask = 0.25 + 0.75 * ((x.detach() > 1 / 255) | (y.detach() > 1 / 255)).float()
    error = (p - y).float()
    norm = mask.mean().clamp_min(0.1)
    terms = {
        'l1': (mask * torch.sqrt(error.square() + 1e-6)).mean() / norm,
        'mse': (mask * error.square()).mean() / norm,
        'ssim': 1 - ssim(p, y),
        'gradient': gradient_loss(p, y),
        'wavelet': F.l1_loss(haar(p, 2), haar(y, 2)),
        'raw': F.l1_loss(raw, y),
        # The squared residual is a fitted target, so stop-gradient leaves log-variance
        # as the only differentiable variable: it is supervised purely as a precision.
        'nll': 0.5 * ((raw - y).square().detach() / var + var.log()).mean(),
        'curl': (
            result['raw_wavelet_edges'][:, 0]
            + result['raw_wavelet_edges'][:, 2]
            - result['raw_wavelet_edges'][:, 1]
        )
        .square()
        .mean(),
    }
    total = sum(w[k] * v for k, v in terms.items())
    return total, terms


def anchor_consistency(a, b):
    # Paired views of the same anatomy should predict compatible contrast differences.
    return F.smooth_l1_loss(a['edges'], b['edges'])
