"""Image-quality metrics for synthesized slices.

Every entry point takes 2-D ``uint8`` arrays in [0, 255]. PSNR, SSIM and NRMSE
follow the standard definitions; ``vif`` is the pixel-domain implementation of
Sheikh and Bovik (four scales, ``sigma_nsq = 2``) so evaluation needs no
pretrained weights.
"""

import numpy as np
from scipy.signal import correlate2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

METRICS = ('PSNR', 'SSIM', 'RMSE', 'NRMSE', 'VIF')


def _as_float(image):
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError('expected a 2-D single-channel image')
    return array.astype(np.float64)


def psnr(truth, pred):
    return float(peak_signal_noise_ratio(truth, pred, data_range=255))


def ssim(truth, pred):
    return float(structural_similarity(truth, pred, data_range=255))


def rmse(truth, pred):
    a, b = _as_float(truth), _as_float(pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nrmse(truth, pred):
    a, b = _as_float(truth), _as_float(pred)
    spread = float(a.max() - a.min())
    if spread <= 0:
        raise ValueError('constant reference image has no NRMSE scale')
    return float(np.sqrt(np.mean((a - b) ** 2)) / spread)


def _window(size, sigma):
    axis = np.arange(size) - (size - 1) / 2
    g = np.exp(-(axis**2) / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def _filter(image, kernel):
    return correlate2d(image, kernel, mode='valid')


def vif(truth, pred, sigma_nsq=2.0):
    """Pixel-domain VIF: information fidelity summed over four scales."""
    a, b = _as_float(truth), _as_float(pred)
    if a.shape != b.shape:
        raise ValueError('VIF requires matching shapes')
    if min(a.shape) < 17 or min(b.shape) < 17:
        raise ValueError('VIF requires images of at least 17x17')
    eps = 1e-10
    numerator = 0.0
    denominator = 0.0
    for scale in range(1, 5):
        size = 2 ** (4 - scale + 1) + 1
        kernel = _window(size, size / 5.0)
        if scale > 1:
            a = _filter(a, kernel)[::2, ::2]
            b = _filter(b, kernel)[::2, ::2]
        mu_a = _filter(a, kernel)
        mu_b = _filter(b, kernel)
        var_a = np.clip(_filter(a * a, kernel) - mu_a * mu_a, 0, None)
        var_b = np.clip(_filter(b * b, kernel) - mu_b * mu_b, 0, None)
        cov = _filter(a * b, kernel) - mu_a * mu_b
        gain = cov / (var_a + eps)
        residual = var_b - gain * cov
        flat_a = var_a < eps
        var_b_flat = var_b[flat_a]
        gain[flat_a] = 0.0
        residual[flat_a] = var_b_flat
        var_a[flat_a] = 0.0
        flat_b = var_b < eps
        gain[flat_b] = 0.0
        residual[flat_b] = 0.0
        negative = gain < 0
        residual[negative] = var_b[negative]
        gain[negative] = 0.0
        residual[residual <= eps] = eps
        numerator += np.sum(np.log10(1 + gain**2 * var_a / (residual + sigma_nsq)))
        denominator += np.sum(np.log10(1 + var_a / sigma_nsq))
    if denominator <= 0:
        raise ValueError('degenerate reference image for VIF')
    return float(numerator / denominator)


def evaluate_pair(truth, pred):
    """All metrics for one synthesized slice against its reference."""
    if np.asarray(truth).shape != np.asarray(pred).shape:
        raise ValueError('prediction/reference shape mismatch')
    return {
        'PSNR': psnr(truth, pred),
        'SSIM': ssim(truth, pred),
        'RMSE': rmse(truth, pred),
        'NRMSE': nrmse(truth, pred),
        'VIF': vif(truth, pred),
    }
