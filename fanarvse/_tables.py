"""Lloyd-Max quantizers of a unit Gaussian (codebooks of the refinement code) and their distortions."""

from __future__ import annotations

import functools
import math

import torch


@functools.cache
def lloyd_max(bits: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Optimal scalar quantizer of N(0, 1) with 2**bits levels.

    Returns (levels, thresholds, mse) on the CPU in float32; computed by Lloyd iterations on a fine
    quadrature grid, so the result is deterministic.
    """
    if bits == 0:
        return torch.zeros(1), torch.zeros(0), 1.0
    x = torch.linspace(-9.0, 9.0, 400_001, dtype=torch.float64)
    w = torch.exp(-0.5 * x * x)
    w = w / w.sum()
    n = 2**bits
    # start from the Gaussian quantiles of the cell midpoints
    levels = math.sqrt(2.0) * torch.special.erfinv(2.0 * (torch.arange(n, dtype=torch.float64) + 0.5) / n - 1.0)
    for _ in range(500):
        th = (levels[1:] + levels[:-1]) / 2
        idx = torch.bucketize(x, th)
        mass = torch.zeros(n, dtype=torch.float64).index_add_(0, idx, w)
        new = torch.zeros(n, dtype=torch.float64).index_add_(0, idx, w * x) / mass.clamp(min=1e-300)
        new = (new - new.flip(0)) / 2  # the optimum is symmetric; this removes the bias of grid points that sit on a threshold
        if torch.allclose(new, levels, atol=1e-12, rtol=0):
            levels = new
            break
        levels = new
    th = (levels[1:] + levels[:-1]) / 2
    mse = float((w * (x - levels[torch.bucketize(x, th)]) ** 2).sum())
    return levels.float(), th.float(), mse


def distortion(bits: int) -> float:
    return lloyd_max(bits)[2]
