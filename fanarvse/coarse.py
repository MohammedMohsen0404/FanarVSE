"""Coarse cells: spherical k-means on a sample, then every vector is assigned to its best cell."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def spherical_kmeans(host: torch.Tensor, n_cells: int, sample: int, iters: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """Unit-norm centroids (n_cells, d) trained on a random sample of the rows of `host` (CPU, fp16)."""
    g = torch.Generator().manual_seed(seed)
    n = len(host)
    idx = torch.randperm(n, generator=g)[: min(sample, n)]
    samp = host[idx].to(device).float()
    if len(samp) < n_cells:
        raise ValueError(f"{n_cells} cells need at least as many vectors (got {len(samp)})")
    cen = samp[torch.randperm(len(samp), generator=g)[:n_cells].to(device)].clone()
    for _ in range(iters):
        a = assign_rows(samp, cen)
        cen = F.normalize(torch.zeros_like(cen).index_add_(0, a, samp) + 1e-9, dim=1)
    return cen


@torch.no_grad()
def assign_rows(X: torch.Tensor, cen: torch.Tensor, chunk: int = 50_000) -> torch.Tensor:
    return torch.cat([(X[j : j + chunk] @ cen.T).argmax(1) for j in range(0, len(X), chunk)])


@torch.no_grad()
def assign(host: torch.Tensor, cen: torch.Tensor, chunk: int = 100_000) -> torch.Tensor:
    """Best cell of every row of `host` (CPU), computed on the GPU in chunks."""
    dev = cen.device
    return torch.cat([(host[j : j + chunk].to(dev).float() @ cen.T).argmax(1) for j in range(0, len(host), chunk)])
