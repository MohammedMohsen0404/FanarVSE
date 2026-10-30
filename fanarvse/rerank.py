"""Stage 3: exact scores of the ambiguous candidates, read from pinned host memory by the GPU (zero-copy)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


class _HostPtr:
    """Lets Triton take a pinned CPU tensor as a pointer (unified virtual addressing)."""

    def __init__(self, t: torch.Tensor):
        self.t, self.dtype = t, t.dtype

    def data_ptr(self) -> int:
        return self.t.data_ptr()


def pin(host: torch.Tensor, chunk: int = 100_000) -> torch.Tensor:
    H = torch.empty(host.shape, dtype=host.dtype, pin_memory=True)
    for j in range(0, len(host), chunk):
        H[j : j + chunk].copy_(host[j : j + chunk])
    return H


@triton.jit
def _exact_kernel(h_ptr, ids_ptr, amb_ptr, q_ptr, score_ptr, KC, D: tl.constexpr, BD: tl.constexpr):
    # score[q, i] = <x_ids[q, i], q> for flagged candidates; one program per candidate, the others exit at once
    i = tl.program_id(0)
    q = tl.program_id(1)
    if tl.load(amb_ptr + q * KC + i) != 0:
        r = tl.load(ids_ptr + q * KC + i).to(tl.int64)
        acc = tl.zeros((BD,), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BD):
            j = d0 + tl.arange(0, BD)
            jm = j < D
            acc += tl.load(h_ptr + r * D + j, mask=jm, other=0.0).to(tl.float32) * tl.load(q_ptr + q * D + j, mask=jm, other=0.0)
        tl.store(score_ptr + q * KC + i, tl.sum(acc, axis=0))


def exact_scores(H: torch.Tensor, ids: torch.Tensor, flag: torch.Tensor, Q: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Writes <x_ids, q> into out where flag is set (ids are original row ids of the pinned matrix H)."""
    B, KC = ids.shape
    _exact_kernel[(KC, B)](
        _HostPtr(H), ids.contiguous(), flag.to(torch.int8).contiguous(), Q.float().contiguous(), out, KC, D=Q.shape[1], BD=512
    )
    return out
