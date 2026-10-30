"""Stage 2: tiered refinement code with error bars (paper, Section 5, Corollary 3).

The residual x - c (c: global mean) is expressed in the principal basis V of the data, and the coordinates are
split into groups of 32 that receive 1-8 bits each by greedy reverse water-filling. Consecutive groups with the
same width form a tier; each tier has its own Haar rotation, a Lloyd-Max scalar codebook, a per-vector scale and
an error scale sigma_t. The estimate is

    est = <q, c> + sum_t scale_t <q_t, codebook[idx_t]>,   sigma^2 = sum_t ||q_t||^2 sigma_t^2,

which is unbiased for any codebook after a Haar rotation (Theorem 1), with a sub-Gaussian tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

from ._tables import distortion, lloyd_max

G = 32  # coordinates per allocation group
MAX_TIERS = 6


@dataclass
class TieredCode:
    tiers: list  # [(start, end, bits)] in the principal basis
    planes: list  # per tier: (bits, N, (end - start) // 8) uint8 bit planes, cell-sorted order
    scale: torch.Tensor  # (T, N) fp16
    sig: torch.Tensor  # (T, N) fp16
    VP: torch.Tensor  # (d, d) fp32: principal basis times the block-diagonal tier rotations
    codebooks: list  # per tier: (2**bits,) fp32 Lloyd-Max levels
    center: torch.Tensor  # (d,) fp32 global mean c

    def bytes_per_vector(self) -> float:
        return sum((e - s) * b for s, e, b in self.tiers) / 8 + 4 * len(self.tiers)


def allocate(eigvals: torch.Tensor, total_bits: int, bmin: int = 1, bmax: int = 8) -> list:
    """Greedy bit allocation over groups of G principal directions; returns tiers [(start, end, bits)]."""
    d = len(eigvals)
    ng = d // G
    lamg = eigvals.reshape(ng, G).sum(1).cpu().numpy()
    b = np.full(ng, bmin)
    left = total_bits - bmin * d
    while left >= G:
        gain = [lamg[i] * (distortion(b[i]) - distortion(b[i] + 1)) if b[i] < bmax else -1 for i in range(ng)]
        i = int(np.argmax(gain))
        b[i] += 1
        left -= G
    tiers, st = [], 0
    for i in range(1, ng + 1):
        if i == ng or b[i] != b[st]:
            tiers.append((st * G, i * G, int(b[st])))
            st = i
    if len(tiers) > MAX_TIERS:
        raise ValueError(f"bit allocation produced {len(tiers)} tiers (at most {MAX_TIERS}); change refine_bits_per_dim")
    return tiers


_SH = None


def _pack(bit: torch.Tensor) -> torch.Tensor:
    global _SH
    if _SH is None or _SH.device != bit.device:
        _SH = torch.arange(7, -1, -1, device=bit.device, dtype=torch.uint8)
    return ((bit.reshape(bit.shape[0], -1, 8).to(torch.uint8)) << _SH).sum(2).to(torch.uint8)


@torch.no_grad()
def build(
    host: torch.Tensor, order: torch.Tensor, center: torch.Tensor, sample: torch.Tensor, total_bits: int, seed: int, chunk: int = 16384
) -> TieredCode:
    """Encode the rows host[order] (CPU fp16) in that order."""
    dev = center.device
    S = host[sample].to(dev).float() - center
    S = S / S.norm(dim=1, keepdim=True).clamp(min=1e-12)
    ev, V = torch.linalg.eigh(torch.cov(S.T))
    del S
    lam, V = ev.flip(0).clamp(min=1e-12), V.flip(1).contiguous()
    tiers = allocate(lam, total_bits)
    gen = torch.Generator(device=dev).manual_seed(100 + seed)
    rots = [torch.linalg.qr(torch.randn(e - s, e - s, device=dev, generator=gen))[0] for s, e, _ in tiers]
    books = {b: (lloyd_max(b)[0].to(dev), lloyd_max(b)[1].to(dev)) for b in {b for _, _, b in tiers}}
    n = len(order)
    planes = [torch.empty(b, n, (e - s) // 8, dtype=torch.uint8, device=dev) for s, e, b in tiers]
    scale = torch.empty(len(tiers), n, dtype=torch.float16, device=dev)
    sig = torch.empty(len(tiers), n, dtype=torch.float16, device=dev)
    for j in range(0, n, chunk):
        Z = (host[order[j : j + chunk]].to(dev).float() - center) @ V
        m = len(Z)
        for t, ((s, e, b), P) in enumerate(zip(tiers, rots, strict=True)):
            F = Z[:, s:e] @ P
            nf = F.norm(dim=1).clamp(min=1e-12)
            dt = e - s
            Fu = F / nf[:, None] * dt**0.5
            cb, th = books[b]
            q = torch.bucketize(Fu, th)
            Y = cb[q]
            yn = Y.norm(dim=1)
            ct = (Fu * Y).sum(1) / (yn * dt**0.5)
            for p in range(b):
                planes[t][p, j : j + m] = _pack(((q >> (b - 1 - p)) & 1).bool())
            scale[t, j : j + m] = (nf / (ct * yn)).half()
            sig[t, j : j + m] = (nf * torch.sqrt((1 - ct**2).clamp(min=0)) / (ct * (dt - 1) ** 0.5)).half()
    D = torch.zeros(V.shape[0], V.shape[0], device=dev)
    for (s, e, _), P in zip(tiers, rots, strict=True):
        D[s:e, s:e] = P
    VP = (V @ D).contiguous()
    return TieredCode(tiers, planes, scale, sig, VP, [books[b][0].float().contiguous() for _, _, b in tiers], center)


@triton.jit
def _tier(
    acc,
    var,
    q,
    rows,
    mmask,
    B,
    N,
    t,
    qk_ptr,
    planes_ptr,
    cb_ptr,
    scale_ptr,
    sig_ptr,
    qn2_ptr,
    NBITS: tl.constexpr,
    KB: tl.constexpr,
    KP: tl.constexpr,
    BM: tl.constexpr,
):
    cj = tl.arange(0, KP)
    cmask = cj < KB
    m2 = mmask[:, None] & cmask[None, :]
    a = tl.zeros((BM,), dtype=tl.float32)
    for k in tl.static_range(8):
        code = tl.zeros((BM, KP), dtype=tl.int32)
        for p in tl.static_range(NBITS):
            byt = tl.load(planes_ptr + p * N * KB + rows[:, None] * KB + cj[None, :], mask=m2, other=0)
            code += (((byt >> (7 - k)) & 1).to(tl.int32)) << (NBITS - 1 - p)
        val = tl.load(cb_ptr + code)
        qrow = tl.load(qk_ptr + k * B * KB + q * KB + cj, mask=cmask, other=0.0)
        a += tl.sum(tl.where(m2, val * qrow[None, :], 0.0), axis=1)
    s = tl.load(scale_ptr + t * N + rows, mask=mmask, other=0.0).to(tl.float32)
    g = tl.load(sig_ptr + t * N + rows, mask=mmask, other=0.0).to(tl.float32)
    return acc + a * s, var + tl.load(qn2_ptr + t * B + q) * g * g


@triton.jit
def _score_kernel(
    pos_ptr,
    qc_ptr,
    est_ptr,
    sd_ptr,
    scale_ptr,
    sig_ptr,
    qn2_ptr,
    B,
    M,
    N,
    q0,
    p0,
    c0,
    q1,
    p1,
    c1,
    q2,
    p2,
    c2,
    q3,
    p3,
    c3,
    q4,
    p4,
    c4,
    q5,
    p5,
    c5,
    NB0: tl.constexpr,
    KB0: tl.constexpr,
    KP0: tl.constexpr,
    NB1: tl.constexpr,
    KB1: tl.constexpr,
    KP1: tl.constexpr,
    NB2: tl.constexpr,
    KB2: tl.constexpr,
    KP2: tl.constexpr,
    NB3: tl.constexpr,
    KB3: tl.constexpr,
    KP3: tl.constexpr,
    NB4: tl.constexpr,
    KB4: tl.constexpr,
    KP4: tl.constexpr,
    NB5: tl.constexpr,
    KB5: tl.constexpr,
    KP5: tl.constexpr,
    BM: tl.constexpr,
):
    q = tl.program_id(1)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    mmask = rm < M
    rows = tl.load(pos_ptr + q * M + rm, mask=mmask, other=0)
    acc = tl.zeros((BM,), dtype=tl.float32) + tl.load(qc_ptr + q)
    var = tl.zeros((BM,), dtype=tl.float32)
    if NB0 > 0:
        acc, var = _tier(acc, var, q, rows, mmask, B, N, 0, q0, p0, c0, scale_ptr, sig_ptr, qn2_ptr, NB0, KB0, KP0, BM)
    if NB1 > 0:
        acc, var = _tier(acc, var, q, rows, mmask, B, N, 1, q1, p1, c1, scale_ptr, sig_ptr, qn2_ptr, NB1, KB1, KP1, BM)
    if NB2 > 0:
        acc, var = _tier(acc, var, q, rows, mmask, B, N, 2, q2, p2, c2, scale_ptr, sig_ptr, qn2_ptr, NB2, KB2, KP2, BM)
    if NB3 > 0:
        acc, var = _tier(acc, var, q, rows, mmask, B, N, 3, q3, p3, c3, scale_ptr, sig_ptr, qn2_ptr, NB3, KB3, KP3, BM)
    if NB4 > 0:
        acc, var = _tier(acc, var, q, rows, mmask, B, N, 4, q4, p4, c4, scale_ptr, sig_ptr, qn2_ptr, NB4, KB4, KP4, BM)
    if NB5 > 0:
        acc, var = _tier(acc, var, q, rows, mmask, B, N, 5, q5, p5, c5, scale_ptr, sig_ptr, qn2_ptr, NB5, KB5, KP5, BM)
    tl.store(est_ptr + q * M + rm, acc, mask=mmask)
    tl.store(sd_ptr + q * M + rm, tl.sqrt(var), mask=mmask)


def score(code: TieredCode, Q: torch.Tensor, pos: torch.Tensor, BM: int = 32):
    """Estimates and standard deviations (B, M) of <q, x> for the candidates at sorted positions pos (B, M)."""
    B, M = pos.shape
    N = code.planes[0].shape[1]
    Zq = Q.float() @ code.VP
    qc = (Q.float() @ code.center).contiguous()
    qn2 = torch.stack([(Zq[:, s:e] ** 2).sum(1) for s, e, _ in code.tiers]).contiguous()
    args, cex = [], {}
    for t in range(MAX_TIERS):
        if t < len(code.tiers):
            s, e, b = code.tiers[t]
            kb = (e - s) // 8
            qt = Zq[:, s:e].reshape(B, kb, 8).permute(2, 0, 1).contiguous()
            args += [qt, code.planes[t], code.codebooks[t]]
            cex.update({f"NB{t}": b, f"KB{t}": kb, f"KP{t}": max(16, triton.next_power_of_2(kb))})
        else:
            args += [qn2, qn2, qn2]
            cex.update({f"NB{t}": 0, f"KB{t}": 1, f"KP{t}": 16})
    est = torch.empty(B, M, device=Q.device)
    sd = torch.empty(B, M, device=Q.device)
    _score_kernel[(triton.cdiv(M, BM), B)](pos.contiguous(), qc, est, sd, code.scale, code.sig, qn2, B, M, N, *args, BM=BM, **cex)
    return est, sd
