"""Stage 1: exact top-K selection over the probed cells by threshold-and-emit (paper, Section 5).

Every vector x in cell c is coded as the trellis code of W1^T (x - m_c) with a per-vector scale, so its
estimate is <q, m_c> + scale * <W1^T q, decoded code>. A first pass computes block maxima over the
nearest cells only and sets a threshold t0 (the r-th largest maximum); the emit pass appends every
item with estimate >= t0 to a per-query buffer and counts them. If a query emitted at least K and at
most the buffer size, its top K of the buffer is the exact top K of its probed cells (Lemma 3); the
others are flagged and recomputed by `exact`.

Two kernel families: a static work list over all (cell, block) pairs of the index for batches, and a
dense (query x cell) probe mask for up to 32 queries. Neither synchronises with the host, so a whole
search can be captured as one CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

from .trellis import decode_tile

BN = 32  # vectors per code block
BQ = 32  # queries per tile in the batch kernels
SMALL = 32  # batches up to this size use the probe-mask kernels


@dataclass
class Stage1:
    """Stage-1 data of an index (all on the GPU); positions refer to the cell-sorted order."""

    words: torch.Tensor  # (N, W) int32 packed trellis codes of the cell residuals
    scale: torch.Tensor  # (N,) fp16 per-vector scale
    tab: torch.Tensor  # (2**L,) fp32 decoding table (alpha * T)
    W1: torch.Tensor  # (d, R) fp32 projection: principal directions times a random rotation
    means16: torch.Tensor  # (n_cells, d) fp16 cell means
    inv_mnorm: torch.Tensor  # (n_cells,) fp32 1 / |m_c|
    off: torch.Tensor  # (n_cells + 1,) int64 cell boundaries in sorted order
    L: int

    def __post_init__(self):
        self._blocks = None

    @property
    def n_cells(self) -> int:
        return len(self.off) - 1

    def blocks(self):
        """All (cell, block) programs of the index, with cell bounds and the largest number of blocks per cell."""
        if self._blocks is None:
            s, e = self.off[:-1], self.off[1:]
            nb = torch.where(e > s, (e - 1) // BN - s // BN + 1, 0)
            cell = torch.repeat_interleave(torch.arange(len(s), device=s.device), nb)
            first = torch.cumsum(nb, 0) - nb
            blk = s[cell] // BN + torch.arange(len(cell), device=s.device) - first[cell]
            mb = int((int((e - s).max()) + BN - 1) // BN + 1)
            i32 = lambda x: x.to(torch.int32).contiguous()  # noqa: E731
            self._blocks = (i32(cell), i32(blk), i32(s[cell]), i32(e[cell]), mb)
        return self._blocks


def route(QM: torch.Tensor, s1: Stage1, nprobe: int) -> torch.Tensor:
    """Cells ranked by <q, m_c> / |m_c| (the spherical k-means centroid direction), nearest first."""
    return (QM.float() * s1.inv_mnorm).topk(nprobe, 1).indices


def first_cells(s1: Stage1, K: int, beta: float, nprobe: int) -> int:
    n = s1.words.shape[0]
    return min(max(int(np.ceil(beta * K / (n / s1.n_cells / BN + 1))), 1), nprobe)


# ---------------------------------------------------------------------------------------------- batch kernels
@triton.jit
def _first_pass(
    q_ptr,
    qrows_ptr,
    c_qoff_ptr,
    c_qn_ptr,
    p_cell_ptr,
    p_blk_ptr,
    p_s_ptr,
    p_e_ptr,
    words_ptr,
    tab_ptr,
    scale_ptr,
    qc_ptr,
    bmc_ptr,
    MB,
    W: tl.constexpr,
    R: tl.constexpr,
    RP: tl.constexpr,
    L: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
):
    # block maxima of the estimates for every (query, cell) pair of the first pass
    p = tl.program_id(0)
    c = tl.load(p_cell_ptr + p)
    qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c)
        blk = tl.load(p_blk_ptr + p)
        cs = tl.load(p_s_ptr + p)
        rn = blk * BN + tl.arange(0, BN)
        nmask = (rn >= cs) & (rn < tl.load(p_e_ptr + p))
        j = tl.arange(0, RP)
        v = decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
        s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ)
            qmask = rqi < qn
            rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            qc = tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)
            est = tl.where(nmask[None, :], s[None, :] * tl.dot(q, tl.trans(v)) + qc[:, None], -1e30)
            tl.store(bmc_ptr + (qoff + rqi) * MB + (blk - cs // BN), tl.max(est, axis=1), mask=qmask)


@triton.jit
def _emit(
    q_ptr,
    qrows_ptr,
    c_qoff_ptr,
    c_qn_ptr,
    p_cell_ptr,
    p_blk_ptr,
    p_s_ptr,
    p_e_ptr,
    words_ptr,
    tab_ptr,
    scale_ptr,
    qc_ptr,
    t0_ptr,
    cnt_ptr,
    bval_ptr,
    bpos_ptr,
    CAP,
    W: tl.constexpr,
    R: tl.constexpr,
    RP: tl.constexpr,
    L: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
):
    # append every item with estimate >= t0[q] to query q's buffer; cnt[q] counts all of them (also beyond CAP)
    p = tl.program_id(0)
    c = tl.load(p_cell_ptr + p)
    qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c)
        blk = tl.load(p_blk_ptr + p)
        rn = blk * BN + tl.arange(0, BN)
        nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p))
        j = tl.arange(0, RP)
        v = decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
        s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ)
            qmask = rqi < qn
            rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            est = s[None, :] * tl.dot(q, tl.trans(v)) + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
            ok = keep & (idx < CAP)
            tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
            tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)


def _pairs(probe: torch.Tensor, n_cells: int, QM: torch.Tensor):
    """Queries grouped by cell (CSR over cells, fixed-size ops) and the constant <q, m_c> of every pair in that order."""
    B, npr = probe.shape
    flat = probe.flatten().to(torch.int32)
    _, o = torch.sort(flat, stable=True)
    qid = (o // npr).to(torch.int32).contiguous()
    cnt = torch.zeros(n_cells, dtype=torch.int32, device=probe.device).scatter_add_(0, flat.long(), torch.ones_like(flat))
    off = torch.cumsum(cnt, 0) - cnt
    qc = QM[qid.long(), flat[o].long()].float().contiguous()
    return qid, off.to(torch.int32).contiguous(), cnt.contiguous(), qc, o


def select_batch(Qr, QM, s1: Stage1, K: int, probe, beta=1.1, rho=0.75, M=None, t0=None):
    """Guess-and-verify top-K for a batch. Returns (pos, est, ok, cnt); ok[q] is False when query q emitted fewer
    than K or more than M items. With t0 given (one threshold per query) the first pass is skipped."""
    B, R = Qr.shape
    N, W = s1.words.shape
    RP = triton.next_power_of_2(R)
    NC = s1.n_cells
    M = M or max(4096, 16 * K)
    pcell, pblk, ps, pe, MB = s1.blocks()
    P = len(pcell)
    Qh = Qr.half().contiguous()
    meta = dict(W=W, R=R, RP=RP, L=s1.L, BQ=BQ, BN=BN, num_stages=1)
    if t0 is None:
        n1 = first_cells(s1, K, beta, probe.shape[1])
        qid, coff, ccnt, qc, o = _pairs(probe[:, :n1].contiguous(), NC, QM)
        bmc = torch.full((B * n1, MB), -1e30, device=Qr.device)
        _first_pass[(P,)](Qh, qid, coff, ccnt, pcell, pblk, ps, pe, s1.words, s1.tab, s1.scale, qc, bmc, MB, **meta)
        inv = torch.empty_like(o)
        inv[o] = torch.arange(len(o), device=Qr.device, dtype=o.dtype)
        v = bmc[inv].reshape(B, n1 * MB)
        top = v.topk(min(int(np.ceil(rho * K)), v.shape[1]), 1).values
        t0 = torch.where(top > -1e29, top, torch.full_like(top, 1e30)).min(1).values.contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=Qr.device)
    bval = torch.full((B, M), -1e30, device=Qr.device)
    bpos = torch.zeros((B, M), dtype=torch.int32, device=Qr.device)
    qid, coff, ccnt, qc, _ = _pairs(probe, NC, QM)
    _emit[(P,)](Qh, qid, coff, ccnt, pcell, pblk, ps, pe, s1.words, s1.tab, s1.scale, qc, t0, cnt, bval, bpos, M, **meta)
    t = bval.topk(K, 1)
    return bpos.gather(1, t.indices).long(), t.values, (cnt >= K) & (cnt <= M), cnt


# ---------------------------------------------------------------------------------------------- small-batch kernel
@triton.jit
def _mask_scan(
    q_ptr,
    pm_ptr,
    off_ptr,
    words_ptr,
    tab_ptr,
    scale_ptr,
    qc_ptr,
    t0_ptr,
    cnt_ptr,
    bval_ptr,
    bpos_ptr,
    bm_ptr,
    B,
    NC,
    CAP,
    NB,
    LEVEL,
    MAXMODE: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    RP: tl.constexpr,
    L: tl.constexpr,
    BN: tl.constexpr,
    BQ: tl.constexpr,
):
    # one program per (cell, block); pm[q, c] = 0 not probed, 1 probed, 2 probed and in the first pass
    c = tl.program_id(0)
    jb = tl.program_id(1)
    s = tl.load(off_ptr + c).to(tl.int32)
    e = tl.load(off_ptr + c + 1).to(tl.int32)
    b = s // BN + jb
    if b * BN < e:
        rq = tl.arange(0, BQ)
        qmask = rq < B
        lvl = tl.load(pm_ptr + rq * NC + c, mask=qmask, other=0).to(tl.int32)
        use = qmask & (lvl >= LEVEL)
        if tl.max(use.to(tl.int32), axis=0) > 0:
            rn = b * BN + tl.arange(0, BN)
            nmask = (rn >= s) & (rn < e)
            j = tl.arange(0, RP)
            v = decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=use[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            sc = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
            qc = tl.load(qc_ptr + rq * NC + c, mask=qmask, other=0.0)
            est = sc[None, :] * tl.dot(q, tl.trans(v)) + qc[:, None]
            m2 = use[:, None] & nmask[None, :]
            if MAXMODE:
                tl.atomic_max(bm_ptr + rq * NB + b, tl.max(tl.where(m2, est, -1e30), axis=1), mask=use)
            else:
                t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
                keep = m2 & (est >= t0[:, None])
                idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
                ok = keep & (idx < CAP)
                tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
                tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :] + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)


def select_small(Qr, QM, s1: Stage1, K: int, probe, beta=1.1, rho=0.75, CAP=8192):
    """Guess-and-verify top-K for at most SMALL queries, without a work list. Returns (pos, est, ok, cnt)."""
    B, R = Qr.shape
    assert B <= SMALL
    N, W = s1.words.shape
    NB = triton.cdiv(N, BN)
    RP = triton.next_power_of_2(R)
    NC = s1.n_cells
    MB = s1.blocks()[4]
    bq = max(16, triton.next_power_of_2(B))
    Qh = Qr.half().contiguous()
    qc = QM.float().contiguous()
    nprobe = probe.shape[1]
    n1 = first_cells(s1, K, beta, nprobe)
    lvl = torch.ones(B, nprobe, dtype=torch.int8, device=Qr.device)
    lvl[:, :n1] = 2
    pm = torch.zeros(B, NC, dtype=torch.int8, device=Qr.device).scatter_(1, probe, lvl)
    bm = torch.full((B, NB), -1e30, device=Qr.device)
    d = torch.zeros(1, device=Qr.device)
    di = torch.zeros(1, dtype=torch.int32, device=Qr.device)
    meta = dict(W=W, R=R, RP=RP, L=s1.L, BN=BN, BQ=bq)
    _mask_scan[(NC, MB)](Qh, pm, s1.off, s1.words, s1.tab, s1.scale, qc, d, di, d, di, bm, B, NC, CAP, NB, 2, MAXMODE=True, **meta)
    t0 = bm.topk(min(int(np.ceil(rho * K)), NB), 1).values[:, -1].contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=Qr.device)
    bval = torch.full((B, CAP), -1e30, device=Qr.device)
    bpos = torch.zeros((B, CAP), dtype=torch.int32, device=Qr.device)
    _mask_scan[(NC, MB)](Qh, pm, s1.off, s1.words, s1.tab, s1.scale, qc, t0, cnt, bval, bpos, d, B, NC, CAP, NB, 1, MAXMODE=False, **meta)
    t = bval.topk(K, 1)
    return bpos.gather(1, t.indices).long(), t.values, (cnt >= K) & (cnt <= CAP), cnt


def select(Qr, QM, s1: Stage1, K: int, probe):
    """Graph-safe selection for any batch size; returns (pos, est, ok)."""
    if len(Qr) <= SMALL:
        return select_small(Qr, QM, s1, K, probe)[:3]
    return select_batch(Qr, QM, s1, K, probe)[:3]


@torch.no_grad()
def exact(Qr, QM, s1: Stage1, K: int, probe):
    """Exact top-K of the probed cells for queries flagged by `select` (synchronises with the host).

    With r = K the threshold is met by at least K items whenever the first pass sees K valid blocks (Lemma 3);
    the buffer then grows until nothing is dropped. A query whose first pass sees fewer than K blocks emits
    every probed item instead.
    """
    M = max(65536, 16 * K)
    pos, est, ok, cnt = select_batch(Qr, QM, s1, K, probe, beta=1.5, rho=1.0, M=M)
    over = torch.nonzero(cnt > M).flatten()
    if len(over):  # the threshold is deterministic, so a buffer as large as the largest count drops nothing
        p2, e2, ok2, _ = select_batch(Qr[over], QM[over], s1, K, probe[over], beta=1.5, rho=1.0, M=int(cnt[over].max()))
        pos[over], est[over], ok[over] = p2, e2, ok2
    few = torch.nonzero(~ok).flatten()
    if len(few):
        sizes = (s1.off[probe[few] + 1] - s1.off[probe[few]]).sum(1)
        m = max(int(sizes.max()), K)
        t0 = torch.full((len(few),), -1e30, device=Qr.device)
        p2, e2, _, c2 = select_batch(Qr[few], QM[few], s1, min(K, m), probe[few], M=m, t0=t0)
        if p2.shape[1] < K:  # fewer than K items in the probed cells: pad with the best one (results stay valid ids)
            p2 = torch.cat([p2, p2[:, :1].expand(-1, K - p2.shape[1])], 1)
            e2 = torch.cat([e2, torch.full((len(few), K - e2.shape[1]), -1e30, device=Qr.device)], 1)
        pos[few], est[few] = p2, e2
    return pos, est
