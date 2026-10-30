"""Bitshift trellis codes for the stage-1 filter (paper, Section 4).

A code with memory L stores a bit stream b_0 b_1 ...; coordinate j is decoded from the L-bit window
s_j = b_j ... b_{j+L-1} as alpha * T[s_j], where T is a fixed table of 2**L Gaussian quantiles in a
pseudo-random order. Encoding is a Viterbi search for the path of minimum squared error. Here k = 1
bit per coordinate: the state for coordinate j+1 shifts in one new bit.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


def make_table(L: int, seed: int = 0, device: str = "cuda") -> torch.Tensor:
    """2**L Gaussian quantiles (cell midpoints) in a fixed pseudo-random order."""
    S = 2**L
    perm = np.random.default_rng(seed).permutation(S)
    p = torch.from_numpy((perm + 0.5) / S)
    return torch.special.ndtri(p).float().to(device)


def _predecessors(L: int, device) -> torch.Tensor:
    s = torch.arange(2**L, device=device)
    return torch.stack([(s >> 1) | (u << (L - 1)) for u in range(2)], 1)


@torch.no_grad()
def encode_reference(X: torch.Tensor, T: torch.Tensor, L: int, alpha: float) -> torch.Tensor:
    """Viterbi path (n, r) of minimum squared error for rows X (n, r); pure PyTorch, used for tests and fitting alpha."""
    n, r = X.shape
    vals = alpha * T
    preds = _predecessors(L, X.device)
    cost = (X[:, :1] - vals[None]) ** 2
    back = torch.empty(r - 1, n, 2**L, dtype=torch.uint8, device=X.device)
    for j in range(1, r):
        best, arg = cost[:, preds].min(2)
        back[j - 1] = arg.to(torch.uint8)
        cost = best + (X[:, j : j + 1] - vals[None]) ** 2
    s = cost.argmin(1)
    path = torch.empty(n, r, dtype=torch.long, device=X.device)
    path[:, -1] = s
    for j in range(r - 1, 0, -1):
        s = preds[s, back[j - 1].gather(1, s[:, None])[:, 0].long()]
        path[:, j - 1] = s
    return path


@triton.jit
def _viterbi_kernel(x_ptr, vals_ptr, scr_ptr, back_ptr, path_ptr, n, L: tl.constexpr, S: tl.constexpr, SW: tl.constexpr):
    # one program per vector: path costs in a double-buffered scratch row, one backpointer bit per state
    v = tl.program_id(0)
    s = tl.arange(0, S)
    vals = tl.load(vals_ptr + s)
    sa = scr_ptr + v * 2 * S
    x0 = tl.load(x_ptr + v * n)
    tl.store(sa + s, (x0 - vals) * (x0 - vals))
    tl.debug_barrier()
    p0 = s >> 1
    p1 = p0 | (1 << (L - 1))
    bw = tl.arange(0, SW)
    for j in range(1, n):
        src = sa + ((j - 1) % 2) * S
        dst = sa + (j % 2) * S
        c0 = tl.load(src + p0)
        c1 = tl.load(src + p1)
        take1 = c1 < c0
        xj = tl.load(x_ptr + v * n + j)
        tl.store(dst + s, tl.where(take1, c1, c0) + (xj - vals) * (xj - vals))
        bits = tl.reshape(take1.to(tl.int32), (SW, 32))
        word = tl.sum(bits << tl.arange(0, 32)[None, :], axis=1)
        tl.store(back_ptr + (v * (n - 1) + (j - 1)) * SW + bw, word)
        tl.debug_barrier()
    last = sa + ((n - 1) % 2) * S
    c = tl.load(last + s)
    m = tl.min(c, axis=0)
    st = tl.min(tl.where(c == m, s, S), axis=0)
    tl.store(path_ptr + v * n + n - 1, st)
    for jj in range(1, n):
        j = n - jj
        w = tl.load(back_ptr + (v * (n - 1) + (j - 1)) * SW + st // 32)
        u = (w >> (st % 32)) & 1
        st = (st >> 1) | (u << (L - 1))
        tl.store(path_ptr + v * n + j - 1, st)


@torch.no_grad()
def encode(X: torch.Tensor, T: torch.Tensor, L: int, alpha: float, chunk: int = 8192) -> torch.Tensor:
    """Same result as encode_reference, computed by a fused Triton kernel (about 50x faster)."""
    n, r = X.shape
    S = 2**L
    vals = (alpha * T).float().contiguous()
    path = torch.empty(n, r, dtype=torch.int32, device=X.device)
    for i in range(0, n, chunk):
        x = X[i : i + chunk].float().contiguous()
        b = len(x)
        scr = torch.empty(b, 2, S, device=X.device)
        back = torch.empty(b, r - 1, S // 32, dtype=torch.int32, device=X.device)
        _viterbi_kernel[(b,)](x, vals, scr, back, path[i : i + b], r, L=L, S=S, SW=S // 32, num_warps=8)
    return path.long()


def paths_to_words(path: torch.Tensor, L: int) -> torch.Tensor:
    """Pack state paths (n, r) into the bit stream: L bits of initial state, then one bit per coordinate; int32 words (n, W)."""
    n, r = path.shape
    nbits = r + L - 1
    W = (nbits + 31) // 32
    bits = torch.zeros(n, W * 32, dtype=torch.int64, device=path.device)
    sh = torch.arange(L - 1, -1, -1, device=path.device)
    bits[:, :L] = (path[:, :1] >> sh) & 1
    bits[:, L:nbits] = path[:, 1:] & 1
    w = (bits.reshape(n, W, 32) << torch.arange(31, -1, -1, device=path.device)).sum(2)
    return (w & 0xFFFFFFFF).to(torch.int32)


def decode_reference(words: torch.Tensor, tab: torch.Tensor, r: int, L: int) -> torch.Tensor:
    """Decoded values (n, r) of packed codes; pure PyTorch, used by tests."""
    n, W = words.shape
    w = words.long() & 0xFFFFFFFF
    bits = ((w[:, :, None] >> torch.arange(31, -1, -1, device=words.device)) & 1).reshape(n, W * 32)
    weights = 2 ** torch.arange(L - 1, -1, -1, device=words.device)
    idx = torch.stack([(bits[:, j : j + L] * weights).sum(1) for j in range(r)], 1)
    return tab[idx]


@triton.jit
def decode_tile(words_ptr, rn, nmask, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, tab_ptr):
    """Decode rows rn (BN,) of packed codes into a (BN, RP) register tile; each row's words are loaded once."""
    j = tl.arange(0, RP)
    jm = j < R
    wi = j // 32
    off = j % 32
    w0 = tl.zeros((rn.shape[0], RP), dtype=tl.int64)
    w1 = tl.zeros((rn.shape[0], RP), dtype=tl.int64)
    for w in tl.static_range(W):
        col = tl.load(words_ptr + rn * W + w, mask=nmask, other=0).to(tl.int64) & 0xFFFFFFFF
        w0 = tl.where(wi[None, :] == w, col[:, None], w0)
        w1 = tl.where(wi[None, :] + 1 == w, col[:, None], w1)
    comb = (w0 << 32) | w1
    win = (comb >> (64 - L - off[None, :])) & ((1 << L) - 1)
    m2 = nmask[:, None] & jm[None, :]
    return tl.where(m2, tl.load(tab_ptr + win, mask=m2, other=0.0), 0.0)


def fit_alpha(T: torch.Tensor, L: int, r: int, seed: int = 0) -> float:
    """Table scale alpha of minimum distortion on N(0, 1) rows of length r."""
    g = torch.Generator(device=T.device).manual_seed(seed)
    G = torch.randn(512, r, device=T.device, generator=g)
    best = None
    for a in np.linspace(0.6, 1.3, 15):
        path = encode(G, T, L, float(a))
        m = float(((G - a * T[path]) ** 2).mean())
        if best is None or m < best[0]:
            best = (m, float(a))
    return best[1]
