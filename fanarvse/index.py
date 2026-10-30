"""The FanarVSE index: build, search, save and load."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from . import coarse, refine, rerank, stage1, trellis
from .config import IndexConfig, SearchParams

FORMAT_VERSION = 1


def _as_host_fp16(X, dpad: int) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        t = X
    else:
        with warnings.catch_warnings():  # read-only arrays (e.g. np.load with mmap_mode) are fine: the index never writes to them
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            t = torch.as_tensor(X)
    if t.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {tuple(t.shape)}")
    t = t.detach().to("cpu", torch.float16)
    if t.shape[1] < dpad:
        t = torch.nn.functional.pad(t, (0, dpad - t.shape[1]))
    return t.contiguous()


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    q: torch.Tensor
    outputs: tuple


@dataclass
class Index:
    """GPU vector index for maximum inner-product search (use normalised vectors for cosine similarity).

    Build with `Index.build(X)`; search with `index.search(Q, k)`. The compressed codes live on the GPU; the
    full vectors stay in pinned host RAM and are read only for the few candidates whose error bars overlap the
    top-k boundary.
    """

    config: IndexConfig
    dim: int
    s1: stage1.Stage1
    code: refine.TieredCode
    order: torch.Tensor  # (N,) int32: original id of each cell-sorted position
    host: torch.Tensor  # (N, dpad) fp16 full vectors in pinned host memory, original order
    normalize: bool = False
    build_seconds: float = 0.0
    _graphs: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------------------------------- build
    @classmethod
    @torch.no_grad()
    def build(cls, X, config: IndexConfig | None = None, normalize: bool = False, device: str = "cuda") -> Index:
        """Build an index over the rows of X (numpy array or tensor, float16 or float32, shape (N, d))."""
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            raise RuntimeError("FanarVSE needs a CUDA GPU")
        cfg = config or IndexConfig()
        t0 = time.perf_counter()
        n, d = X.shape
        dpad = -(-d // refine.G) * refine.G
        host = _as_host_fp16(X, dpad)
        if normalize:
            host = torch.nn.functional.normalize(host.float(), dim=1).half()
        R = min(cfg.stage1_dims, dpad)
        if R > 512:
            raise ValueError("stage1_dims above 512 exceeds the shared memory of the stage-1 kernels")
        n_cells = cfg.cells_for(n)
        g = torch.Generator().manual_seed(cfg.seed)
        sample = torch.randperm(n, generator=g)[: min(cfg.pca_sample, n)]

        # global centre and stage-1 projection: leading principal directions times a Haar rotation
        S = host[sample].to(device).float()
        center = S.mean(0)
        ev, V = torch.linalg.eigh(torch.cov((S - center).T))
        del S
        gd = torch.Generator(device=device).manual_seed(cfg.seed)
        W1 = (V.flip(1)[:, :R] @ torch.linalg.qr(torch.randn(R, R, device=device, generator=gd))[0]).contiguous()
        L = cfg.trellis_memory
        T = trellis.make_table(L, device=device)
        alpha = trellis.fit_alpha(T, L, R, seed=cfg.seed)

        # cells, cell-sorted order, cell means
        cen = coarse.spherical_kmeans(host, n_cells, cfg.kmeans_sample, cfg.kmeans_iters, cfg.seed, device)
        cell = coarse.assign(host, cen)
        order = torch.argsort(cell, stable=True)
        off = torch.searchsorted(cell[order], torch.arange(n_cells + 1, device=device))
        cell_sorted = cell[order]
        oc = order.cpu()
        means = torch.zeros(n_cells, dpad, device=device)
        for j in range(0, n, 65536):
            means.index_add_(0, cell_sorted[j : j + 65536], host[oc[j : j + 65536]].to(device).float())
        means /= (off[1:] - off[:-1]).clamp(min=1).float()[:, None]

        # stage 1: trellis codes of W1^T (x - m_c), scaled so that the estimate of <q, x - m_c> is unbiased in expectation
        tab = (alpha * T).contiguous()
        words = None
        scale = torch.empty(n, dtype=torch.float16, device=device)
        for j in range(0, n, 16384):
            Rr = host[oc[j : j + 16384]].to(device).float() - means[cell_sorted[j : j + 16384]]
            Z = Rr @ W1
            nz = Z.norm(dim=1).clamp(min=1e-6)
            Zs = Z / nz[:, None] * R**0.5
            path = trellis.encode(Zs, T, L, alpha)
            w = trellis.paths_to_words(path, L)
            if words is None:
                words = torch.empty(n, w.shape[1], dtype=torch.int32, device=device)
            words[j : j + len(w)] = w
            Y = tab[path]
            yn = Y.norm(dim=1)
            fip = (Zs * Y).sum(1) / (yn * R**0.5)
            scale[j : j + len(w)] = (nz / (fip * yn)).half()
        means16 = means.half().contiguous()
        s1 = stage1.Stage1(
            words=words,
            scale=scale,
            tab=tab,
            W1=W1,
            means16=means16,
            inv_mnorm=(1.0 / means.norm(dim=1).clamp(min=1e-12)).contiguous(),
            off=off.contiguous(),
            L=L,
        )
        del means

        # stage 2: tiered refinement code of x - c
        code = refine.build(host, oc, center, sample, int(round(cfg.refine_bits_per_dim * dpad)), cfg.seed)
        idx = cls(config=cfg, dim=d, s1=s1, code=code, order=order.to(torch.int32).contiguous(), host=rerank.pin(host), normalize=normalize)
        torch.cuda.synchronize()
        idx.build_seconds = time.perf_counter() - t0
        return idx

    # ------------------------------------------------------------------------------------------- properties
    @property
    def n(self) -> int:
        return len(self.order)

    @property
    def n_cells(self) -> int:
        return self.s1.n_cells

    def memory_report(self) -> dict:
        """Bytes held on the GPU (per-vector arrays and shared tables) and in pinned host RAM."""
        size = lambda t: t.numel() * t.element_size()  # noqa: E731
        s1, c = self.s1, self.code
        per_s1 = size(s1.words) + size(s1.scale) + size(self.order)
        per_ref = sum(size(p) for p in c.planes) + size(c.scale) + size(c.sig)
        tables = size(s1.tab) + size(s1.W1) + size(s1.means16) + size(s1.inv_mnorm) + size(s1.off) + size(c.VP) + size(c.center)
        tables += sum(size(b) for b in c.codebooks)
        gpu = per_s1 + per_ref + tables
        return {
            "gpu_bytes": gpu,
            "gpu_bytes_per_vector": gpu / self.n,
            "stage1_bytes_per_vector": per_s1 / self.n,
            "refine_bytes_per_vector": per_ref / self.n,
            "shared_tables_bytes": tables,
            "host_bytes": size(self.host),
        }

    # ------------------------------------------------------------------------------------------- search
    def _rank(self, Q, pos, z: float, k: int):
        """Stages 2 and 3 for candidates pos: error bars, exact reads of the ambiguous ones, top-k ids."""
        est, sd = refine.score(self.code, Q, pos)
        lo, hi = est - z * sd, est + z * sd
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]
        amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = self.order[pos]
        # surely-in candidates rank first (at most k of them, Lemma 4), then the best ambiguous ones by exact score
        big = 2 * (est.abs() + z * sd).amax(1, keepdim=True) + 1
        key = torch.where(sure, est + big, torch.full_like(est, -3e38))
        rerank.exact_scores(self.host, ids, amb, Q, key)
        top = key.topk(k, 1).indices
        return ids.gather(1, top).long(), amb.sum(1)

    def _gpu(self, Q, p: SearchParams, k: int):
        Qr = Q @ self.s1.W1
        QM = Q.half() @ self.s1.means16.T
        probe = stage1.route(QM, self.s1, p.nprobe)
        pos, _, ok = stage1.select(Qr, QM, self.s1, p.candidates, probe)
        I, reads = self._rank(Q, pos, p.z, k)
        return ok, I, reads

    def _fallback(self, Q, p: SearchParams, k: int):
        Qr = Q @ self.s1.W1
        QM = Q.half() @ self.s1.means16.T
        probe = stage1.route(QM, self.s1, p.nprobe)
        pos, _ = stage1.exact(Qr, QM, self.s1, p.candidates, probe)
        return self._rank(Q, pos, p.z, k)

    def _run(self, Q, p: SearchParams, k: int):
        key = (len(Q), p, k)
        g = self._graphs.get(key)
        if g is None:
            q = Q.clone()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):  # compile kernels and warm the caches outside the graph
                    self._gpu(q, p, k)
            torch.cuda.current_stream().wait_stream(st)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = self._gpu(q, p, k)
            g = self._graphs[key] = _Graph(graph, q, outs)
        g.q.copy_(Q)
        g.graph.replay()
        ok, I, reads = g.outputs
        I, reads = I.clone(), reads.clone()
        if not bool(ok.all()):  # the only host synchronisation: rare queries whose guessed threshold failed
            bad = torch.nonzero(~ok).flatten()
            I[bad], reads[bad] = self._fallback(Q[bad], p, k)
        return I, reads

    @torch.no_grad()
    def search(
        self,
        Q,
        k: int = 10,
        preset: str = "balanced",
        params: SearchParams | None = None,
        batch_size: int = 1024,
        return_scores: bool = False,
    ):
        """Top-k ids (numpy int64, shape (B, k)) of each query row of Q by inner product.

        preset: "fast", "balanced", "high" or "max" (recall@10 0.982 / 0.992 / 0.996 / 0.999 on OpenAI-1M);
        params overrides it. With return_scores, also returns the exact inner products (B, k), sorted.
        """
        p = params or SearchParams.preset(preset, self.n_cells)
        if not k <= p.candidates <= self.n:
            raise ValueError(f"need k <= candidates <= number of vectors (k={k}, candidates={p.candidates}, n={self.n})")
        Qt = torch.as_tensor(Q).float()
        if Qt.ndim != 2 or Qt.shape[1] != self.dim:
            raise ValueError(f"expected queries of shape (B, {self.dim}), got {tuple(Qt.shape)}")
        Qt = torch.nn.functional.pad(Qt, (0, self.host.shape[1] - self.dim)).cuda().contiguous()
        if self.normalize:
            Qt = torch.nn.functional.normalize(Qt, dim=1)
        out = []
        for s in range(0, len(Qt), batch_size):
            q = Qt[s : s + batch_size]
            b = len(q)
            size = b if b <= stage1.SMALL else min(batch_size, 1 << (b - 1).bit_length())
            if size > b:  # pad with copies of the first row so that a few CUDA graphs serve every batch
                q = torch.cat([q, q[:1].expand(size - b, -1)])
            out.append(self._run(q, p, k)[0][:b])
        I = torch.cat(out) if out else torch.empty(0, k, dtype=torch.long, device="cuda")
        if not return_scores:
            return I.cpu().numpy()
        S = torch.empty(I.shape, device="cuda")
        rerank.exact_scores(self.host, I, torch.ones_like(I, dtype=torch.bool), Qt, S)
        S, o = S.sort(1, descending=True)
        return I.gather(1, o).cpu().numpy(), S.cpu().numpy()

    def clear_cache(self):
        """Release the CUDA graphs captured for each batch shape and parameter set."""
        self._graphs.clear()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------------------------- persistence
    def save(self, path: str):
        s1, c = self.s1, self.code
        cpu = lambda t: t.detach().cpu()  # noqa: E731
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "config": self.config.to_dict(),
                "dim": self.dim,
                "normalize": self.normalize,
                "stage1": {
                    "words": cpu(s1.words),
                    "scale": cpu(s1.scale),
                    "tab": cpu(s1.tab),
                    "W1": cpu(s1.W1),
                    "means16": cpu(s1.means16),
                    "inv_mnorm": cpu(s1.inv_mnorm),
                    "off": cpu(s1.off),
                    "L": s1.L,
                },
                "refine": {
                    "tiers": c.tiers,
                    "planes": [cpu(p) for p in c.planes],
                    "scale": cpu(c.scale),
                    "sig": cpu(c.sig),
                    "VP": cpu(c.VP),
                    "codebooks": [cpu(b) for b in c.codebooks],
                    "center": cpu(c.center),
                },
                "order": cpu(self.order),
                "host": self.host,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> Index:
        d = torch.load(path, map_location="cpu", weights_only=False)
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported index format {d.get('format_version')!r} (this version reads {FORMAT_VERSION})")
        a, r = d["stage1"], d["refine"]
        g = lambda t: t.to(device)  # noqa: E731
        s1 = stage1.Stage1(
            words=g(a["words"]),
            scale=g(a["scale"]),
            tab=g(a["tab"]),
            W1=g(a["W1"]),
            means16=g(a["means16"]),
            inv_mnorm=g(a["inv_mnorm"]),
            off=g(a["off"]),
            L=a["L"],
        )
        code = refine.TieredCode(
            tiers=[tuple(t) for t in r["tiers"]],
            planes=[g(p) for p in r["planes"]],
            scale=g(r["scale"]),
            sig=g(r["sig"]),
            VP=g(r["VP"]),
            codebooks=[g(b) for b in r["codebooks"]],
            center=g(r["center"]),
        )
        return cls(
            config=IndexConfig(**d["config"]),
            dim=d["dim"],
            s1=s1,
            code=code,
            order=g(d["order"]),
            host=rerank.pin(d["host"]),
            normalize=d["normalize"],
        )


def recall(I: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """Mean fraction of the true top-k found among the returned top-k ids."""
    I, truth = np.asarray(I)[:, :k], np.asarray(truth)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(I, truth, strict=True)]))


@torch.no_grad()
def ground_truth(X, Q, k: int = 10, chunk: int = 200_000) -> np.ndarray:
    """Exact top-k ids by inner product, computed on the GPU in fp32 (for measuring recall)."""
    Qt = torch.as_tensor(Q).float().cuda()
    best_v = torch.full((len(Qt), k), -float("inf"), device="cuda")
    best_i = torch.zeros((len(Qt), k), dtype=torch.long, device="cuda")
    for j in range(0, len(X), chunk):
        S = Qt @ torch.as_tensor(X[j : j + chunk]).float().cuda().T
        v, i = torch.cat([best_v, S], 1), torch.cat([best_i, torch.arange(j, j + S.shape[1], device="cuda").expand(len(Qt), -1)], 1)
        t = v.topk(k, 1)
        best_v, best_i = t.values, i.gather(1, t.indices)
    return best_i.cpu().numpy()
