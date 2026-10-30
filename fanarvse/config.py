"""Build and search parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Search presets measured on 998k OpenAI-ada-002 vectors with 4,096 cells (paper, Table 3).
# nprobe is stored for 4,096 cells and scaled with the index's number of cells.
_PRESET_CELLS = 4096
PRESETS = {
    "fast": dict(nprobe=256, candidates=200, z=2.0),  # recall@10 0.982
    "balanced": dict(nprobe=384, candidates=400, z=2.0),  # recall@10 0.992
    "high": dict(nprobe=768, candidates=400, z=2.0),  # recall@10 0.996
    "max": dict(nprobe=1536, candidates=800, z=2.0),  # recall@10 0.999
}


@dataclass
class IndexConfig:
    """Parameters fixed when the index is built.

    n_cells: number of coarse cells; None picks a power of two near N / 244 (4,096 for 1M vectors).
    stage1_dims: principal directions kept by the stage-1 code (R in the paper).
    trellis_memory: state bits of the bitshift trellis (L in the paper).
    refine_bits_per_dim: average bits per dimension of the tiered refinement code.
    """

    n_cells: int | None = None
    stage1_dims: int = 256
    trellis_memory: int = 10
    refine_bits_per_dim: float = 2.5
    kmeans_iters: int = 15
    kmeans_sample: int = 200_000
    pca_sample: int = 50_000
    seed: int = 0

    def cells_for(self, n: int) -> int:
        if self.n_cells is not None:
            return int(self.n_cells)
        target = max(1, n / 244)
        p = 1
        while p * 2 <= target * 1.4142:
            p *= 2
        return max(16, p)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchParams:
    """Parameters of one search.

    nprobe: cells scanned per query.
    candidates: stage-1 candidates per query (K in the paper).
    z: half-width of the error bars in standard deviations; candidates whose interval
       straddles the top-k boundary are scored exactly.
    """

    nprobe: int
    candidates: int
    z: float = 2.0

    @staticmethod
    def preset(name: str, n_cells: int) -> SearchParams:
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        p = PRESETS[name]
        nprobe = max(1, min(n_cells, round(p["nprobe"] * n_cells / _PRESET_CELLS)))
        return SearchParams(nprobe=nprobe, candidates=p["candidates"], z=p["z"])
