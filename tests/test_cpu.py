"""Tests that need no GPU."""

import numpy as np
import pytest
import torch

from fanarvse import IndexConfig, SearchParams, _tables, refine, trellis


@pytest.mark.parametrize("bits,mse", [(1, 0.3634), (2, 0.1175), (3, 0.03454), (4, 0.009497)])
def test_lloyd_max_matches_known_distortion(bits, mse):
    levels, th, m = _tables.lloyd_max(bits)
    assert abs(m - mse) < 2e-4
    assert torch.all(levels[1:] > levels[:-1])
    assert torch.allclose(levels, -levels.flip(0), atol=1e-5)  # symmetric


def test_allocation_spends_the_budget_on_high_variance_groups():
    lam = torch.tensor(np.geomspace(10, 0.01, 256), dtype=torch.float32)
    tiers = refine.allocate(lam, 2 * 256)
    assert tiers[0][0] == 0 and tiers[-1][1] == 256
    bits = [b for s, e, b in tiers for _ in range(s, e)]
    assert sum(bits) <= 2 * 256 and sum(bits) > 2 * 256 - refine.G
    assert bits == sorted(bits, reverse=True)  # more variance never gets fewer bits


def test_packing_round_trip():
    L, n, r = 10, 50, 64
    g = torch.Generator().manual_seed(0)
    start = torch.randint(0, 2**L, (n, 1), generator=g)
    new_bits = torch.randint(0, 2, (n, r - 1), generator=g)
    path = [start[:, 0]]
    for j in range(r - 1):  # each state shifts in one new bit
        path.append(((path[-1] << 1) | new_bits[:, j]) & (2**L - 1))
    path = torch.stack(path, 1)
    T = trellis.make_table(L, device="cpu")
    words = trellis.paths_to_words(path, L)
    assert torch.equal(trellis.decode_reference(words, T, r, L), T[path])


def test_reference_encoder_beats_independent_rounding():
    L, r = 8, 64
    T = trellis.make_table(L, device="cpu")
    X = torch.randn(64, r, generator=torch.Generator().manual_seed(1))
    path = trellis.encode_reference(X, T, L, 1.0)
    mse = float(((X - T[path]) ** 2).mean())
    assert mse < 0.33  # a 1-bit scalar quantizer gets 0.363


def test_cells_and_presets():
    cfg = IndexConfig()
    assert cfg.cells_for(998_000) == 4096
    assert IndexConfig(n_cells=100).cells_for(10) == 100
    p = SearchParams.preset("balanced", 4096)
    assert (p.nprobe, p.candidates, p.z) == (384, 400, 2.0)
    assert SearchParams.preset("balanced", 1024).nprobe == 96
    with pytest.raises(ValueError):
        SearchParams.preset("nope", 4096)
