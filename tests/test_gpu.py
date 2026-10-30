"""Tests of the paper's guarantees on the GPU kernels, and of the Index API."""

import numpy as np
import pytest
import torch

import fanarvse
from fanarvse import refine as refine_mod
from fanarvse import stage1, trellis

from .conftest import clustered, gpu

pytestmark = gpu


def test_fused_encoder_equals_reference_viterbi():
    L, r = 10, 96
    T = trellis.make_table(L)
    X = torch.randn(256, r, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    a = trellis.encode(X, T, L, 0.9)
    b = trellis.encode_reference(X, T, L, 0.9)
    cost = lambda p: ((X - 0.9 * T[p]) ** 2).sum(1)  # noqa: E731
    assert torch.allclose(cost(a), cost(b), rtol=1e-5)  # same optimum (paths may differ only between exact ties)
    assert (a == b).float().mean() > 0.99


def _reference_estimates(index, Q, probe):
    """Stage-1 estimates of every vector of the probed cells, computed in PyTorch with the kernels' fp16 inputs."""
    s1 = index.s1
    Qr = (Q @ s1.W1).half().float()
    QM = (Q.half() @ s1.means16.T).float()
    dec = trellis.decode_reference(s1.words, s1.tab, s1.W1.shape[1], s1.L).half().float()
    est = (Qr @ dec.T) * s1.scale.float()
    cell = torch.searchsorted(s1.off, torch.arange(index.n, device="cuda"), right=True) - 1
    est = est + QM.gather(1, cell.expand(len(Q), -1))
    probed = torch.zeros(len(Q), s1.n_cells, dtype=torch.bool, device="cuda").scatter_(1, probe, True)
    return torch.where(probed.gather(1, cell.expand(len(Q), -1)), est, torch.full_like(est, -1e30))


@pytest.mark.parametrize("batch", [5, 64])
def test_stage1_selects_the_exact_top_K(index, data, batch):
    Q = torch.as_tensor(data[1][:batch]).cuda()
    Q = torch.nn.functional.pad(Q, (0, index.host.shape[1] - Q.shape[1]))
    K, nprobe = 100, 24
    QM = Q.half() @ index.s1.means16.T
    probe = stage1.route(QM, index.s1, nprobe)
    pos, est, ok = stage1.select(Q @ index.s1.W1, QM, index.s1, K, probe)
    pos2, _ = stage1.exact(Q @ index.s1.W1, QM, index.s1, K, probe)
    pos = torch.where(ok[:, None], pos, pos2)
    ref = _reference_estimates(index, Q, probe)
    kth = ref.topk(K, 1).values[:, -1:]
    chosen = ref.gather(1, pos)
    assert torch.all(chosen >= kth - 1e-3)  # every selected item is in the top K (up to float rounding at ties)
    assert len(set(pos[0].tolist())) == K


def test_refinement_estimate_is_unbiased_with_calibrated_error_bars(index, data):
    Q = torch.nn.functional.pad(torch.as_tensor(data[1][:50]).cuda(), (0, index.host.shape[1] - data[1].shape[1]))
    pos = torch.randint(0, index.n, (50, 400), device="cuda")
    est, sd = refine_mod.score(index.code, Q, pos)
    X = index.host.cuda().float()[index.order.long()[pos]]
    true = (X * Q[:, None, :]).sum(-1)
    z = ((est - true) / sd).flatten()
    assert abs(float(z.mean())) < 0.05
    assert float(z.std()) < 1.0  # the engine's sigma is conservative (it uses |q| for |q_perp|)
    assert float((z.abs() > 4).float().mean()) < 1e-3


def test_recall_rises_with_probes(index, data):
    X, Q = data
    gt = fanarvse.ground_truth(X, Q, 10)
    r_small = fanarvse.recall(index.search(Q, 10, params=fanarvse.SearchParams(8, 100)), gt)
    r_all = fanarvse.recall(index.search(Q, 10, params=fanarvse.SearchParams(index.n_cells, 400)), gt)
    assert r_small < r_all
    assert r_all > 0.99


def test_every_batch_size_gives_the_same_result(index, data):
    Q = data[1][:70]
    p = fanarvse.SearchParams(32, 200)
    ref = index.search(Q, 10, params=p, batch_size=70)
    for bs in (1, 3, 32, 33, 64):
        assert np.array_equal(index.search(Q, 10, params=p, batch_size=bs), ref)


def test_scores_are_exact_and_sorted(index, data):
    X, Q = data
    I, S = index.search(Q[:20], 10, return_scores=True)
    assert np.allclose(S, np.einsum("bkd,bd->bk", X[I], Q[:20]), atol=2e-3)  # fp16 storage of the vectors
    assert np.all(np.diff(S, axis=1) <= 0)


def test_save_and_load(index, data, tmp_path):
    p = tmp_path / "idx.pt"
    index.save(p)
    other = fanarvse.Index.load(p)
    Q = data[1][:50]
    assert np.array_equal(other.search(Q, 10), index.search(Q, 10))
    assert other.memory_report()["gpu_bytes"] == index.memory_report()["gpu_bytes"]


def test_input_checks(index, data):
    with pytest.raises(ValueError):
        index.search(data[1][:, :50], 10)
    with pytest.raises(ValueError):
        index.search(data[1], 10, params=fanarvse.SearchParams(8, 5))


def test_tiny_index_with_few_items_per_probe():
    X, Q = clustered(2_000, 64, 20, seed=3)
    idx = fanarvse.Index.build(X, fanarvse.IndexConfig(n_cells=64))
    gt = fanarvse.ground_truth(X, Q, 10)
    I = idx.search(Q, 10, params=fanarvse.SearchParams(nprobe=2, candidates=200))  # 2 cells hold fewer than 200 items
    assert I.shape == (20, 10) and I.min() >= 0 and I.max() < 2_000
    assert fanarvse.recall(idx.search(Q, 10, params=fanarvse.SearchParams(64, 400)), gt) > 0.99
