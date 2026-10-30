import numpy as np
import pytest
import torch

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count() > 0), reason="needs a CUDA GPU")


def clustered(n, d, nq, seed=0, clusters=200, spread=0.6):
    """Unit vectors around random centres, and queries near data points."""
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((clusters, d)).astype(np.float32)
    X = C[rng.integers(0, clusters, n)] + spread * rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = X[rng.integers(0, n, nq)] + 0.3 * rng.standard_normal((nq, d)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return X, Q


@pytest.fixture(scope="session")
def data():
    return clustered(30_000, 160, 300)


@pytest.fixture(scope="session")
def index(data):
    import fanarvse

    return fanarvse.Index.build(data[0], fanarvse.IndexConfig(n_cells=128))
