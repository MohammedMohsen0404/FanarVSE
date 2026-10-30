"""Build an index, search it, and measure recall against exact search.

The data are synthetic clustered unit vectors, a rough stand-in for real embeddings (uniformly random vectors have
no neighbourhood structure, so no approximate index can do well on them). Replace X and Q with your own data.
"""

import numpy as np

import fanarvse

rng = np.random.default_rng(0)
n, d = 200_000, 768
centres = rng.standard_normal((2000, d)).astype(np.float32)
X = centres[rng.integers(0, len(centres), n)] + 0.7 * rng.standard_normal((n, d)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)  # unit vectors: inner product = cosine similarity
Q = X[rng.integers(0, n, 1000)] + 0.02 * rng.standard_normal((1000, d)).astype(np.float32)

index = fanarvse.Index.build(X)
print(f"built in {index.build_seconds:.1f} s, {index.memory_report()['gpu_bytes_per_vector']:.0f} B/vector on the GPU")

ids = index.search(Q, k=10)  # preset "balanced"
ids, scores = index.search(Q[:5], k=10, preset="high", return_scores=True)
print("first query:", ids[0][:5], scores[0][:5])

truth = fanarvse.ground_truth(X, Q, k=10)
for preset in fanarvse.PRESETS:
    print(f"{preset:9s} recall@10 = {fanarvse.recall(index.search(Q, 10, preset=preset), truth):.4f}")

index.save("quickstart.index")
index = fanarvse.Index.load("quickstart.index")
