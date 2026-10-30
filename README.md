# FanarVSE

**Memory-light GPU vector search with error bars from coding theory.**

FanarVSE is a GPU index for maximum inner-product (and cosine) search over embeddings. It keeps about 570 bytes per vector on the GPU, under a fifth of what graph indexes need, and leaves the full vectors in host RAM. Each candidate's score comes with an error bar. Only the few candidates whose bars overlap the top-k boundary are scored exactly, usually about six per query, read straight from host memory by the GPU.

| 998k OpenAI-ada-002 vectors (1536-d), one RTX 3050 Laptop GPU (6 GB), batch 500 | recall@10 | queries/s | GPU memory |
|---|---|---|---|
| FanarVSE, preset `fast` | 0.982 | 26,964 | 570 B/vector |
| FanarVSE, preset `balanced` | 0.992 | 24,879 | 570 B/vector |
| FanarVSE, preset `high` | 0.996 | 21,512 | 570 B/vector |
| FanarVSE, preset `max` | 0.999 | 13,558 | 570 B/vector |
| CAGRA (cuVS), best configuration near the same recall | 0.991 | 4,421 | 3,200 B/vector |

The method and the full evaluation, against CAGRA, IVF-RaBitQ, IVF-PQ and FAISS on eight datasets, are in the paper: *FanarVSE: Fast, Memory-Light GPU Vector Search with Error Bars from Coding Theory* (Mohammed Mohsen). Its experiment scripts and raw results are archived in the separate **FanarVSE-paper** repository.

## Install

FanarVSE needs an NVIDIA GPU, Python 3.10 or newer, and PyTorch with CUDA.

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/MohammedMohsen0404/FanarVSE
```

On Windows this installs `triton-windows`; on Linux, Triton comes with PyTorch.

## Quickstart

```python
import numpy as np
import fanarvse

X = np.load("embeddings.npy")  # (N, d) float16 or float32
index = fanarvse.Index.build(X)  # about 30 s for 1M x 1536 on a laptop GPU

ids = index.search(Q, k=10)  # (B, 10) int64 row ids of X, best first
ids, scores = index.search(Q, k=10, preset="high", return_scores=True)

index.save("my.index")
index = fanarvse.Index.load("my.index")
```

For cosine similarity, pass unit vectors or build with `normalize=True`. The full example is [`examples/quickstart.py`](examples/quickstart.py).

### Recall and speed

The presets trade throughput for recall. Their recall was measured on OpenAI embeddings, so check it on your own data with `fanarvse.ground_truth` and `fanarvse.recall`.

| preset | cells probed (of 4,096) | candidates | recall@10 on OpenAI-1M |
|---|---|---|---|
| `fast` | 256 | 200 | 0.982 |
| `balanced` (default) | 384 | 400 | 0.992 |
| `high` | 768 | 400 | 0.996 |
| `max` | 1,536 | 800 | 0.999 |

For full control, pass `params=fanarvse.SearchParams(nprobe=..., candidates=..., z=...)`. Throughput grows with the batch size (`batch_size`, default 1,024), because queries that probe the same cell share its scan.

## How it works

1. **Stage 1, filter (GPU).** Vectors are grouped into cells by spherical k-means. Each vector's offset from its cell mean is projected on 256 principal directions and stored as a 1-bit bitshift trellis code (42 B with its scale and id). Trellis codes come within a fraction of a decibel of the Shannon bound. A threshold-and-emit kernel returns the exact top-K estimates over the probed cells.
2. **Stage 2, refine (GPU).** A tiered refinement code (504 B) estimates each candidate's score and its standard deviation. The estimate is unbiased for any codebook after a random rotation, with an exact error law (Theorem 1 of the paper).
3. **Stage 3, error-bar check (host RAM).** Candidates whose interval `estimate ± z·σ` straddles the top-k boundary are scored exactly from the full fp16 vectors in pinned host memory. If every interval covers its true score, the result is exactly the top-k of the candidates.

The whole search for a batch is one CUDA graph. [`docs/design.md`](docs/design.md) maps each module to the paper.

## Status and limits

Version 0.1 is the paper's default engine as a library: build, search, save and load.

- **Not yet supported:**
  - adding or deleting vectors after the build;
  - the paper's throughput, host-resident and NVMe modes;
  - L2 distance;
  - multiple GPUs.
- **Tested** on Windows 10 and on Linux (Ubuntu under WSL2), both with an RTX 3050 Laptop GPU. The benchmark numbers above are from Windows.
- **Memory:** the full vectors stay in pinned host RAM, 2 bytes per dimension per vector.

See [`CHANGELOG.md`](CHANGELOG.md) and the roadmap in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, the tests a pull request must pass, and how to show that a change keeps recall and speed.

## Citation

If you use FanarVSE, please cite the paper (full reference to be added on publication):

```
Mohammed Mohsen. FanarVSE: Fast, Memory-Light GPU Vector Search with Error Bars from Coding Theory.
```

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
