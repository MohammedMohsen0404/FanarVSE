# Contributing to FanarVSE

Thank you for helping. This guide covers the development setup, what a pull request needs, and where help is most useful.

## Development setup

```
git clone https://github.com/OWNER/FanarVSE
cd FanarVSE
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

## Before you open a pull request

1. **Tests pass on a GPU:** `pytest`. The tests in `tests/test_gpu.py` check the paper's guarantees directly:
   - stage 1 returns the exact top-K of its estimates;
   - the refinement estimate is unbiased with calibrated error bars;
   - every batch size gives the same result;
   - save/load round-trips.
   CI runs only the CPU tests (GitHub's runners have no GPU), so please run the GPU tests locally and say on which GPU.
2. **Lint and format:** `ruff check .` and `ruff format .`.
3. **Performance-sensitive changes** (kernels, search path, build) need a before/after benchmark:
   ```
   python benchmarks/bench_npy.py BASE.npy QUERIES.npy GROUND_TRUTH.npy
   ```
   Paste both outputs in the pull request: recall@10 per preset, queries per second, GPU bytes per vector. The reference set is the 998k OpenAI-ada-002 DBpedia embeddings (Hugging Face `KShivendu/dbpedia-entities-openai-1M`). A change should not lower recall, and should not lower throughput by more than run-to-run noise (about 3%) unless that is its purpose.
4. **Keep the guarantees.** Stage 1 must stay an exact top-K of its estimates (or flag the query for `stage1.exact`), and the refinement estimate must stay unbiased with its σ. If you change the codes, add a test of the property you rely on.
5. **One topic per pull request,** with a clear description of what changes and why. Add a line to `CHANGELOG.md` under "Unreleased".

## Code layout

| Module | Role | Paper |
|---|---|---|
| `fanarvse/index.py` | `Index`: build, search (CUDA graphs, batching, fallback), save/load | Section 5 |
| `fanarvse/config.py` | `IndexConfig`, `SearchParams`, presets | Table 3 |
| `fanarvse/coarse.py` | spherical k-means cells | Section 5 |
| `fanarvse/trellis.py` | bitshift trellis table, Viterbi encoder, bit packing, decoder | Section 4 |
| `fanarvse/stage1.py` | threshold-and-emit selection kernels, exact fallback | Section 5, Lemma 3 |
| `fanarvse/refine.py` | tiered refinement code and fused scoring kernel | Section 5, Corollary 3 |
| `fanarvse/rerank.py` | exact scores from pinned host memory | Section 5 |
| `fanarvse/_tables.py` | Lloyd-Max codebooks | Section 5 |

`docs/design.md` explains the data layout and the search path in more detail.

## Where help is most welcome

- Other GPUs and native Linux machines: run the tests and the benchmark, and report results.
- Adding and deleting vectors after the build.
- The paper's throughput mode (int8 stage 1, lookup-table stage 2), which is about 2.7× faster at large batches.
- The host-resident and NVMe modes (under 8 B and 66 B of GPU memory per vector).
- L2 distance (the reduction is in Section 5.2 of the paper).
- Multi-GPU and sharded indexes.

Please open an issue to discuss larger changes before you start.

## Reporting bugs

Open an issue with your GPU, driver, OS, Python, PyTorch and Triton versions, the dataset shape, and a minimal script that reproduces the problem.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
