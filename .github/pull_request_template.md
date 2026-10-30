## What and why

<!-- What does this change, and why? Link the issue if there is one. -->

## Checks

- [ ] `pytest` passes on a GPU (which GPU, OS, driver?):
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `CHANGELOG.md` updated under "Unreleased"

## Benchmark (kernels, search path or build)

<!-- Output of `python benchmarks/bench_npy.py BASE.npy QUERIES.npy GROUND_TRUTH.npy` before and after,
     with the dataset and GPU. Recall must not drop; throughput must not drop beyond noise unless intended. -->
