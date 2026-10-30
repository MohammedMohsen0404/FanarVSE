"""Benchmark FanarVSE on a dataset stored as .npy files: build time, GPU memory, and recall@10 / QPS per preset.

    python benchmarks/bench_npy.py BASE.npy QUERIES.npy [GROUND_TRUTH.npy] [--batch 500] [--save index.pt]

BASE: (N, d) float16/float32 vectors; QUERIES: (B, d); GROUND_TRUTH: (B, >=10) int ids of the exact top-10 by inner
product (computed on the GPU if omitted). On 998k OpenAI-ada-002 vectors (the paper's main set) the presets give
recall@10 of about 0.982 / 0.992 / 0.996 / 0.999.
"""

import argparse
import json
import time

import numpy as np
import torch

import fanarvse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("queries")
    ap.add_argument("ground_truth", nargs="?")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--save")
    a = ap.parse_args()

    X = np.load(a.base, mmap_mode="r")
    Q = np.load(a.queries).astype(np.float32)
    gt = np.load(a.ground_truth) if a.ground_truth else fanarvse.ground_truth(X, Q, 10)
    idx = fanarvse.Index.build(np.ascontiguousarray(X))
    mem = idx.memory_report()
    print(
        json.dumps(
            {
                "n": idx.n,
                "dim": idx.dim,
                "cells": idx.n_cells,
                "build_s": round(idx.build_seconds, 1),
                "gpu_bytes_per_vector": round(mem["gpu_bytes_per_vector"], 1),
            }
        ),
        flush=True,
    )
    if a.save:
        idx.save(a.save)
    for preset in fanarvse.PRESETS:
        I = idx.search(Q, 10, preset=preset, batch_size=a.batch)  # also warms up the CUDA graphs
        times = []
        for _ in range(a.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            idx.search(Q, 10, preset=preset, batch_size=a.batch)
            times.append(time.perf_counter() - t0)
        print(
            json.dumps({"preset": preset, "recall@10": round(fanarvse.recall(I, gt, 10), 4), "qps": round(len(Q) / min(times))}), flush=True
        )


if __name__ == "__main__":
    main()
