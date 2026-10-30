# Design

This note maps the code to the paper and records the decisions a contributor needs to know.

## Data layout

After the build, vectors are sorted by cell. "Positions" below index this sorted order, and `Index.order[pos]` gives the original row id.

| Array | Where | Size per vector | Purpose |
|---|---|---|---|
| `s1.words` | GPU, int32 (N, W) | 36 B for R = 256 | packed trellis codes of W1ᵀ(x − m_c) |
| `s1.scale` | GPU, fp16 (N,) | 2 B | per-vector scale of the stage-1 estimate |
| `order` | GPU, int32 (N,) | 4 B | sorted position → original id |
| `code.planes` | GPU, uint8 per tier | ≈ 480 B for d = 1536 | bit planes of the tiered refinement code |
| `code.scale`, `code.sig` | GPU, fp16 (T, N) | 4 B per tier | per-tier scale and error scale |
| `host` | pinned host RAM, fp16 (N, d) | 2d B | full vectors for exact scores |

Shared tables on the GPU:
- fp16 cell means, which also route queries;
- the stage-1 projection `W1`;
- the query-side rotation `VP` of the refinement code;
- the trellis table;
- the Lloyd-Max codebooks.

## Search path

For a batch of queries, `Index._gpu` runs:

1. `QM = Q · means16ᵀ`. Routing takes the top `nprobe` cells by `QM / |m_c|`, and `QM` also supplies the per-cell constant ⟨q, m_c⟩ of the stage-1 estimate.
2. `stage1.select`:
   - batches above 32 queries use a fixed work list over all (cell, block) pairs (`select_batch`);
   - smaller batches use a (query × cell) probe mask (`select_small`).
   Both guess a threshold from a first pass over the nearest cells, emit every item above it, and flag queries whose count shows the guess failed.
3. `refine.score`: estimate and σ for every candidate.
4. `Index._rank`:
   - a candidate is **surely in** if its lower bound beats the (k+1)-th upper bound;
   - it is **ambiguous** if its upper bound reaches the k-th lower bound.
   Ambiguous candidates are scored exactly by `rerank.exact_scores`, reading pinned host memory over PCIe. The result is the surely-in candidates followed by the best ambiguous ones.

Steps 1–4 contain no host synchronisation, so `Index._run` records them as one CUDA graph per (batch size, parameters, k). After the replay, flagged queries (usually under 1%) are recomputed eagerly by `stage1.exact`, which guarantees an exact top-K.

`Index.search` splits queries into batches of `batch_size` and pads the last batch to a power of two, so that a few graphs serve every request size. `Index.clear_cache()` releases the graphs.

## Guarantees and where they are tested

| Statement | Code | Test |
|---|---|---|
| Stage 1 returns the exact top-K of its estimates over the probed cells | `stage1.select` + `stage1.exact` | `test_stage1_selects_the_exact_top_K` |
| The refinement estimate is unbiased, and σ bounds its spread | `refine.build`, `refine.score` | `test_refinement_estimate_is_unbiased_with_calibrated_error_bars` |
| Every batch size gives the same result | `Index.search` | `test_every_batch_size_gives_the_same_result` |
| The fused encoder finds the minimum-distortion path | `trellis.encode` | `test_fused_encoder_equals_reference_viterbi` |

If every interval covers its true score, the result is exactly the top-k of the candidates (paper, Lemma 4). With the default z = 2 this is not certified, but in the paper's measurements it lost no neighbour that stage 1 had found. For a per-query failure bound δ over K candidates, use z with 2K·exp(−z²/2) ≤ δ (z ≈ 4.9 for δ = 0.5%, K = 400).

## Differences from the research code

The library builds the same index as the paper's final engine, with four simplifications:

- The stage-1 code is encoded once, directly against the cell means. The research build first encoded against a global centre and then re-encoded against the cell means; the final code has the same construction.
- Lloyd-Max codebooks are computed deterministically by quadrature instead of by k-means on samples, which removes the FAISS dependency.
- Surely-in candidates rank first with an offset derived from the data, not a fixed `+10`, so unnormalised vectors work.
- All state lives on the `Index` object, and the inheritance chain of engine versions is gone.

On OpenAI-1M the library matches the paper's engine: 569.7 B/vector, recall@10 0.982–0.999 for the four presets, and equal or higher throughput.
