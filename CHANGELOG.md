# Changelog

## Unreleased

## 0.1.0

First release: the default engine of the FanarVSE paper as a library.

- `Index.build`, `Index.search` (presets and explicit `SearchParams`, any batch size, optional exact scores), `Index.save` / `Index.load`, `Index.memory_report`.
- Stage 1: cell-residual bitshift trellis codes with exact threshold-and-emit selection and an exact fallback.
- Stage 2: tiered refinement code with error bars; stage 3: exact scores from pinned host memory.
- Tests of the paper's guarantees, a benchmark script, CPU CI.
- On 998k OpenAI-ada-002 vectors: 570 B/vector on the GPU, recall@10 0.992 at 24,879 queries/s (RTX 3050 Laptop, batch 500).
