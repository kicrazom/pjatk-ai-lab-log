# Benchmark Results — Embargo Policy

This directory holds outputs from `benchmarks/scripts/` runs.

## Public (`hardware_envelope/`)

Released immediately to the public repository. Contents:

- Sanity load configurations (model loads / does not load)
- VRAM consumption per GPU at given `max_model_len` and `gpu_memory_utilization`
- Required environment variables and engineering workarounds
- Load times, KV cache size at load
- Sanity throughput (single-prompt smoke test)

These are **engineering facts** needed for reproducibility. They cannot be
"scooped" — they are configuration data, not research findings.

## Embargoed (`scaling/`)

Held local-only until publication of the corresponding paper
(target venue: IEEE Access; arXiv preprint precedes journal submission).
Files in `scaling/` are git-ignored except `.gitkeep`. Contents:

- Full scaling tables: throughput at N ∈ {10, 25, 50, 100, 200, 500, 1000}
- Latency distributions: P50, P95, P99 per concurrency level
- KV cache utilization curves vs concurrency
- Cross-model comparative claims (e.g. model A vs model B at fixed N)
- Scaling laws and quantization tradeoff interpretations
- Plots and figures intended for paper

## Rationale

This split protects priority on research findings while making the
infrastructure layer fully reproducible. Anyone reading the eventual
paper can reconstruct the experimental setup from the public repository.

For Polish-language specialized models (Bielik, PLLuM), the embargo is
applied more strictly than for general-purpose baselines (Qwen, Mistral,
Mixtral) due to the smaller and more concentrated nature of the Polish
AI research community.

## Release schedule

| Phase | Release condition |
|-------|-------------------|
| Hardware envelope | Continuous — pushed with each benchmark session |
| Scaling data, plots | With arXiv preprint (cs.DC primary) |
| Full reproducibility bundle | With IEEE Access publication |
