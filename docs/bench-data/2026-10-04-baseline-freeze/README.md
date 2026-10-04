# Exact 64K inference baseline freeze — 2026-10-04

This directory records the completed Windows/CUDA exact-inference phase. The reference is
the existing 65,507-token Qwen3.6-35B-A3B Q4_K_M run. No model inference or 64K benchmark
was repeated while preparing this freeze. The model and raw traces remain local.

## Measured reference

| Quantity | Result |
| --- | ---: |
| Prompt tokens | 65,507 |
| Decode | 9.389 tokens/s |
| Prefill | approximately 15.77 tokens/s |
| Prefill elapsed | approximately 4,153.5 s |
| Prefill expert rereads | 3,334,981 |
| RAM expert cache | 4,400 MiB |
| GPU expert cache | 1,500 MiB |
| Dedicated VRAM safety rail | 128 MiB |
| CUDA KV cache | Q4_0 K / Q4_0 V |
| KV allocation | approximately 1,440 MiB |
| CUDA compute buffer | approximately 70.3 MiB |
| Direct quantized CUDA FlashAttention | active |
| F16 FlashAttention scratch | 0 bytes |
| Demand I/O coalescing | enabled |

These numbers describe one existing full-length run, not an average or a new result. The
model is `Qwen3.6-35B-A3B-Q4_K_M.gguf`, architecture `qwen35moe`, with exact top-8 routing
and unchanged Q4_K_M weights. The source for the reference values is the previously recorded
local baseline and the [endgame analysis](../2026-10-03-exact-endgame/README.md). The
model SHA-256 is in [model.sha256](model.sha256). The deterministic prompt recipe and byte
hash are in [prompt-recipe.ps1](prompt-recipe.ps1) and [prompt.sha256](prompt.sha256).

Use [final-config.txt](final-config.txt) for the reference flags. In particular, keep GPU KV
enabled: **do not use `--no-kv-offload`**. The quantized CUDA attention path is only
established for the recorded placement and environment. `--io-coalesce` and the 128 MiB
safety rail are explicit choices; their omitted defaults differ.

## Replay results and rejected experiments

The following are separate from the measured 64K result. Their detailed methods, controls,
and limits are in the [expert I/O report](../2026-10-03-expert-io/README.md) and
[endgame analysis](../2026-10-03-exact-endgame/README.md).

| Candidate | Evidence and decision |
| --- | --- |
| RAM cache policies | Keep 4,400 MiB and the current prefill retention/decode LRU. Segmented, frequency, layer-quota, and history alternatives gave no material net benefit on the short access trace. |
| GPU per-layer allocation | A nonuniform layout lost 150 held-out hits in the 256-token route replay. Rejected. |
| RAM-to-GPU route prediction | The best replay reduced misses about 0.20% while requesting about 16,883 MiB of speculative uploads. Rejected. |
| Paired sidecar | Held-out simulation predicted 30.70% fewer read windows. The corrected short real-model prototype cut physical windows 34.85%, but prefill worsened from about 197.1 to 219.7 s and decode from about 6.778 to 6.418 tokens/s. The production sidecar path was removed; its builder and replay remain isolated research tools. |
| DirectStorage | Isolated probe only, never linked into production. Single-request latency lost at 512 KiB, 1 MiB, and 2 MiB. A local 4 MiB median advantage did not match the production RAM-to-GPU promotion path. Rejected for integration. |
| CUDA graph/launch work | Nsight Systems observed about 81 graph launches and about 6 ms host launch-call time per token after warmup in a short diagnostic. Roughly 40 routing round trips are structurally required. An optimistic removable bound is about 3 ms/token, below the phase's material-gain threshold. Rejected. |
| CPU/GPU expert concurrency | Relevant branches currently execute serially. Instrumented short-run timing did not justify a safe scheduler change. Future architectural work. |
| Expert-centric prefill | Short-trace simulation gave a 94.44% ideal I/O reduction. Current `llama_decode` ordering has no legal layer-major implementation seam for it. No production change; next architectural phase candidate. |
| Partial GPU expert residency | The optimistic avoided-CPU-work replay lost to complete bundles at each tested fraction. Rejected before production implementation. |

The 94.44% and similar prefill scaling figures are theoretical or replay outcomes, **not**
measured 64K speedups. Short real-model diagnostics are labelled as such above.

## Validation and reproducibility

The final optimization tree previously reached a CUDA Release build, 30/30 BigMoeOnEdge
CTest passes, `git diff --check`, and clang-format 18 checks. During this freeze, only
repository consistency, hashes, and metadata were checked; no full suite or model inference
was rerun. Old llama.cpp CTest registrations in one generated build tree referred to tests
not built with `LLAMA_BUILD_TESTS=OFF`; that stale build metadata was removed, and was not a
production regression. The directly relevant `config_validate` and `gpu_expert_pool` tests
also passed after the CUDA Release build on 2026-10-04.

See [environment.txt](environment.txt) for observed host/toolchain facts,
[git-state.txt](git-state.txt) for source revisions, and
[baseline-manifest.json](baseline-manifest.json) for machine-readable measured and replay
categories. Local paths in the configuration identify this host only; substitute the model
path on another machine, then verify its SHA-256 before comparing results.
