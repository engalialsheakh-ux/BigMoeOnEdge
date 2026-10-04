# MAX_SPEED endgame ledger

All runs below use the same Qwen3.6-35B-A3B Q4_K_M GGUF, greedy `Hello`
prompt, context 1024, ubatch 64, full CUDA dense/KV placement, 3000 MiB RAM
cache, 2048 MiB VRAM cache, four I/O lanes, overlap, anonymous dense weights,
and strict mapping release unless a row changes one of those settings. Raw
stdout/stderr, route CSV, and diagnostic replay utility remain under ignored
`build/max-speed/`; no local model path is copied into this document.

| Gate/candidate | Evidence | Decision |
| --- | --- | --- |
| P0 Release/backend | Release CUDA build targets architecture 86; CPU backend reports AVX2/FMA/F16C. The selected 6 GiB CUDA device and GPU KV are confirmed in real-model logs. | Keep the current native build. |
| P0 Flash Attention | 64-token screens: explicit on 5.236, explicit off 5.135, auto 5.368 tok/s, with the same 147.07 MiB/token streamed. A later six-thread/window-4 pair measured off 7.012 versus auto 7.035 tok/s, with different generated trajectories. | Retain auto; off showed no reliable gain. |
| P0 CPU affinity | At 128 tokens and six CPU threads: unrestricted 5.939 versus first-twelve-logical-CPU mask `0xFFF` 7.585 tok/s. The 64-token unrestricted/affined pair was 5.454/6.902. Expert counts and SSD bytes matched. `--cpu-affinity-mask` exposes the measured mask in the Windows CLI. | Keep the opt-in mask for this machine; the final 256-token combined configuration has only a small rate edge. |
| P1 RAM 4096 MiB | At 64 tokens with six threads and the same mask: 4096 MiB gave 6.712 tok/s and 134.89 MiB/token SSD; 3000 MiB gave 6.884 tok/s and 147.07 MiB/token. The larger cache increased the process working set by about 1.1 GiB and offered no rate gain. | Retain 3000 MiB; no larger RAM budget without more headroom. |
| P2 CUDA route trace | 256 decode steps × 40 layers × top-8 = exactly 81,920 rows. `gpu_hit` sums to 42,604, matching runtime counters. The prefill added 320 rows. | Trace is usable for offline replay; traced tok/s is excluded from benchmarks. |
| P2 replay | Current repeat-8/LRU model predicts 42,760 GPU hits versus 42,604 observed (0.37% error). Plain LRU 40,987, LFU 38,857, HOT+STAGE 41,112; these lose. Repeat-4 predicts 42,581 hits with 8,208 MiB H2D versus 9,560 MiB for repeat-8. A bounded first-half-trained nonuniform layout did not improve held-out hits. | Screen repeat-4 live; reject the three losing policies and tested nonuniform layout. |
| P2 repeat-4 live | At 64 tokens, repeat-8 6.408 versus repeat-4 6.624 tok/s; at 128, 7.542 versus 7.585. H2D fell 41.29 to 36.76 MiB/token and SSD 124.07 to 121.02 MiB/token at 128; expert accounting remained exact. | Keep window 4 as the local opt-in configuration: final 256-token B reduced H2D/SSD and had a small rate edge; default 8 preserves Phase 2 behavior. |
| P2 larger VRAM | 2048 MiB succeeded with a 4307 MiB sampled dedicated peak and 80 MiB shared. A 2304 MiB request was rejected by the existing 768 MiB dedicated-memory safety rail before generation. | Stop the upward pool search at 2048 MiB. |
| P2 persistent hot-start | A separate 128-token calibration prompt seeded 1,173 expert bundles in replay of the `Hello` trace. It added 1,386 hits during the first 128 tokens, but 1,330 occurred in the first 32 and the last 32 gained none. Seeding would preload roughly 2 GiB from storage and upload it before decode. | Reject persistent hot-start for steady decode; its large startup transfer buys no sustained rate gain. |
| P3 packed upload | Slot-major packed bundles cut 128-token upload wall time from 1112 to 746 ms and copies from three to one per promotion. The exact same hits, misses, and SSD bytes yielded 6.740 versus 7.585 tok/s; the graph compute component rose 0.064 to 0.080 s/token. Reverting restored a 64-token rate of 6.872 after a packed 5.951. | Reject the packed slot layout: it slows CUDA expert compute more than it saves H2D dispatch. Retain the existing layout and a `gpu_upload_calls` diagnostic. |
| P4 existing CUDA fusion | The pinned CUDA backend already has a quantized `MUL_MAT_ID` + `GLU` fused path. A temporary backend probe on the real model reported the exact expert branch eligible. Reordering the graph's gate/up construction passed focused correctness tests but did not improve a 64-token screen, so it was reverted. | Use the existing fusion; no new kernel is justified by this probe. |
| P5/P6 MTP | The local GGUF metadata and tensor list contain no NextN/MTP field or tensor. The target model's exact MTP gate requires a compatible GGUF carrying the trained head. | External asset required for MTP; continue other exact gates. |
| P7/P8 SSD economics | A 64-token CUDA run with direct I/O and strict mapping release read 146.01 MiB/token and stalled 0.072 s/token. The I/O trace counted 258 reads/token, all 512 KiB, with only 255 contiguous runs/token and 0.7% alignment waste. A same-file direct microbenchmark under six CPU-burning threads measured 788/1962/989 MiB/s at 1/4/8 lanes. Four lanes already deliver about four times the engine trace's ~480 MiB/s per-lane busy rate; eight lanes regress. | Reject read coalescing, more lanes, and predictive extra SSD reads on the current evidence. Retain four direct lanes and strict mapping release. |
| P0 AVX-VNNI | The MSVC Release CPU backend compiled with AVX-VNNI as well as AVX2/FMA/F16C. Matched 64-token, six-thread affined screens measured 6.404 tok/s with AVX-VNNI and 6.405 without; expert hits, misses, SSD and upload bytes were identical. | Reject AVX-VNNI for this path; restore the AVX2 build. |
| P2 event trace | The v3 diagnostic trace records route rows plus candidate (`C`), staged (`Q`) and published (`P`) GPU cache events with bundle bytes and eviction flag. A 256-token run contained 81,920 decode route rows and 42,457 GPU hits, matching runtime; 4,856 published promotions, 3,683 evictions and 8,194.50 MiB of published bundles also matched runtime counters. | Trace gate complete; keep v3 diagnostics. |

## Measurement environment

An initial interleaved 256-token sequence suffered a power-state change: Windows
reported battery discharge, and later rates fell from about 7 to 5 tok/s with
unchanged expert and I/O counters. Those rates are retained only as diagnostic
evidence, not a final performance conclusion. The machine was subsequently
connected to AC power before the AVX-VNNI comparison and final sequence below.

## Final AC-powered exact comparison

The final AVX2 Release binary used the same model, prompt `Hello`, 256 greedy
tokens, context 1024, ubatch 64, 3000 MiB RAM cache, 2048 MiB GPU pool, four
I/O lanes, full CUDA dense/KV placement, overlap and strict mapping release.
Windows reported AC power at the start and end of every valid run. A used the
Phase 2 equivalent four CPU threads, unrestricted affinity and admission
window 8; B used six threads, affinity mask `0xFFF` and window 4. Order was
`A B B A A B`.

| Run | tok/s | GPU hits | CPU misses | SSD MiB/token | H2D MiB/token | H2D calls/token | Peak working/private MiB | Dedicated/shared GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 6.988 | 42,604 | 39,316 | 140.35 | 37.11 | 65.98 | 3768 / 8737 | 4307 / 80 |
| B1 | 7.088 | 42,457 | 39,463 | 134.29 | 32.01 | 56.91 | 3771 / 8720 | 4307 / 80 |
| B2 | 7.161 | 42,457 | 39,463 | 134.29 | 32.01 | 56.91 | 3769 / 8718 | 4307 / 80 |
| A2 | 7.138 | 42,604 | 39,316 | 140.35 | 37.11 | 65.98 | 3772 / 8809 | 4307 / 80 |
| A3 | 7.012 | 42,604 | 39,316 | 140.35 | 37.11 | 65.98 | 3767 / 8715 | 4307 / 80 |
| B3 | 7.116 | 42,457 | 39,463 | 134.29 | 32.01 | 56.91 | 3772 / 8719 | 4307 / 80 |

Median A is 7.012 and median B is 7.116 tok/s (+1.48% same-binary). B is
only +0.48% above the previous Phase 2 median of 7.082 tok/s; the improvement
is within run variation, so no speedup beyond Phase 2 is established. A varied
2.15% and B 1.03%, below the protocol's 10% expansion threshold. Every run
accounted for 81,920 selected contributions. B GPU hit rate was 51.83%, CPU
misses 154.15/token, promotions 18.97/token and evictions 14.39/token. Minimum
sampled available RAM in B was 2,154 MiB. Sampled process GPU shared usage
stayed at 80 MiB in both arms; no attributable spill was observed. A later
single 256-token six-thread affined/window-8 screen gave 6.758 tok/s and
restored the higher 140.35 MiB/token SSD and 37.11 MiB/token H2D costs.

The main sustained bottleneck remains the exact CPU miss and SSD path; window
4 reduces transfers but does not make the overall improvement large enough to
distinguish confidently from local rate variation. A matched second SODIMM is
an external bandwidth/capacity hypothesis, not part of this software result.
The processor [supports two memory channels](https://www.intel.com/content/www/us/en/products/sku/232101/intel-core-i713650hx-processor-24m-cache-up-to-4-90-ghz/specifications.html),
while the tested laptop currently has one populated SODIMM; no second-module
benchmark was performed.

The offline replay's held-out second-half hits were 22,061 for repeat-8 and
22,263 for repeat-4. Repeat-8 simulated 5,665 promotions, 4,492 evictions,
1,542 zero-reuse entries and 0.224 H2D MiB per useful hit; repeat-4 simulated
4,864, 3,691, 1,159 and 0.193 respectively. The eight donor layers selected
by first-half marginal loss were 9, 1, 2, 0, 4, 6, 5 and 22; the eight proposed
receivers were 29, 13, 33, 36, 31, 10, 23 and 12. The tested nonuniform layout
lost second-half hits (21,997 versus 22,061), so uniform slot allocation stayed.

## Separate TURBO_MAX screen

The existing explicit `--n-expert-used 6` knob changes the model's routed top-k
from 8 to 6. It is not EXACT_MAX and its output quality was not evaluated here.
With all other B settings unchanged, three AC-powered 256-token runs measured
10.391, 10.746 and 10.628 tok/s (median 10.628, about 49.4% above B). Each
counted 61,440 selected contributions, 34,532 GPU hits and 26,908 CPU misses;
SSD was 85.17 MiB/token, H2D 26.31 MiB/token and sampled dedicated/shared GPU
usage 4305/80 MiB. The faster rate is a semantic tradeoff, not an exact gain.

## Regression

- CUDA Release build succeeded with parallel MSBuild after Windows test runtime
  directories were isolated. Plain `ctest --test-dir build-cuda -C Release
  --output-on-failure` passed 24/24 in 298.77 s, including both opt-in real Qwen
  integration tests.
- CPU Release build and plain CTest passed: 17 passed, five GPU-only skipped,
  zero failures (22 registered tests, 130.28 s).
- clang-format 18 dry-run on modified C/C++ and `git diff --check` passed.

## Remaining external gate

The local Qwen GGUF contains no trained NextN/MTP head. Exact MTP and adaptive
MoE-cost-aware MTP require a compatible GGUF with that head; the current model
was not modified or replaced.
- Optional TURBO_MAX is separate and has not been evaluated.
