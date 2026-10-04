# CUDA expert-cache goal log

## Scope

Implement and verify an exact SSD/RAM/VRAM MoE path with mixed GPU hits and CPU misses.
The original local goal contract is retained outside the committed project artifacts.
This log records baseline, implementation, and final verification evidence;
no performance or correctness claim is inferred from a previous release's measurements.

## TODO and gate status

- [x] Gate 0: current checkout, baseline tests, memory, and 256-token CUDA run.
- [x] Gate 1: design audit against pinned APIs and current expert-cache research.
- [x] Gate 2: bounded VRAM pool and safe slot lifetime.
- [x] Gate 3: exact upload and all-hit GPU execution.
- [x] Gate 4: background promotion from bounded staging.
- [x] Gate 5: exact mixed GPU-hit / CPU-miss execution.
- [x] Gate 6: SSD-to-RAM-to-VRAM hierarchy.
- [x] Gate 7: cache policy and prefill protection.
- [x] Gate 8: real-model correctness.
- [x] Gate 9: oversize allocation, stale-lease, teardown, and full-suite checks.
- [x] Gate 10: interleaved performance comparison.

## Current checkout

- BigMoeOnEdge HEAD: `864ea87` (`feat: add hybrid CUDA dense and KV offload`).
- Pinned llama.cpp submodule: `dce9698510b76746db625b644ef2af899517d0ad`.
- Initial worktree: `.gitignore` modified; goal contract untracked. Both are user-owned
  and must remain untouched.
- No GPU expert cache implementation or CLI option exists in the current checkout.
- Current CUDA build tree registers 18 tests, including the opt-in real-model gate.

## Baseline commands and results

| Gate | Command | Result |
| --- | --- | --- |
| CUDA tests | `ctest --test-dir build-cuda -C Release --output-on-failure` | 18/18 passed; 239.94 s, including the real-model gate (85.34 s). |
| Fresh baseline | `bmoe-cli -m MODEL -p "Explain how a mixture-of-experts model routes tokens and manages expert weights across storage and memory." -n 256 -c 1024 -t 4 --ubatch 64 --moe-stream --dense-weights anon --release-mmap --require-release-mmap --cache-mb 3000 --cache-ceil-mb 3000 --io-threads 4 --overlap --n-gpu-layers -1 --csv baseline.csv` | 256 tokens; 6.352 tok/s, 0.157 s/token; prefill 3.401 s, TTFT 16.610 s; 73.1% RAM cache hit, 134.42 MiB read/token, 0.051 s/token overlap stall. Strict release succeeded: one 19,473 MiB mapping unmapped. |
| Live-memory baseline repeat | Same model, prompt, and flags, sampled with `nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu` and Windows process memory once per second | 256 tokens; 6.443 tok/s, 0.155 s/token; peak dedicated GPU use 2273 MiB (50 samples), peak process working set 3856 MiB, peak private bytes 6764.8 MiB; strict mapping release succeeded. |

Both fresh decode rates lie within the previously documented three-run span of
6.149–7.364 tok/s. They are not a new median. The CUDA model, KV,
and compute allocations logged 1910.32, 20.00, and 153.82 MiB. Raw output is
retained in the ignored `build/goal-expert-cache/` directory until identifying
paths and device details have been removed for a shareable evidence copy.
The memory samples show no unexplained full-bank materialization. Shared GPU
memory was not sampled in this first pass and remains a required final gate.

## Research notes

- The pinned CPU `mul_mat_id` loop has the existing expert-ready hook. The current
  `RouterHook` isolates `ffn_moe_topk` and copies CUDA-backed routing IDs to host.
- Current llama.cpp expert-cache proposals warn that a slot remap must be tied to the
  graph split that consumes it. This is a design constraint, not yet a verified local
  implementation.

## Pool allocation experiment (Gate 2 in progress)

- Added `gpu.expert_cache_mb`, `--gpu-expert-cache-mb`, and a bounded device
  buffer with per-layer projection layout, dummy-slot reservations, slot state,
  generation checks, pin protection, and deterministic teardown. The allocation
  is explicitly labelled `allocation-only` in the runtime log; expert execution
  still uses the original CPU branch. Zero remains the default.
- Reconfigured CUDA with `cmake -S . -B build-cuda -A x64 -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=86`; Release build passed. `ctest --test-dir
  build-cuda -C Release -R 'gpu_expert_pool|config_validate'
  --output-on-failure` passed 2/2. The pool test checks unpublished invisibility,
  pin-protected eviction, stale lease generations, and three allocate/free cycles.
- One-token real Qwen3.6 CLI probe with `--gpu-expert-cache-mb 512`,
  `--n-gpu-layers -1`, `--moe-stream`, 3000 MiB RAM cache, overlap, anonymous
  dense weights, and strict mapping release reserved 512 MiB; the slot layout
  used 511 MiB with 263 real slots. Strict release unmapped 19,473 MiB.
  Raw local log: ignored `build/goal-expert-cache/pool-512.log`.
- The same probe with a 10000 MiB GPU cache exited 1 with `insufficient
  dedicated GPU memory for expert cache and 768 MiB safety rail`. Raw local
  log: ignored `build/goal-expert-cache/pool-oversize.log`.
- Added opt-in `hybrid_gpu_qwen36_pool_allocation`, which passed in 76.70 s.
  Its resident, streamed, and overlap outputs match under CUDA/KV placement,
  and the test emits the 3000 MiB RAM and 512 MiB GPU budgets.
- Gate 2 remains open: shared GPU memory and dedicated peak during the new run
  require interpretation; full CUDA and CPU suites must be rerun after the
  implementation stabilizes. There is no upload or GPU expert execution yet.

### Paired memory and output probe

The same binary ran the real Qwen3.6 model with `-p Hello -n 16 -c 256 -t 2
--ubatch 64 --moe-stream --dense-weights anon --release-mmap
--require-release-mmap --cache-mb 3000 --io-threads 2 --overlap
--n-gpu-layers -1`, first with `--gpu-expert-cache-mb 512` and then without.
Both produced identical generated text and successfully released the model
mapping. Windows GPU process shared-usage samples peaked at exactly
81,788,928 bytes in **both** runs; the new pool added no measured shared usage.
`nvidia-smi` dedicated GPU use peaked at 2755 MiB with the pool and 2243 MiB
without, a 512 MiB delta. The sample cadence was about 2 seconds, with five
and six samples respectively, so these are sampled peaks, not exhaustive
allocator high-water marks. Raw logs and samples are in the ignored
`build/goal-expert-cache/` directory. The observed nonzero Windows shared
counter prevents claiming literal zero shared usage at this stage; it is
baseline-identical and needs attribution before the final memory gate.

### CUDA and CPU regression suites

- `ctest --test-dir build-cuda -C Release --output-on-failure`: 20/20
  passed in 293.24 s, including both Qwen3.6 variants and the new GPU pool
  lifetime test.
- `cmake -S . -B build -A x64 -DGGML_CUDA=OFF`, followed by
  `cmake --build build --config Release --parallel 4` and
  `ctest --test-dir build -C Release --output-on-failure`: build passed;
  16 CPU tests passed and the two GPU-only tests skipped, 128.15 s.

### Exact device-copy and first CUDA kernel probe (Gate 3 in progress)

- Added synchronous full-projection upload. RESERVED slots remain invisible
  until every required projection has been copied; `publish` checks a per-slot
  completion mask. Backend `tensor_set` is synchronous on this pinned CUDA
  implementation. Readback compares all bytes of a patterned bundle.
- Added a CUDA `MUL_MAT_ID` graph using three distinct F32 expert matrices in
  the pool, routed in a different order from their original IDs. Its nine
  output scalars matched the independently calculated result. The command
  `ctest --test-dir build-cuda -C Release -R
  'gpu_expert_(pool|matmul)' --output-on-failure` passed 2/2.
- This establishes exact byte transport and one all-hit projection. It does
  **not** establish a full MoE gate/up/activation/down layer, the CPU/GPU
  mixed merge, per-ubatch remap, or real-model GPU expert execution.
- A second synthetic graph now executes all-hit CUDA gate, up, SiLU-gated
  activation, down, and weighted expert sum from three distinct experts.
  `ctest --test-dir build-cuda -C Release -R '^gpu_expert_moe_layer$'
  --output-on-failure` passed. It compares against a resident-GPU graph
  using the same input and routing, so backend arithmetic semantics match.
- The pinned fork working tree now has one opt-in `ggml-cpu.c` extension:
  `MUL_MAT_ID` checks a host expert-to-GPU-slot table in `src[3]`, zeros the
  CPU destination rows for GPU hits, and leaves the existing expert-ready
  hook to run only for misses. `ctest --test-dir build-cuda -C Release -R
  '^cpu_expert_skip$' --output-on-failure` passed with distinct weights and
  hook-call counts. The extension is inert while `src[3]` is null; no graph
  currently attaches it during real inference. The seam is documented in
  `docs/seam.md`.
- The [llama.cpp draft #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
  uses a second GPU `MUL_MAT_ID` chain, CPU skip table, and weighted output
  merge. The [pool RFC #28248](https://github.com/ggml-org/llama.cpp/discussions/28248)
  documents stale-map corruption through a `RESHAPE` view and a later
  scheduler-reserve invalidation. Any local remap test must use distinct
  expert weights, check route telemetry against ground truth, and rebuild
  graphs across ubatches before it can establish correctness.

### Phase 2 bridge check

- The pool now owns stable metadata tensors and layer descriptors until after
  `llama_context` destruction. The selected GPU was already passed as
  `auto_devices[0]`; that vector contains only the requested `--main-gpu` device.
- The dormant graph bridge compiled. Its first real-model all-miss run exposed
  a host-map tensor shape mismatch in the CPU skip check; the host map now has
  `n_expert` elements in dimension zero, while the CUDA gather table remains
  `[1, n_expert]`.
- With that correction, same-binary one-token Qwen3.6 cache-on and cache-off
  runs both produced `Hello,`, exited successfully, and reported the same
  411.0 MiB of streamed reads. The four focused CUDA/CPU cache tests passed.
- This was an all-miss bridge check before routing maps and promotion were wired.

### Mixed executor and short budget probes

- The top-k callback now pins ready GPU slots, uploads matching host/device maps,
  and stages only CPU misses. The map is cleared after `llama_decode()` returns,
  before pins are released. One-token prefill uses the original graph.
- A focused mixed test uses distinct weights, four CPU threads, all-miss,
  all-hit, and two consecutive mixed maps. It caught a CUDA fallback sorter
  assumption that selected expert IDs were unique; multiple misses share zero
  slot, so the local pinned-fork CUDA loop now retains every occurrence.
- A 64-token real Qwen3.6 run at 512 MiB VRAM cache completed with 5,483 GPU
  hits, 14,997 CPU misses, 2,560 promotions, strict mapping release, and
  10,715.5 MiB of streamed reads. The one-token cache-on all-miss control
  with promotion disabled produced the same first token as cache-off.
- In preliminary 64-token probes, budgets 512/1024/1536/2048 MiB yielded
  5.909/6.135/6.452/6.667 tok/s, respectively, before admission tuning.
  Sampled dedicated memory at 1024/1536/2048 was 3283/3795/4307 MiB.
  The 2048 MiB probe retained over 1.8 GiB of nominal dedicated headroom.
- One frequency-gated admission rule (a miss repeated within eight tokens)
  reduced 2048 MiB upload from 4233.94 to 2878.88 MiB and evictions from
  1336 to 533 in a 64-token probe. Decode rose from 6.667 to about 7.19
  tok/s. Background upload from bounded staging then completed a 64-token
  run with the same hit/miss and byte totals, at about 7.25 tok/s; this short
  result is not the final A/B benchmark.
- A separate CUDA transfer stream with pinned staging was tested after the
  background worker. At 2048 MiB, a 64-token probe slowed to 5.99 tok/s,
  produced fewer hits (8679 versus 9098) and more streamed reads (9412.8
  versus 8945.6 MiB). The independent stream was removed; the bounded worker
  remains, performing complete synchronous H2D copies outside the decode
  caller's thread. It publishes only after all three projections finish.
- A later 64-token smoke on the final admission/worker/prefill path at 2048 MiB
  completed normally with strict mapping release, 8,679 GPU hits and 11,801
  CPU misses (20,480 selected contributions), 1,739 promotions, 566 evictions,
  and 9,412.8 MiB of streamed reads. Both execution branches and SSD fallback
  remained active; this short run was not used for the final rate comparison.

### Final same-binary comparison

After the first-token counter correction and the final code build, six 256-token
greedy runs used the interleaved order `A B B A A B`. A disabled the GPU expert
cache; B used 2048 MiB. Both used the same model, `Hello` prompt, context 1024,
ubatch 64, four CPU threads, four I/O lanes, 3000 MiB RAM cache, overlap,
anonymous dense weights, strict mapping release, full CUDA placement, and GPU KV.

| Run | GPU cache MiB | Decode tok/s | GPU hits | CPU misses | Upload MiB | Streamed read MiB | Sampled dedicated peak MiB | Sampled shared peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 0 | 6.909 | 0 | 0 | 0 | 38813.8 | 2256.7 | 80 |
| B1 | 2048 | 7.082 | 42604 | 39316 | 9500.62 | 35928.8 | 4306.8 | 80 |
| B2 | 2048 | 7.078 | 42604 | 39316 | 9500.62 | 35928.8 | 4306.8 | 80 |
| A2 | 0 | 6.921 | 0 | 0 | 0 | 38813.8 | 2256.7 | 80 |
| A3 | 0 | 6.967 | 0 | 0 | 0 | 38813.8 | 2256.7 | 80 |
| B3 | 2048 | 7.140 | 42604 | 39316 | 9500.62 | 35928.8 | 4306.8 | 80 |

Median A was 6.921 tok/s; median B was 7.082 tok/s, a 2.33% increase.
Within-arm rate spread was 0.84% for A and 0.88% for B, below the 10% threshold
for additional runs. Each B run counted exactly 81,920 selected expert
contributions (256 tokens x 40 layers x top-8): GPU hit rate 52.01%, CPU misses
153.58/token, upload 37.11 MiB/token, and SSD reads 140.35 MiB/token.
A read 151.62 MiB/token. B promoted 5,630 bundles and evicted 4,457 slots.
The peak dedicated memory sample left about 1.8 GiB of nominal 6 GiB VRAM
headroom. The Windows shared-memory counter was identical in both arms.
Sampling can miss brief peaks; these are observed values rather than formal
allocation bounds. The explicit pool limit and bounded staging queue provide
the implementation bounds. Raw local logs remain in ignored `build/` files.

### Final regression

- `cmake --build build-cuda --config Release --parallel 4` passed.
- `ctest --test-dir build-cuda -C Release --output-on-failure` passed 24/24
  in 290.50 s, including both opt-in real Qwen3.6 integration tests.
- `cmake -S . -B build -A x64 -DGGML_CUDA=OFF` and the Release build passed.
  `ctest --test-dir build -C Release --output-on-failure` passed all 17 CPU
  tests; five GPU-only cases skipped as configured, total 128.31 s.
- clang-format 18 dry-run passed for changed project C/C++ files.
- `git diff --check` and the pinned submodule's diff check passed.
- No commit or push was made.
