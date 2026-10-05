# Hybrid CUDA decode on Windows

The host CLI can place dense model work and KV state on a CUDA GPU while BigMoeOnEdge streams routed MoE experts from storage into CPU memory. CUDA is optional: `--n-gpu-layers 0` is the default, including in a CUDA-enabled build. The Android app and its NPU prefill path are unchanged.

## Build and run

Install the NVIDIA CUDA Toolkit and a supported MSVC toolchain, then configure a separate build tree:

```powershell
$env:CUDA_PATH_V13_4 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
py -m pip install gguf numpy
cmake -S . -B build-cuda -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --config Release --parallel 4
ctest --test-dir build-cuda -C Release --output-on-failure
```

The versioned environment variable is needed if the CUDA Visual Studio integration cannot find the Toolkit during CMake's compiler checks. Set it to the installed Toolkit version and path. Architecture `86` is for an Ampere GPU; choose the architecture of the actual device. The build also works without CUDA in the ordinary `build` tree.

Install the Python packages before the first CMake configure when tests are enabled. CMake checks `gguf` and `numpy` in its selected Python interpreter and reports a setup error if either is missing. On Windows, CMake stages llama.cpp, ggml backend, and CUDA runtime DLLs beside the CLI and test executables, so the plain CTest command needs no manual `PATH` change.

For a model with a `qwen35moe` recipe, a lossless streamed run is:

```powershell
build-cuda/cli/Release/bmoe-cli.exe -m model.gguf -p 'The capital of Japan is' -n 256 -c 1024 -t 4 --ubatch 64 --moe-stream --dense-weights anon --release-mmap --require-release-mmap --cache-mb 3000 --cache-ceil-mb 3000 --io-threads 4 --overlap --n-gpu-layers -1
```

`--n-gpu-layers N` also accepts a positive partial layer count. `--main-gpu N` selects a CUDA device; a missing requested device fails explicitly. KV and KQV offload are on when CUDA layers are requested; `--no-kv-offload` keeps them on CPU. `--flash-attn auto|on|off` exposes llama.cpp's attention choice. `--require-release-mmap` turns a declined or incomplete mapping release into an error, so a benchmark cannot silently run with a different mapping state.

The Windows CLI accepts `--cache-type-k f16|q8_0|q4_0` and `--cache-type-v f16|q8_0|q4_0` independently. Both default to `f16` when omitted; they select llama.cpp's existing KV cache storage types without changing model weights. The `bmoe-kv-cache: K=... V=...` startup line reports the types passed to the successfully created context. llama.cpp checks whether the chosen types work with the model and attention mode; in particular, a quantized V cache may require Flash Attention.

For the Qwen3.5 MoE hybrid model, `--prefill-layer-tile N|auto` enables an
experimental layer-major prompt schedule. Manual `N` is a power of two from
512 through 8192 and cannot exceed the context or session batch capacity.
`auto` grows toward the smallest supported tile covering the prompt, capped at
8192, session batch capacity, and one eighth of currently available host RAM
for the F32 activation plane; unknown memory
falls back to 512. A continued session chooses again for its new prompt suffix.
The physical CUDA graphs and quantized FlashAttention remain width 64. It
requires `--ubatch 64 --moe-stream --n-gpu-layers -1`, a context of at least
512 tokens, and no speculation or prefill device. It is off by default and
does not change one-token decode. At hidden width 2,048 the reusable host
activation plane is 4 MiB at 512 and 64 MiB at 8192. Runtime diagnostics
report the selected tile and plane size. If tiled execution fails after state
mutation, the context is invalidated and the session must be reopened. The
[wide-prefill report](bench-data/2026-10-04-cuda-wide-prefill/README.md) records
the replay sweep, exact-output gates and measured 64K comparison.

For quantized KV on pre-Ada CUDA, eligible D=256 Q4_0/Q8_0 GQA batches of 2–64 queries use a CUDA MMA kernel that stages only the current quantized K/V tile and expands it into the MMA tile. Eligible single-query decode uses the same path automatically at 4096 or more KV positions on Ampere; shorter decode uses a vector FlashAttention kernel that shares K/V reads across query heads. Neither path allocates a context-sized F16 K/V copy. The MMA path requires NVIDIA cp.async support, aligned K/V rows, a padded KV length, a mask without ALiBi, no sparse gather, and enough shared memory. Other shapes retain llama.cpp's existing MMA/tile fallback and may convert K/V to F16 scratch. `--cuda-fa-telemetry` reports the actual paths and scratch allocation at exit.

`GGML_CUDA_FATTN_MMA_QUANT=auto|direct|legacy` selects the Q4_0/Q8_0 MMA policy before launch. `auto` is the default; `direct` also selects MMA for eligible single-query decode; `legacy` retains the original vector/MMA selection and full F16 conversion in MMA. Set `GGML_CUDA_FATTN_VEC_NCOLS2=1` to disable vector GQA sharing for a legacy comparison. The synthetic D=256, GQA=6, 16K-KV Q4_0 vector op took 415 microseconds with GQA sharing versus 660 microseconds with its control on the local 6 GiB Ampere GPU; this measures one operator, not model throughput. The existing [65,507-token exact baseline](bench-data/2026-10-04-baseline-freeze/README.md) measured direct quantized CUDA FlashAttention with `auto` and no F16 scratch.

In a separate synthetic D=256, GQA=6, 16K-KV, two-query operator run on AC power, direct MMA took 171 microseconds for Q4_0 and 235 microseconds for Q8_0. The legacy MMA control took 1095 and 1135 microseconds, respectively, and allocated 64 MiB of F16 K/V scratch; direct MMA allocated none. The normal vector GQA path took 714 and 739 microseconds. These are single operator measurements and do not predict the CLI's token rate.

For a synthetic single-query Q4_0 operator on the same Ampere class, vector versus direct MMA took 140/64 microseconds at 4096 KV, 412/196 at 16384 KV, and 1534/700 at 65536 KV. At 4352 KV with 64 queries, direct MMA took 679 microseconds versus 700 for legacy MMA, with zero versus 17 MiB of F16 K/V scratch. These measurements select the Ampere decode threshold and validate the wider prefill path; they are not real-model throughput results.

`--no-kv-offload` can keep FlashAttention off the CUDA backend as well as keep the KV cache in host memory. The quantized hybrid prefill gate with that flag produced zero CUDA FlashAttention calls. Check `bmoe-cuda-fa: calls` in any real-model comparison before attributing a throughput change to this CUDA kernel; a zero-call run measures a different path.

During prefill, the RAM expert cache reserves room for the largest bound layer's full expert bank
(clamped to the cache budget) and divides the remaining bytes across MoE layers by their discovered
expert-bank sizes. Each layer retains its most frequent routed experts from the current batch;
later occurrence breaks frequency ties. Other demanded experts are still read and used, then may
be evicted after their layer finishes. On decode the protection ends and the usual global LRU uses
the full configured cache. `BMOE_DONE` reports retained hits and prefill cache occupancy alongside
`prefill_read_mib` for comparing prompt runs.

CUDA overlap requires an expert cache. The zero-cache overlap path uses shared host slots across layers and failed the resident-output comparison on a small hybrid graph; the CLI rejects that combination. Zero-cache serial streaming remains available. CUDA mode currently rejects row streaming, prefetch and routing policies, speculation, and decide operations whose callbacks would need separate device-access work.

During multi-token prefill, streamed expert matmuls stay on CPU even when llama.cpp would otherwise offload a wide host-weight operation to CUDA. The expert cache commits only routed slices of each host tensor; llama.cpp's selected-expert copy includes padding past a selected group, which can read an uncommitted neighboring page. GPU-backed dense tensors still execute on CUDA, and prefill remains batched. The synthetic `hybrid_gpu_batched_prefill` gate covers the wide CUDA schedule with a 256-expert/top-8 fixture, quantized CPU KV, and serial/overlapped streaming.

Partial positive `--n-gpu-layers` placement has a separate mapping limitation. A diagnostic 10-layer run succeeded with the mapping intact but crashed after `--release-mmap`, even though the capture graph's inspected pointers looked safe. The engine now declines mapping release for every positive partial placement, and strict release fails before generation. Use `-1` for a full eligible CUDA placement when mapping release is required.

To register the real-model correctness test, set `BMOE_QWEN36_MODEL` to the local GGUF path before configuring `build-cuda`, then run `ctest --test-dir build-cuda -C Release -R hybrid_gpu_qwen36 --output-on-failure`. This test is absent from normal CI and compares CUDA resident, serial streamed and overlapped streamed outputs with a 3000 MiB expert cache and GPU KV placement. The gate logs the cache size and KV placement used by its serial and overlap runs.

## Experimental VRAM expert cache

Add `--gpu-expert-cache-mb 2048` to the full CUDA command above to reserve a bounded pool for routed experts. Zero keeps the established CUDA dense/KV path. The pool requires `--moe-stream` and `--n-gpu-layers -1`; the existing CUDA restrictions on routing policies and speculation still apply. One-token decode uses the exact top-k IDs: a READY VRAM hit runs gate/up/down on CUDA, and a miss runs the existing RAM/SSD-backed CPU expert path. The two rows are added before the model's original router weighting. Prefill keeps the existing path. The engine copies only already-resident RAM bundles into a bounded staging queue after decode; a background uploader publishes complete bundles for later tokens. It never reads the SSD only to warm VRAM.

`--gpu-safety-mb N` sets the dedicated VRAM that must remain free after the GPU expert-cache allocation. The default is 768 MiB; the minimum accepted value is 128 MiB. The allocation diagnostic reports total and currently free dedicated VRAM, the requested expert-cache MiB, the configured safety MiB, and the actual remaining MiB after allocation. Failed prechecks report the projected remaining MiB instead.

`--expert-access-trace PATH` records exact host-visible expert demand batches in a compact binary file for `py scripts/expert-access-replay.py PATH --budget-mb 4400`. It is off by default and captures the IDs at the existing router boundary without adding a CUDA barrier. The replay checks every batch's lookup, hit, reread, eviction, and logical read-byte counts against the live cache before comparing policies. The trace is diagnostic I/O and its run is not a speed benchmark. The file format is versioned and includes each layer's projection offsets and slice sizes.

`--io-coalesce` is an opt-in exact demand-I/O mode for overlapped streaming. Within a projection's ascending expert IDs, adjacent misses whose source and destination slices are both contiguous become one read of at most 4 MiB of requested expert bytes. It never merges across a file, layer, projection, gap, or the two-wave publication boundary. The existing `FileReader` aligns that one read for Windows unbuffered I/O and copies the exact requested span into the already committed cache slices; completion publishes every included expert's ready flag. The default keeps the existing one-slice reads. At teardown `bmoe-io-coalesce` reports logical and physical MiB, read windows, merged ranges, and alignment over-read MiB. These counters count demand expert reads; speculative reads are excluded.

`--gpu-expert-no-promote` keeps the pool empty for an all-miss correctness control. The `bmoe-gpu-cache` diagnostic prints `gpu_expert_hits`, `cpu_expert_misses`, `gpu_promotions`, `gpu_evictions`, `gpu_cache_resident`, `gpu_upload_mib`, and `gpu_upload_ms` at session teardown. A requested cache on an unsupported one-token MoE graph logs that the CPU path remains active. The focused `gpu_expert_mixed` test covers all-miss, mixed, all-hit, and changed mappings with four CPU threads. This experimental path uses the local pinned-fork delta described in [seam.md](seam.md).

On Windows, `--cpu-affinity-mask 0xFFF` restricts this CLI process to the first twelve logical CPUs. A local six-thread decode screen improved substantially with this mask on the tested hybrid CPU; confirm the correct P-core logical CPU mask for any other machine. `--gpu-expert-window N` controls the repeated-miss admission window (default 8). `--route-trace PATH` now works with the CUDA expert-cache path and adds a `gpu_hit` column for cache-policy analysis. Trace runs are diagnostic and should not be used as throughput benchmarks. The cache diagnostic also reports `gpu_upload_calls`.

In a same-binary, interleaved 256-token comparison on the Qwen3.6-35B-A3B Q4_K_M integration model, a 2048 MiB GPU expert cache yielded 7.082 median decode tok/s versus 6.921 with the GPU expert cache disabled (+2.33%). The six runs used `A B B A A B`, with all other settings equal: 3000 MiB RAM expert cache, context 1024, ubatch 64, four CPU threads, four I/O lanes, overlap, anonymous dense weights, strict mapping release, full CUDA placement, GPU KV, and greedy decode. The three A rates were 6.909, 6.921, and 6.967 tok/s; the three B rates were 7.082, 7.078, and 7.140 tok/s in run order. Each B run counted 42,604 GPU hits and 39,316 CPU misses (52.01% GPU hit rate, 153.58 CPU misses/token), 37.11 MiB/token uploaded, and 140.35 MiB/token streamed from storage; A streamed 151.62 MiB/token. Sampled peak dedicated GPU process memory was 4306.8 MiB for B and 2256.7 MiB for A, on a 6 GiB GPU; sampled shared usage was 80 MiB in both arms. These are sampled peaks, not allocator high-water marks. See the [experiment log](bench-data/2026-10-01-cuda-expert-cache/goal-log.md) for the probe and correctness details.

A later AC-powered max-speed screen kept the same model and cache budgets. Six CPU threads with `--cpu-affinity-mask 0xFFF --gpu-expert-window 4` yielded 7.088, 7.161 and 7.116 tok/s against 6.988, 7.138 and 7.012 tok/s for the Phase 2 equivalent four-thread/window-8 configuration, in `A B B A A B` order. The medians were 7.116 and 7.012 tok/s (+1.48% same binary). This is only +0.48% versus the earlier 7.082 tok/s Phase 2 median and lies within local run variation; it does not establish a sustained speedup beyond Phase 2. Window 4 did reduce SSD traffic from 140.35 to 134.29 MiB/token and H2D traffic from 37.11 to 32.01 MiB/token. The [max-speed ledger](bench-data/2026-10-02-max-speed/ledger.md) records the controls, memory samples and rejected candidates.

Separately, the existing opt-in `--n-expert-used 6` setting measured 10.391, 10.746 and 10.628 tok/s (median 10.628) with the six-thread/window-4 settings. It reduces the model's routed top-k from 8 to 6 and changes its semantics; its quality was not evaluated. This TURBO_MAX figure must not be compared as an exact-model improvement.

## Placement and correctness

The model loader keeps `load_mode=MMAP` and `use_extra_bufts=false`. llama.cpp's CPU tensor override selects routed expert weights by its architecture-wide MoE pattern; `no_host=true` prevents a CUDA pinned-host allocation of the complete expert set. Captured GPU-owned dense tensors are excluded from host anonymous relocation. The `ffn_moe_topk` result can live on CUDA, so the router copies only its small ID tensor through the public ggml backend API before staging experts. An expert-ready hook in the pinned CPU `mul_mat_id` kernel still waits for streamed slices under a mixed CUDA/CPU graph.

On the local Qwen3.6-35B-A3B Q4_K_M integration model, sampled expert tensors were `CPU_Mapped` and file-backed after load. The CUDA model buffer was 1910 MiB, while the CPU mapping represented about 19 GiB of expert-heavy weights. Private process commit immediately after load was about 2.48 GiB in the hybrid probe versus 0.39 GiB in the CPU probe using the same binary; this increase is consistent with GPU model allocation, not a full expert copy. One diagnostic decode observed 2964 CUDA output buffers, 480 CPU output buffers, and 343 CUDA host output buffers; all 120 routed `MUL_MAT_ID` nodes had CPU outputs, while 30 gated-delta-net and 30 SSM-convolution nodes had CUDA outputs. The CPU expert-ready hook fired 3840 times in that decode. The small resident/streamed/overlap gate passed with a cache, and the real model's resident output matched both serial and overlap streamed output with the same CUDA placement, with KV on CPU and with KV on GPU.

## Measured decode

These are medians of three 256-token greedy runs on a Windows host with 16 GiB RAM, NVMe storage, and a 6 GiB Ampere CUDA GPU. All runs used the same Qwen3.6-35B-A3B Q4_K_M file, prompt, `--overlap`, 3000 MiB expert cache, context 1024, ubatch 64, four CPU threads, four I/O lanes, anonymous host dense weights, and a confirmed mapping release. Each result is from the CUDA-enabled binary; the CPU row requested zero GPU layers. GPU and CPU floating-point paths can choose different later tokens, so speed comparisons are indicative rather than teacher-forced identical-trajectory measurements.

| Placement | Decode tok/s | Seconds/token | Prefill (5 tokens) | TTFT | Read MiB/token | Cache hit | Overlap stall s/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU, `ngl=0` | 4.928 | 0.203 | 1.387 s | 11.091 s | 125.85 | 75.8% | 0.059 |
| CUDA, KV on CPU | 5.422 | 0.184 | 1.697 s | 11.811 s | 128.35 | 75.4% | 0.053 |
| CUDA, KV on GPU | 7.335 | 0.136 | 1.562 s | 9.091 s | 128.35 | 75.4% | 0.044 |
| CUDA, KV on GPU, Flash Attention off | 6.680 | 0.150 | 1.664 s | 11.255 s | 127.02 | 75.6% | 0.048 |

The three `CUDA, KV on GPU` runs spanned 6.149 to 7.364 tok/s. With KV on CPU, llama.cpp logged 292 graph splits and a 20 MiB CPU KV buffer; with KV on GPU, it logged 122 splits and a 20 MiB CUDA KV buffer. This reduction in splits is the strongest observed explanation for the decode gain. The model, KV and compute allocations reported by llama.cpp were about 1910, 20 and 154 MiB respectively, leaving substantial room within the 6 GiB physical VRAM budget. A sampled GPU reading during CUDA decode showed about 2.2 GiB dedicated memory use and 21–45% utilization. Flash Attention off did not improve the median, and its output trajectory differed, so this test does not establish a precise attention-only speed effect.

With the mapping left intact, exploratory 128-token runs at 20 and 30 CUDA layers reached 3.47 and 3.59 tok/s, with 501 and 323 graph splits. A shorter 10-layer smoke run reached 2.47 tok/s with 683 splits. These runs are not directly comparable to the full-placement table because strict mapping release was unavailable for positive partial placements. Full placement (`-1`) remained within the VRAM budget and was the useful setting on this host.

The CLI logs whether the model mapping was released. A pre-existing Windows teardown warning (`UnmapViewOfFile failed`) also appeared after successful runs, including CPU runs; it did not change the measured release result or process exit status.
