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

CUDA overlap requires an expert cache. The zero-cache overlap path uses shared host slots across layers and failed the resident-output comparison on a small hybrid graph; the CLI rejects that combination. Zero-cache serial streaming remains available. CUDA mode currently rejects row streaming, route tracing, prefetch and routing policies, speculation, and decide operations whose callbacks would need separate device-access work.

Partial positive `--n-gpu-layers` placement has a separate mapping limitation. A diagnostic 10-layer run succeeded with the mapping intact but crashed after `--release-mmap`, even though the capture graph's inspected pointers looked safe. The engine now declines mapping release for every positive partial placement, and strict release fails before generation. Use `-1` for a full eligible CUDA placement when mapping release is required.

To register the real-model correctness test, set `BMOE_QWEN36_MODEL` to the local GGUF path before configuring `build-cuda`, then run `ctest --test-dir build-cuda -C Release -R hybrid_gpu_qwen36 --output-on-failure`. This test is absent from normal CI and compares CUDA resident, serial streamed and overlapped streamed outputs with a 3000 MiB expert cache and GPU KV placement. The gate logs the cache size and KV placement used by its serial and overlap runs.

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
