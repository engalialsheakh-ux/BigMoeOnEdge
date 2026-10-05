# CUDA wide-prefill investigation - 2026-10-04

Working branch: `feature/cuda-wide-prefill`. Frozen parent: `29dda6bba2927c0cb8e0b7d5f305ff28454a0343`.
Local llama.cpp fork branch: `bmoe/cuda-wide-prefill`, commit
`eac6c3a20f896c436e48d4a560ff35e6101bce9c`. The parent milestone pins
this commit as a submodule; neither repository has been pushed or merged.
No model, prompt, quantization, routing, or decode policy changes are intended.

## Hypothesis

The physical 64-token prefill graph repeats expert-layer work. Increasing `n_ubatch`
could amortize expert movement without changing token order or exact routing. The
existing CUDA Q4 KV direct FlashAttention path currently admits at most 64 queries,
so that dispatch and graph reservation are feasibility gates.

## Measured reference (existing logs, no new inference)

The frozen 65,507-token result is approximately 4,153.5 s prefill, 15.77 tokens/s,
3,334,981 expert rereads, and 9.389 decode tokens/s. It used a 4,400 MiB RAM expert
cache, 1,500 MiB GPU expert pool, 128 MiB dedicated VRAM safety, Q4_0 CUDA K/V, and
64-token ubatches. See [the baseline freeze](../2026-10-04-baseline-freeze/README.md).

The existing 3,401-token short trace has 54 prefill graph chunks and 40 expert layers.
Its measured prefill time is 196.658 s, 17.29 tokens/s, 149,905 expert rereads,
269,097.7 MiB physical expert reads, and 97.59 s demand stall. The baseline pool
allocation diagnostic reported 6,143.5 MiB dedicated VRAM total, 1,656.0 MiB free,
1,500 MiB requested, 128 MiB safety, and 156.0 MiB remaining after allocation.
The margin *above* safety was only 28.0 MiB at that allocation point.

## Simulated width replay

`scripts/prefill-width-replay.py` groups consecutive chunks of the recorded exact
routing trace, replays the current RAM cache, and estimates coalesced read windows.
It verifies the width-64 replay against every recorded per-layer observation. Wider
rows are **simulated**: a real wider graph could have different timing and memory
behavior. The table is neither a measured speedup nor a new model run.

Reproduce locally from the repository root:

```powershell
py scripts/prefill-width-replay.py build-cuda/expert-endgame-probe/baseline-repeat.access --widths 64 128 256 512 1024 2048
```

The access trace, the real-model stdout/stderr,
and the temporary operator test logs remain under ignored `build-cuda/`; the
frozen model and prompt hashes are recorded by the baseline freeze.

| Physical width | Graph chunks | Mean distinct experts/layer | Replay rereads | Replay physical MiB | Reread reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 54 | 127.7 | 149,905 | 269,097.7 | 0% |
| 128 | 27 | 150.6 | 93,890 | 174,035.7 | 37.37% |
| 256 | 14 | 164.2 | 53,225 | 105,084.0 | 64.49% |
| 512 | 7 | 179.4 | 27,565 | 61,590.6 | 81.61% |
| 1024 | 4 | 189.1 | 14,495 | 39,448.4 | 90.33% |
| 2048 | 2 | 205.4 | 5,294 | 23,863.8 | 96.47% |

## Static memory and dispatch estimates

This model has hidden width 2,048, 40 layers, 256 experts/layer, top-8 routing,
and a 432 MiB quantized expert bank per layer. Two whole-layer banks would occupy
864 MiB before staging metadata. A single F32 hidden activation plane at widths
128/256/512/1024/2048 is 1/2/4/8/16 MiB, respectively; this is one plane, **not**
the total graph reservation. An F16 262,144-key attention mask at the same widths
is 64/128/256/512/1,024 MiB, respectively. Its actual backend placement must be
verified rather than charged to VRAM by assumption.

In pinned llama.cpp, `ggml_cuda_fattn_mma_use_quantized` refuses `Q->ne[1] > 64`.
The fallback can allocate F16 KV scratch at this long context. No candidate above
64 is approved until direct Q4 attention eligibility, CUDA graph reservation, and
the 128 MiB VRAM rail are verified. Host RAM/page commitment must also be checked.

## Decision log

### Measured reservation and one short integration run

With direct quantized FlashAttention extended past the 64-query dispatch cap,
width 128 reserved 140.63 MiB CUDA compute and 80.39 MiB CUDA_Host compute,
versus 70.32 and 40.33 MiB at width 64. Q4_0 CUDA K/V allocation remained
1,440 MiB. The 1,500 MiB GPU pool **failed the rail**: 1,586 MiB free before
pool allocation would leave 86 MiB, below 128 MiB. A 1,400 MiB pool allocated
successfully and left approximately 186 MiB. At width 128, observed total
dedicated VRAM was 6,143.5 MiB; the model buffer was 1,910.32 MiB, the
quantized KV buffer 1,440 MiB, and the recurrent state buffer 62.81 MiB.
Free dedicated VRAM before the pool was 1,586.0 MiB. Other allocations and
Windows driver ownership account for the remainder; the named buffers must
not be subtracted again from the reported free figure. This was a session-open probe
without a prompt, followed by one real-model short run at width 128 and pool
1,400 MiB. The latter used the existing 3,401-token request and generated 32
tokens. No new 64K run was made.

| Quantity | Existing width 64, pool 1,500 | Measured width 128, pool 1,400 |
| --- | ---: | ---: |
| Prompt tokens | 3,401 | 3,401 |
| Prefill | 196.658 s | 135.979 s |
| Prefill speed | 17.29 tokens/s | 25.01 tokens/s |
| Expert rereads | 149,905 | 93,531 |
| Prefill read | 269,097.7 MiB | 173,348.1 MiB |
| Prefill I/O cumulative | 366.025 s | 232.939 s |
| Prefill stall | 97.590 s | 57.815 s |
| Decode | 6.728 tokens/s | 5.796 tokens/s |
| Major faults per decode token | 0 | 0 |

The candidate has 30.86% less prefill time and 37.61% fewer rereads on this
short run. Direct Q4 attention handled 128 queries without F16 scratch. The
generated text **did not match** the recorded width-64 text, however. Two
recorded width-64 runs, and two further width-64 experiments with other I/O
paths, had identical generated text. The candidate also had 13.9% lower short
decode rate; the 100 MiB smaller GPU pool, differing generated route, and
run-to-run operating conditions cannot be separated from this one sample.
Neither failure is acceptable evidence for preserving the frozen decode result.
The widened prefill is therefore **rejected for production** under this Goal's
output-identity and decode-preservation gates, despite the I/O gain.

### Focused attention correctness

The existing llama.cpp backend-ops comparison was run on CUDA for the D=256,
quantized K/V GQA case at 65, 128, and 256 queries. Q4_0 and Q8_0 cases at
128 and 256 all passed CPU reference comparison (4/4); telemetry confirmed
the direct MMA path, no legacy path, and no F16 scratch. The 65-query Q4_0
case also passed. This rules out an obvious wide-direct-attention error at
the operator's tolerance. It does **not** prove byte-identical logits in the
whole hybrid model when graph shape changes.

### Estimated maximum pool budget at wider graphs

Using the measured 70.31 MiB extra CUDA graph reservation per doubling and
the existing 1,656 MiB free-before-pool measurement, these are **estimates**
of maximum pool sizes that retain 128 MiB safety, not allocation probes:

| Width | Estimated CUDA compute MiB | Estimated largest pool MiB |
| ---: | ---: | ---: |
| 128 | 140.6 | 1,457 |
| 256 | 281.3 | 1,317 |
| 512 | 562.5 | 1,035 |
| 1,024 | 1,125 | 472 |
| 2,048 | 2,250 | none |

At 256 and beyond the decode GPU pool would shrink substantially, in addition
to changing whole-model floating-point execution order. They were eliminated
without further model runs. Width 128 is the largest candidate actually
reserved and run; width 64 is the largest verified output-identical width.

### Existing CUDA layer arena feasibility

The current `DeviceExpertArena` allocates two independent device buffers and
reads expert slices directly from GGUF on *every* graph. It does not consume
the RAM expert cache or share the persistent GPU expert pool. At width 64,
the trace contains 127.7 distinct experts per layer and 54 graphs. Routed
staging would require approximately 465,000 MiB of expert uploads/read bytes
before alignment and prediction misses, versus 269,098 MiB of existing
physical expert reads. Whole-layer staging would read
`432 MiB x 40 layers x 54 graphs = 933,120 MiB`. Each option also needs
two 432 MiB CUDA slots (864 MiB) inside the same 1,500 MiB budget, while the
decode GPU pool already owns all 1,500 MiB. The routed path would additionally
issue roughly 2.5 times the current read-window count if each projection were
read separately. These are **static estimates** from the actual trace and
arena code, not measured runtime results. The existing arena cannot simply be
enabled alongside dense CUDA placement: session validation disallows it, and
its `PrefillDevice::init` expects host-resident dense weights. Sharing one
allocation would require pool/arena ownership and graph-lifetime changes.
The traffic penalty makes this a poor candidate for a speculative production
rewrite under the current evidence gate. The existing short run spent
196.658 s in prefill, of which 97.590 s was demand stall and 29.909 s was
cache management, leaving at most 69.159 s for every other kind of work that
moving expert matmuls to CUDA could remove. Using the measured cumulative
366.025 s to read 269,097.7 MiB, the extra approximately 196,000 MiB of
direct routed staging represents roughly 267 s of cumulative reader time
at the observed rate. Even with four perfectly balanced readers, that is
about 67 s before upload and synchronization. This **estimated** lower-bound
comparison is too close to the 69 s ideal compute-saving ceiling to justify
the extra buffer-ownership rewrite as the next implementation.

### Layer-major feasibility

The old idealized layer-major replay suggests a large residual I/O opportunity
versus width 64. A replay of exact layer-major ordering *within 512-token
tiles* predicts 76.87% fewer physical read bytes and 74.14% fewer read
windows while keeping 64-token attention graphs. This is **SIMULATED** I/O,
not a runtime prediction. The bounded transaction is now implemented and
tested; see [the tile design](layer-tile-design.md). It keeps one 4 MiB host
F32 activation plane, reuses small per-microbatch KV/recurrent descriptors,
and executes 64-token graphs across each layer. Errors after device work
starts invalidate the session; no tensor rollback is attempted.

### Outcome

The opt-in layer-tiled path passed the tiny hybrid state gate, one short
real-model output-identity run, and one final 64K run. It has not replaced the
default width-64 path. The frozen width-64 configuration remains the
long-context comparison reference.
The CUDA Release build and focused tiled/ordinary gates passed; the full
BigMoeOnEdge CTest passed 32/32 after short-run acceptance. Both parent and
fork `git diff --check` passed. The project C++ sources pass clang-format 18;
the pinned upstream llama.cpp tree has a different formatting baseline, so
its changed code is reviewed as a focused fork diff.

### Measured short layer-tiled run

One AC-powered Qwen3.6 run used the frozen 3,401-token request, 32 generated
tokens, full CUDA placement, 64-token physical ubatches, 512-token logical tiles,
4,400 MiB RAM and 1,500 MiB GPU expert caches, 128 MiB GPU safety, and Q4_0
CUDA K/V. The generated text exactly matched the accepted width-64 reference.
These are **MEASURED** single-run values, not medians:

| Quantity | Width-64 reference | Layer-tiled | Change |
| --- | ---: | ---: | ---: |
| Prefill elapsed | 196.658 s | 87.256 s | -55.63% |
| Prefill rate | 17.29 tokens/s | 38.98 tokens/s | +125.4% |
| Expert rereads | 149,905 | 27,877 | -81.40% |
| Prefill expert read | 269,097.7 MiB | 62,248.0 MiB | -76.87% |
| Prefill cumulative I/O | 366.025 s | 94.334 s | -74.23% |
| Prefill demand stall | 97.590 s | 17.027 s | -82.55% |
| Prefill cache management | 29.909 s | 6.382 s | -78.66% |
| Decode rate | 6.728 tokens/s | 6.584 tokens/s | -2.14% |

Physical demand reads were 66,176.7 MiB across 93,291 windows, versus
272,445.8 MiB across 340,062 windows in the reference. The direct quantized
FlashAttention path was used for 280 calls, with zero legacy calls and zero
F16 scratch; CUDA compute reservation remained about 70.3 MiB and Q4_0 K/V
allocation 1,440 MiB. The executor reported 7 committed tiles, 54 physical
microbatches, 54 state preparations, 2,160 layer graphs, and a 4,194,304-byte
host activation plane. The measured read reduction closely matches the replay;
decode is within the variation of a single short sample, not a new throughput
claim. The primary remaining cost on this short run is expert I/O and compute
outside the measured 17.027 s demand stall.

**ESTIMATED:** At 64K, 128 complete 512-token tiles each loading all 40
432 MiB expert banks would read about 2,211,840 MiB. Actual routing and cache
reuse can change that substantially. **THEORETICAL:** The host activation plane
still consumes 4 MiB because it is reused after each tile; that bound does not
include graph or model memory.

### Measured final 64K run

One final run used the exact frozen prompt (SHA-256
`4fb37b40ed6e03e794b9249afe78d628f4d1b3dcfcdf2be2604f66f4fc51702f`),
65,507 prompt tokens, 64 generated tokens, and the frozen CUDA/cache settings
plus `--prefill-layer-tile 512`. It exited successfully on AC power. The
baseline values below are the earlier frozen single measurement; its prefill
time and rate were recorded approximately. This is **MEASURED**, with only
one long tiled run:

| Quantity | Frozen width-64 baseline | Layer-tiled 512 | Change |
| --- | ---: | ---: | ---: |
| Prefill elapsed | ~4,153.5 s | 1,615.059 s | ~-61.12% |
| Prefill rate | ~15.77 tokens/s | 40.56 tokens/s | ~+157.2% |
| Expert rereads | 3,334,981 | 664,841 | -80.06% |
| Decode rate | 9.389 tokens/s | 9.032 tokens/s | -3.80% |

Tiled prefill reported 1,144,418.2 MiB expert reads, 1,687.451 s cumulative
reader time across its lanes, 305.515 s demand stall, and 125.907 s cache
management. The coalescer reported 1,149,378.3 MiB physical reads across
1,568,115 windows, with 464,796 merged ranges and 5,865.9 MiB alignment
over-read. The baseline freeze did not retain corresponding long-run byte and
window counters, so no long-run percentage is claimed for those metrics.

The executor committed 128 tiles and prepared 1,024 microbatches exactly once
each, executing 40,960 layer graphs. There were 5,572,245 expert-cache
lookups and 4,897,528 hits: averaged over the 128 tiles, 43,533 lookups and
38,262 hits per tile (87.9% hit rate). This is an aggregate tile-average cache
reuse measure; it does not isolate within-tile reuse from cross-tile retention.
The activation plane stayed 4,194,304 bytes (4 MiB) on host RAM. CUDA K/V
stayed 1,440 MiB and CUDA compute reservation 70.32 MiB. Quantized
FlashAttention recorded 5,170 direct MMA calls, zero legacy calls, zero F16
K/V calls and zero F16 scratch bytes; ten vector calls were also recorded.
Sampled private process memory was about 10.8 GiB and sampled device memory
use about 5.1 GiB; these are snapshots, not high-water marks.

The short generated text matched the frozen width-64 reference exactly. The
earlier 64K run's generated text was not retained in the freeze, so long-run
text identity cannot be asserted from these records. The new run began the
requested numbered continuation. The 3.80% decode difference is from one
long run per configuration; it is a measurement, not evidence of a structural
decode-path change. Demand stall still accounts for about 19% of tiled prefill
wall time, while most elapsed time lies outside the measured stall and cache
management timers. A detailed compute attribution would need a separate
profile and was not part of this single-run validation.

## Adaptive tile follow-up (2026-10-05)

The exact 3,401-token access trace was replayed at the same 4,400 MiB expert
cache budget, 64-token physical width and unchanged expert IDs. The repeated
ID fraction counts expert selections repeated within a logical tile; it is a
reuse opportunity, not an observed cache-hit rate. This replay predicts I/O,
not wall time or the 64K route sequence.

| Tile | Microbatches/tile | Tiles/64K | Host F32 plane | Repeated IDs | Replay rereads | Expert read MiB | Physical read MiB | Physical drop vs 512 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | 128 | 4 MiB | 81.79% | 27,877 | 61,922.8 | 62,248.0 | 0% |
| 1024 | 16 | 64 | 8 MiB | 89.03% | 14,658 | 39,615.8 | 39,825.0 | 36.02% |
| 2048 | 32 | 32 | 16 MiB | 94.04% | 5,357 | 23,920.3 | 24,047.0 | 61.37% |
| 4096 | 64 | 16 | 32 MiB | 96.80% | 0 | 14,880.4 | 14,957.6 | 75.97% |
| 8192 | 128 | 8 | 64 MiB | 96.80% | 0 | 14,880.4 | 14,957.6 | 75.97% |

The 3,401-token trace fits in one 4096 tile, so 8192 adds no benefit on that
trace. A cyclic extension to 16K tokens, explicitly a simulation rather than
new model routing, still reduced physical reads from 26,108.7 MiB at 8192
to 14,957.6 MiB at 16384. The implemented cap stays at 8192: the 64 MiB plane
is small on this 16 GiB host, while the real 64K gain is already near the
compute floor. The hybrid transaction continues to prepare each microbatch
once, records only small KV/recurrent descriptors per microbatch, and fails
closed after any execution error. Manual 512 remains available and the default
remains off. Auto grows toward the smallest power-of-two tile covering the
prompt, capped at 8192 and session batch capacity, subject to one eighth of
available host memory for the plane; unknown memory selects 512.
The descriptor count is one per 64-token microbatch, independent of tile size:
1,024 for the 65,507-token request. No full-prompt activation or recurrent
tensor snapshot is allocated. The F32 plane is the tile-dependent temporary
memory term. KV/recurrent descriptors are reused across all 40 layer graphs
for each microbatch, so increasing tile size does not increase their count.

An AC-powered same-binary 128-token A/B used the same 3,401-token request and
the frozen CUDA/cache configuration. Auto selected 4096 (32 MiB plane):

| Short run | Tile 512 | Auto 4096 |
| --- | ---: | ---: |
| Prefill seconds | 87.078 | 65.412 |
| Prefill tokens/s | 39.06 | 51.99 |
| Expert rereads | 27,877 | 0 |
| Prefill read MiB | 62,248.0 | 14,957.6 |
| Decode tokens/s | 7.470 | 7.618 |

Both runs produced exactly the same 128-token text. The 24.88% prefill-time
improvement clears the implementation gate. The 1.98% decode-rate difference
is within short-run variation and is not a decode optimization claim.

One final 64K run used the exact prior 65,507-token prompt and 64-token
request. The prompt, generation length and output text were identical to the
512-tile run. The invocation used the arguments in
[final-config.txt](../2026-10-04-baseline-freeze/final-config.txt), the
tracked [prompt recipe](../2026-10-04-baseline-freeze/prompt-recipe.ps1),
`GGML_CUDA_FATTN_MMA_QUANT=auto`, and `--prefill-layer-tile auto` in place of
the previous `512`; the CLI was run in session mode from PowerShell with
`n_predict=64`, `think=false`, `clear_kv=true`. Auto selected 8192, eight
tiles, and a 64 MiB host activation plane.

The command below reproduces the final run from the repository root after
setting `$model` to the existing GGUF path. The tracked prompt recipe emits
the verified 300,104-byte prompt, SHA-256
`4fb37b40ed6e03e794b9249afe78d628f4d1b3dcfcdf2be2604f66f4fc51702f`.

```powershell
$model = 'MODEL.gguf'
$promptFile = 'build-cuda/adaptive-prompt.txt'
powershell -NoProfile -ExecutionPolicy Bypass -File docs/bench-data/2026-10-04-baseline-freeze/prompt-recipe.ps1 -OutputPath $promptFile
$prompt = [IO.File]::ReadAllText((Resolve-Path $promptFile))
$request = @{ cmd='generate'; id=1; prompt=$prompt; n_predict=64; think=$false; clear_kv=$true } | ConvertTo-Json -Compress
$env:GGML_CUDA_FATTN_MMA_QUANT = 'auto'
$request | & build-cuda/cli/Release/bmoe-cli.exe --session --model $model --n-predict 64 `
    --moe-stream --ctx-size 262144 --ubatch 64 --threads 6 `
    --cache-mb 4400 --cache-ceil-mb 4400 --io-threads 4 --overlap --io-coalesce `
    --dense-weights anon --release-mmap --require-release-mmap --n-gpu-layers -1 `
    --gpu-expert-cache-mb 1500 --gpu-expert-window 4 --gpu-safety-mb 128 `
    --cpu-affinity-mask 0xFFF --flash-attn auto --cache-type-k q4_0 --cache-type-v q4_0 `
    --cuda-fa-telemetry --progress --prefill-layer-tile auto
```

| 64K quantity | Tile 512 | Auto 8192 | Change |
| --- | ---: | ---: | ---: |
| Prefill seconds | 1,615.059 | 1,180.404 | -26.91% |
| Prefill tokens/s | 40.56 | 55.50 | +36.83% |
| Expert rereads | 664,841 | 44,620 | -93.29% |
| Prefill expert read MiB | 1,144,418.2 | 92,456.7 | -91.92% |
| Demand stall seconds | 305.515 | 26.072 | -91.47% |
| Cache management seconds | 125.907 | 12.570 | -90.02% |
| Decode tokens/s | 9.032 | 8.456 | -6.38% |

The final run committed eight tiles and 1,024 microbatches with 40,960 layer
graphs. It made 5,572,245 expert-cache lookups and 5,517,749 hits (99.0%).
The coalescer reported 97,437.3 MiB physical reads across 140,769 windows
for prefill plus decode. Sampled process private memory was 10.89 GiB; the
fixed CUDA allocations included 1,440 MiB Q4_0 KV, 70.32 MiB compute and a
1,500 MiB GPU expert pool. The allocation precheck reported 6,143.5 MiB total
and 1,656.0 MiB free dedicated VRAM before the pool, leaving 156 MiB after
its request. These are allocation figures and process snapshots, not peak
VRAM telemetry.

The long-run decode rate is 6.38% below the earlier single 512 run, but the
one-token decode code and 64-token generated text are identical, as are 8,498
GPU expert hits, 11,982 CPU misses and 1,245 promotions. The controlled short
A/B showed no decode slowdown. The long-run difference is reported as measured;
its cause was not isolated. For this 64-token round, prefill plus decode fell
from about 1,622.1 to 1,188.0 seconds (26.8%), so the overall latency gain outweighs
the observed decode difference. The remaining prefill demand stall and cache
management total only 38.642 seconds (3.27% of prefill wall); graph execution,
compute and synchronization form the unseparated dominant remainder. Decode
reported 72.2 ms/token compute, 36.4 ms/token demand stall and 9.7 ms/token
cache management. The asynchronous GPU upload worker transferred 2,100.94 MiB
in 543.718 ms cumulative time; GPU synchronization and graph/setup time were
not isolated by these counters. The earlier eight-token
[Nsight diagnostic](../2026-10-03-exact-endgame/README.md) saw short CUDA graph
intervals of roughly 0.24-0.56 ms/token, but could not assign a steady 64K
synchronization or graph setup cost.
Existing GPU residency is per-layer. Prior held-out allocation and
route-predictor replays did not show a meaningful decode miss reduction, so no
speculative decode-cache rewrite was retained.

CUDA Release built and focused MoE, hybrid, configuration and tile gates
passed 12/12. CPU Release built; its configuration and direct tile gates
passed 3/3 and the qwen3moe MoE gate passed. The tile gate checked 1024 and
8192 arguments, KV/recurrent metadata, post-tile decode and two-turn ChatML
continuation with matching output and state position. Both parent and
fork `git diff --check` and clang-format 18 on changed project C++ files
passed.
