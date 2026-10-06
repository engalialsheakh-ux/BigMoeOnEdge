# Deterministic GPU experts with a persistent agent checkpoint

This experiment keeps the existing 262,144-token context, Q4_0 K/V, 4,400 MiB
RAM expert cache, 1,500 MiB GPU expert pool, 128 MiB GPU safety reserve,
native top-k routing, and adaptive prefill. The model and benchmark prompt are
the same as in the [agent endgame baseline](../2026-10-06-agent-endgame/README.md).
No model or other asset was downloaded. The GPU test device is described only
by its 6 GiB dedicated VRAM capacity.

## Cause and design

The old persistent-state guard required `--gpu-expert-no-promote` because two
restores from the same sequence state could diverge with active promotions.
The earliest observed difference was the GPU execution decision, before the
routed expert IDs differed. A 64-token route trace of two restores showed the
second request using a previously populated expert slot on its first decode.
Clearing only VRAM residency moved the first difference to the third decode:
the first request needed one token to discover the graph's GPU expert branch,
while the second retained that discovery across requests and began promotion
one token earlier. Different CPU/GPU placement can alter floating-point
rounding and then subsequent router choices; arithmetic alone was not the
starting cause.

An independent request now drains pending uploads, clears logical VRAM
residency without reallocating the buffer, resets repeated-miss history, and
rediscovers the first graph's cache branch. Uploads remain in a background
worker between tokens. The next decode waits for that token's prior promotions
to finish, so CPU/GPU selection sees a stable pool throughout the graph.
Duplicate misses inside one decode do not count as cross-token reuse. The
checkpoint still contains only sequence state and tokens, not expert weights.
A checkpoint made with promotions disabled remains compatible because GPU
promotion is decode-only and does not affect saved prefill state.

## Focused evidence

With these changes, two 64-token restores from the same 17,408-token snapshot
produced identical decoded text. Their 20,480 decode route rows per request
(64 tokens × 40 layers × 8 experts) had identical expert IDs and GPU execution
assignments. Five prefill RAM-residency cells differed because the second
request had warmer host cache; GPU routing did not differ. A separate untraced
128-token pair also produced identical text.

| Short 128-token run | First restored request | Second restored request |
| --- | ---: | ---: |
| Dynamic GPU promotion, tokens/s | 8.264 | 8.859 |
| Earlier no-promotion control, tokens/s | 8.308 | 8.513 |

The short rates are diagnostic, not a claimed long-run speedup: the GPU path
can produce different text from the no-promotion control, and a 128-token run
still pays for cache warm-up. Across the two dynamic requests, teardown counted
39,134 GPU expert hits, 42,786 CPU misses, 5,850 promotions, 4,154 evictions,
and 848 resident GPU slots. This establishes useful cache execution, rather
than an allocated but idle pool.

The new tiny-model `gpu_agent_state_gate` requires equal cold, in-process
restored, and restarted output, equal per-layer routed IDs and CPU/GPU
assignments, shorter restored prefill, and actual GPU hits. The existing pool
unit gate checks that clearing logical residency invalidates old leases while
reusing its device allocation.

## Representative agent comparison

The final cold/cached result below uses the existing 17,663-token prompt and
1,292-token completion. The raw logs and state file remain in the ignored
`build-cuda/determinism-probe-1292-plain` directory and `build-cuda` root.

| Quantity | Cold | Restored |
| --- | ---: | ---: |
| Prompt tokens actually prefilled | 17,663 | 255 |
| Prefix tokens restored | 0 | 17,408 |
| Prefill seconds | 326.375 | 9.391 |
| Sequence-state restore seconds | — | 0.032 |
| First generated-token wall seconds | 0.366 | 0.326 |
| Request first response seconds, excluding model load | 326.741 | 9.717 |
| Generated tokens | 1,292 | 1,292 |
| Decode tokens/s | 9.181 | 9.261 |
| Total request seconds, excluding model load | 467.108 | 148.900 |

Both full texts have UTF-8 SHA-256
`3b190aa395b3a8cfada8292f948e60f87adc7afc43c2fbb9e91d0712de67dc5e`.
The 17,408-token checkpoint is 166,622,224 bytes (158.8 MiB), including its
header and tokens; capture took 0.424 s. Model load took 8.714 s once, before
the two requests. The prior no-promotion restored decode rate was 7.802
tokens/s, so 9.261 is an 18.7% improvement; the restored first response remains
in the same roughly 11-second class. These are one full comparison per mode,
not a distribution of benchmark runs. The GPU-enabled output need not match the
earlier all-CPU-expert output because mixed-device arithmetic can round
differently; the cold and restored outputs **within this mode** matched exactly.

Across both requests, 417,514 expert executions hit the GPU pool and 409,366
used the CPU path (50.49% GPU hit rate). There were 55,482 promotions and
53,786 evictions; all 848 real GPU slots were resident at teardown. Total
barrier wait was 7.353 s over both 1,292-token completions, about 2.85 ms per
generated token. A sampled process private-memory value during the comparison
was about 11.1 GiB; this is not a measured peak. The remaining large cost is
the many CPU expert misses and their partially hidden storage work: the
restored run recorded 0.140 s of expert I/O and 0.038 s of non-overlapped stall
per token. No 64K benchmark or 256K performance run was repeated.
