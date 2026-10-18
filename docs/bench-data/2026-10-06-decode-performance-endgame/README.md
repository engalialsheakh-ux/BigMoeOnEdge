# Decode-performance endgame: measured limit of the current path

The starting point is the exact persistent-agent configuration in the
[deterministic GPU-cache report](../2026-10-06-deterministic-gpu-agent-cache/README.md):
262,144 context, Q4_0 K/V, native top-8 routing, 4,400 MiB RAM expert cache,
1,500 MiB GPU expert cache, 128 MiB GPU safety reserve, six affined compute
threads, four direct-I/O lanes, and overlap. The frozen long result restored
17,408 of 17,663 prompt tokens, reached first response in 9.717 s, and decoded
1,292 identical tokens at 9.261 tokens/s. This is a historical, pre-release-mmap
run; the active post-release-mmap expert I/O stall baseline is about 29 ms/token
overall and about 25 ms/token over tokens 129-256. No 64K prefill run or new
17K cold prefill was performed for this investigation.

## Where a token spends time

The historical 1,292-token restored run averaged about 108.0 ms per generated token.
Its pre-release-mmap counters assigned 38.0 ms to non-overlapped expert-read stall,
10.7 ms to expert-cache management, and 59.3 ms to the residual named
`compute_ms`. That last field is **not** a measurement of arithmetic: it also
contains waiting and scheduling. The run streamed 110,867 MiB in total, about
85.8 MiB per token. Major faults were zero. Across the cold and restored
requests, 417,514 selected experts hit GPU and 409,366 ran on CPU; the 848
real GPU slots were populated. The deterministic GPU-promotion barrier cost
7.353 s over both 1,292-token completions, or 2.85 ms per token. Even removing
that wait entirely would save less than 3% of the current token wall time.

A separate 128-token restored run used `--compute-trace-layers`. It measured
121.72 ms/token in the trace and 121.73 ms/token for that run's decoded wall
time. The matched untraced 128-token baseline ran at 8.408 tokens/s versus
8.215 tokens/s with tracing, a 2.3% difference that may include run order and
trace overhead. The expanded `scripts/decode-analyze.py compute` output
partitions its 81 intervals per token as follows:

| Interval | ms/token | Share |
| --- | ---: | ---: |
| Before the first layer | 3.61 | 3.0% |
| First interval of each layer | 22.84 | 18.8% |
| Later intervals of the layers | 81.30 | 66.8% |
| Final layer tail and output head | 13.97 | 11.5% |

In this MoE graph, top-k separates each layer's first and later intervals.
The 81.30 ms group contains expert reads, CPU and GPU expert execution,
activation movement, and synchronization. Its non-overlapped storage waits
were the largest measured subcost in that earlier run (about 40 ms/token in its
matched 128-token untraced screen). The active post-release-mmap baseline is
about 29 ms/token overall and 25 ms/token over tokens 129-256; it is incorrect
to call the whole group CPU compute. A previous
per-node trace found `MUL_MAT_ID` prominent, but its roughly 4,187 barriers per
token inflated runtime to 319 ms/token and cannot establish an untraced
CPU/GPU kernel saving.

## Smallest direct experiment

The existing `--io-two-wave` option publishes first-projection reads while
the remaining projections' pages are committed. It targets the measured
expert-read wait without changing weights, routing, or GPU placement. Both
screens restored the same checkpoint, used temperature zero, and produced
identical text, GPU-hit counts, SSD byte counts, and promotion counts within
each matched pair. The 256-token pair ran in the reverse order from the
128-token pair to expose order-sensitive short-run gains.

| Historical pre-release-mmap completion | Earlier baseline I/O | Two-wave I/O | Result |
| --- | ---: | ---: | --- |
| 128 tokens, decode tokens/s | 8.408 | 8.810 | Two-wave +4.8% |
| 128 tokens, stall ms/token | 40.1 | 37.9 | Two-wave -2.2 ms |
| 256 tokens, decode tokens/s | 9.804 | 9.768 | Two-wave -0.4% |
| 256 tokens, stall ms/token | 33.4 | 33.0 | Two-wave -0.4 ms |

These stalls are from the earlier experiment; the active post-release-mmap
measurement is about 29 ms/token overall and 25 ms/token over tokens 129-256.
The longer confirmation did not reproduce a throughput gain, so no runtime
flag, default, or documentation recipe was changed. The earlier positive
128-token screen is treated as warm-up/order noise. No long agent benchmark
was justified because there is no accepted optimization to validate.

## Other ceilings and decision

Existing offline and local evidence already rejected more direct-I/O lanes
on this SSD, trained nonuniform GPU-slot allocation, LFU admission, and
speculative GPU uploads. The current four-lane reader was near the measured
throughput knee; a paired sidecar reduced read submissions but slowed an
exact real-model diagnostic. Replaying an older 256-token route trace,
captured with a 2,048 MiB GPU cache, under a simulated 1,500 MiB limit gave
a repeat window of three only 96 more held-out GPU hits than window four's
17,680. This offline screen is far below a credible material-gain gate and is
not a live speed measurement.

Prior CUDA-launch analysis placed its optimistic saving at about 3 ms/token,
and the installed CUDA backend already uses the eligible fused expert path.

The historical 9.261 tokens/s, 9.717 s restored first response, exact replay,
and memory budget remain the result of this earlier investigation. It changed
no production inference code or llama.cpp source. A material step from that
result toward 11 tokens/s would require saving roughly 17 ms of its 108 ms token,
most plausibly by starting *exact*
expert reads before their layer stalls without adding speculative flash
traffic. That would need a verified early-router/graph schedule and bounded
staging across the CUDA hybrid boundary; the present CUDA mode deliberately
disallows the existing prediction and prefetch policies. Such work needs a
separate measured design and correctness gate. The single largest remaining
bottleneck is routed-expert execution, particularly its unhidden storage wait.

## Follow-up exactness and capacity audit

An optional expert-residency manifest prototype restored 2,607 RAM experts
(4,399.3 MiB) from an existing Claude prefix snapshot in 2.35 s. In one
128-token restart pair, startup through the completed answer took 38.327 s
without the manifest and 38.065 s with it. The generated text differed.
Two independent 128-token runs without the prototype produced identical text
hashes and identical expert read volume. The prototype was removed: its
roughly 0.7% end-to-end saving does not justify enabling a path that failed
the exact-output screen. The production Claude bridge remains unchanged.

The physical KV portion of this audit is now an opt-in implementation. Source
inspection found that `llama-model.cpp` passed `cparams.n_ctx_seq` as
`attn_kv_size` to `llama_memory_hybrid`; its attention `llama_kv_cache` creates
full-capacity K/V tensors with `ggml_new_tensor_3d` and allocates their backend
storage with `ggml_backend_alloc_ctx_tensors_from_buft`. The local llama.cpp
delta passes an independent physical `n_kv` while retaining the full logical
`n_ctx_seq`. BigMoe saves the hybrid sequence state, recreates only the context
over the already-loaded model, and restores it. The existing `state_write_data`,
`state_read_meta`, and `state_read_data` paths depend on compatible cell
positions and destination capacity; a focused gate checked live-state growth
and agent/Claude prefix restart restore. At logical context 262,144 with Q4_0
K/V, the real model reserved 180 MiB of KV at startup rather than 1,440 MiB.
The tiny hybrid gate measured roughly 3–4 ms for a 512-to-1,024-cell
grow/recreate/restore event. This is not a real-model growth timing. A separate
real-model Claude prefix restart restored 576 tokens, evaluated 10 prompt
tokens versus 586 cold, and produced identical eight-token greedy output.

Dynamic sharing with the GPU expert pool remains blocked by the exact-output
requirement. `GpuExpertPool::init` derives slot counts and offsets from a fixed
VRAM budget and allocates one device buffer. A trial drained uploads, released
the old context and graph, resized the pool, and restored resident experts
before growing KV. At startup it raised the real model's pool from 1,500 MiB
and 848 slots to 2,724 MiB and 1,574 slots. In one matched 128-token screen,
the original and Elastic-KV-only runs produced identical text at 7.527 and
7.668 tokens/s. The larger-pool run reached 9.103 tokens/s but changed a later
greedy token; its storage stall was 38 ms/token versus 46 ms/token in that
short screen. These values do not replace the active post-release-mmap stall
baseline above. The pool-sharing trial was removed; guaranteeing exact output
across changed CPU/GPU expert placements needs more than a budget or allocator
change.

## Rejected exact shadow VRAM rescue

A separate trial kept the active GPU expert pool fixed at 1,500 MiB and 848
slots. It stored original compressed expert bundles in an inclusive shadow
pool: restoring a bundle to RAM left its shadow copy valid. Four pinned host
staging slots and a dedicated CUDA stream accepted best-effort asynchronous
insertions; a busy ring dropped an insertion without waiting for H2D. Restores
copied shadow VRAM to pinned memory and then to the existing CPU expert slot.
The trial did not change routing, expert arithmetic, or Elastic KV.

A microbenchmark using one real 1.6875 MiB expert bundle measured 0.100 ms
host-visible / 0.201 ms device time for pageable H2D. Pinned staging took
0.077 ms of CPU memcpy, 0.020 ms to enqueue, and 0.147 ms of H2D device time.
The dedicated-stream ring's accepted insertions were host-visible for about
0.105 ms. D2H took 0.145 ms on device and 0.137 ms of host wait, followed by
0.146 ms pinned-to-RAM memcpy (5.81 GiB/s effective). A direct SSD read of
the same bundle took about 1.87 ms. The primitive times justified one bounded
end-to-end screen, but did not establish a throughput gain.

The matched 128-token screen used the same greedy prompt, model, fixed GPU
expert pool, 4,400 MiB RAM cache, and 262,144-token logical / 32,768-token
initial physical KV. Its shadow budget was 1,224 MiB. The generated text was
identical; active GPU hits, CPU misses, promotions, and evictions were also
identical at 17,891 / 23,069 / 3,371 / 2,523. No route trace was captured in
this screen, so it establishes counter parity rather than a row-by-row routing
proof for the rescue variant. An earlier 128-token shadow screen recorded
48,320 identical routing/GPU-hit rows against its baseline.

| 128-token measure | Shadow off | Shadow on, checksum off |
| --- | ---: | ---: |
| Decode throughput | 5.544 tokens/s | 5.406 tokens/s |
| Flash read | 13,134.5 MiB | 10,644.1 MiB |
| Flash I/O stall | 72 ms/token | 57 ms/token |
| RAM-cache management | 12 ms/token | 42 ms/token |
| Shadow hits / misses | — | 1,466 / 9,094 |
| Insert attempts / completed / dropped busy | — | 7,953 / 7,817 / 136 |
| H2D | — | 13,191.2 MiB / 1,462.5 ms device |
| D2H | — | 2,473.9 MiB / 227.6 ms device |
| Staging / restore memcpy | — | 2,250.7 / 760.8 ms |
| Shadow management | — | 3,739.9 ms total, 29.2 ms/token |

With insertion/restore checksums enabled, the same output had zero detected
byte mismatches, but throughput fell to 2.415 tokens/s; that diagnostic run
was not used for the throughput decision. Even without checksum cost, the
shadow path was about 2.5% slower than its matched baseline. The prototype
and its CLI/config/CMake surface were removed. The experimental logs remain
outside the tracked source tree; there is no shadow runtime option.
