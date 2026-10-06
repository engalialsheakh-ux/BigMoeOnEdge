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

| Completion | Earlier baseline I/O | Two-wave I/O | Result |
| --- | ---: | ---: | --- |
| 128 tokens, decode tokens/s | 8.408 | 8.810 | Two-wave +4.8% |
| 128 tokens, stall ms/token | 40.1 | 37.9 | Two-wave -2.2 ms |
| 256 tokens, decode tokens/s | 9.804 | 9.768 | Two-wave -0.4% |
| 256 tokens, stall ms/token | 33.4 | 33.0 | Two-wave -0.4 ms |

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
and memory budget remain the result of this earlier investigation. No production
inference code or llama.cpp source changed. A material step from that result
toward 11 tokens/s would require saving roughly 17 ms of its 108 ms token,
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

Elastic GPU KV capacity is DEFERRED-BLOCKED by the current storage layout. For
this hybrid model, memory creation in `llama-model.cpp` passes `cparams.n_ctx_seq` as
`attn_kv_size` to `llama_memory_hybrid`, whose attention member is a
`llama_kv_cache`. Its constructor creates full-`kv_size` K/V tensors with
`ggml_new_tensor_3d` and immediately allocates their backend storage with
`ggml_backend_alloc_ctx_tensors_from_buft`. The cache's `state_write_data` and
`state_read_data` address saved cells through the current cache dimensions;
`state_read_meta` also finds slots in the already-sized cell array. A growing
physical layout must preserve those serialized positions and exact prefix
restore. Separately, `GpuExpertPool::init` fixes slot counts and offsets from
the configured budget, then allocates one device buffer of that budget. Using
freed KV VRAM for experts and later returning it would require coordinated KV
growth, compatible state read/write, and pool resizing only after uploads and
graph pins drain. A flag or one allocator edit cannot safely provide that
single-sequence behavior while retaining the 262,144-token logical context.
