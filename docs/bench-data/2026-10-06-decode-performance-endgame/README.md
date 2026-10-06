# Decode-performance endgame: measured limit of the current path

The starting point is the exact persistent-agent configuration in the
[deterministic GPU-cache report](../2026-10-06-deterministic-gpu-agent-cache/README.md):
262,144 context, Q4_0 K/V, native top-8 routing, 4,400 MiB RAM expert cache,
1,500 MiB GPU expert cache, 128 MiB GPU safety reserve, six affined compute
threads, four direct-I/O lanes, and overlap. The frozen long result restored
17,408 of 17,663 prompt tokens, reached first response in 9.717 s, and decoded
1,292 identical tokens at 9.261 tokens/s. No 64K prefill run or new 17K cold
prefill was performed for this investigation.

## Where a token spends time

The prior 1,292-token restored run averaged about 108.0 ms per generated token.
Its existing counters assigned 38.0 ms to non-overlapped expert-read stall,
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
are the largest measured subcost (about 40 ms/token in the matched 128-token
untraced run); it is incorrect to call the whole group CPU compute. A previous
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

| Completion | Current I/O | Two-wave I/O | Result |
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

The current 9.261 tokens/s, 9.717 s restored first response, exact replay,
and memory budget remain the accepted result. No production inference code or
llama.cpp source changed. A material step toward 11 tokens/s requires saving
roughly 17 ms of the current 108 ms token, most plausibly by starting *exact*
expert reads before their layer stalls without adding speculative flash
traffic. That would need a verified early-router/graph schedule and bounded
staging across the CUDA hybrid boundary; the present CUDA mode deliberately
disallows the existing prediction and prefetch policies. Such work needs a
separate measured design and correctness gate. The single largest remaining
bottleneck is routed-expert execution, particularly its unhidden storage wait.
