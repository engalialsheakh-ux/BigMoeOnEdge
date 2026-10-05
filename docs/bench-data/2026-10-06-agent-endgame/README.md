# Agent latency: targeted decode, prefill, and partition evidence

The goal is exact Qwen3.6 hybrid inference on the local 16 GB Windows CUDA
machine. The frozen 64K adaptive prefill result remains 55.50 tokens/s with
44,620 rereads and exact generated text; it was not rerun. No quality setting,
expert top-k, KV type, cache size, or context capability was reduced.

## Decode decision

A short deterministic warm-session checkpoint run generated 16 tokens at
6.618 tokens/s. Its per-token telemetry reported 79.3 ms residual compute,
56.4 ms unhidden expert wait, and 15.3 ms cache management. This is a short,
partially warm run and does not replace the earlier approximately 9 tokens/s
longer-context regime. The prior exact-mode investigation already measured
held-out GPU residency, cross-layer prediction, scheduler overlap, and CUDA
graph dispatch. Its optimistic graph-launch saving was approximately
3.1 ms/token (2.9% of the 64K reference); predictor replay reduced GPU
misses by at most 0.20% while requiring more than twice the upload volume.
No new safe decode change cleared the material-gain gate. The existing
asynchronous promotion and exact CPU/GPU branches remain the default.

## Remaining cold-prefill cost

An opt-in timing probe on a 1,351-token cold prefill measured 25.306 s total.
Of this, 24.979 s was inside `graph_compute`, while graph construction and
allocation took 0.028 s, input setup 0.072 s, explicit synchronization
0.002 s, and activation readback 0.194 s. The timer was removed after the
measurement. Demand stall and cache management were 3.382 s and 0.792 s;
these overlap other terms and must not be added to the graph timings.

The frozen 64K adaptive run reported only 38.642 s of combined demand stall
and cache management in 1,180.404 s of prefill wall time (3.27%). Thus
eliminating every remaining such wait would still have a small ceiling.
The dominant `graph_compute` interval includes required expert math and the
scheduler, and the measured graph setup, copies, and explicit sync are too
small to justify a risky fork rewrite. No speculative cold-prefill change was
retained.

## Exact routed-set partition replay

`scripts/prefill-routing-partition-replay.py` reads the existing exact
3,401-token, 64-token-microbatch access trace. For each candidate segment
`[s,e)`, the additive planning cost prices its actual native routed expert
union against current replayed RAM residency, plus a per-tile cost. It obeys
the 64 MiB activation cap and replans after each chosen tile. Expert compute,
per-microbatch launches, and activation readback are fixed by the token count
and therefore cancel in boundary selection. GPU expert slots are decode-only
on this path, so prefill RAM-to-VRAM expert movement is zero. The cache replay
then evaluates the selected boundaries with real eviction and reread rules.
Future router decisions come from the recorded trace: making them available
online before expert execution would require extra routing graphs. This is an
offline upper-bound experiment, not a new runtime mode.

```powershell
py scripts/prefill-routing-partition-replay.py `
    build-cuda/expert-endgame-probe/baseline-repeat.access
py scripts/prefill-routing-partition-replay.py `
    build-cuda/expert-endgame-probe/baseline-repeat.access `
    --max-tile 1024 --activation-limit-mib 8
```

| Cap | Policy | Tiles | Expert misses (MiB) | Rereads | Peak activations (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 64 MiB | adaptive auto | 1 | 14,880.4 | 0 | 26.6 |
| 64 MiB | simple capacity | 54 | 267,845.1 | 149,905 | 0.5 |
| 64 MiB | routed-set cost optimum | 1 | 14,880.4 | 0 | 26.6 |
| 8 MiB | adaptive auto | 4 | 39,340.7 | 14,495 | 8.0 |
| 8 MiB | simple capacity | 54 | 267,845.1 | 149,905 | 0.5 |
| 8 MiB | routed-set cost optimum | 4 | 39,340.7 | 14,495 | 8.0 |

The strongest routed-set planner found no advantage over existing adaptive
tiling on this recorded trace. The simple capacity rule is much worse because
the aggregate routed set over all 40 layers exceeds RAM even for one small
microbatch, although each layer's bank fits. Planning overhead and extra
router graphs therefore cannot pay on this evidence. The replay tool remains
for future traces; the measured agent configuration keeps `--prefill-layer-tile auto`.

## Integrated coding-agent request

The final request was a code-review task over 65,000 characters of this
repository's `session.cpp` plus a short instruction, saved locally as
`build-cuda/final-agent-prompt.txt` (65,484 UTF-8 bytes, SHA-256
`8dbb71d6aa0cbb86a10de503d98d732d54d6f38ebba0a89b24fdc5a5e259bd7d`).
The source text was taken before the later GPU exactness guard, from parent
commit `4b6e755`; the prompt is deliberately kept in the ignored build tree.
Regenerate it with
`py docs/bench-data/2026-10-06-agent-endgame/prompt-recipe.py build-cuda/final-agent-prompt.txt`.
The model was the existing Qwen3.6-35B-A3B Q4_K_M GGUF. The run used a
262,144-token context, Q4_0 K/V on CUDA, 4,400 MiB RAM expert cache,
1,500 MiB GPU expert pool, 128 MiB safety rail, adaptive layer tiles,
greedy decoding, and `--gpu-expert-no-promote`. The pool stays allocated but
empty to keep the output deterministic. No model or asset was downloaded.

Two identical fresh `--session` requests, each capped at 1,292 generated
tokens, ran in one process. The second automatically restored the persisted
hybrid prefix. A third request after process restart checked disk reuse with
a 16-token continuation. All three started with the same 16 output tokens;
the two full 1,292-token outputs were byte-identical (display-text SHA-256
`f788ad728d9dfe5fc291df708b8b1a4eb60d4506cc936d77029f5e21d32423cf`).

| Quantity | Cold | In-process cached | Restart cached |
| --- | ---: | ---: | ---: |
| Prompt tokens actually prefilled | 17,663 | 255 | 255 |
| Prefix tokens restored | 0 | 17,408 | 17,408 |
| Prefill seconds | 337.929 | 10.622 | 11.206 |
| State restore seconds | — | 0.038 | 0.593 |
| First generated-token wall seconds | 0.380 | 0.339 | 0.541 |
| Request first response seconds, excluding model load | 338.309 | 10.999 | 12.340 |
| Model load seconds | 9.301 | already loaded | 8.780 |
| First response seconds, including a fresh model load | 347.610 | — | 21.120 |
| Generated tokens | 1,292 | 1,292 | 16 |
| Decode tokens/s | 8.126 | 7.802 | 6.873 |

The checkpoint saved 17,408 of 17,663 prompt tokens (98.56%). It occupied
158.8 MiB of sequence state (166,622,224 bytes including the file header and
tokens), and capture took 0.472 s. Sampled process private memory during the
large run was about 10.9 GiB, not a measured peak. The configured RAM expert
cache reported 4,399 MiB resident; CUDA Q4_0 KV reserved 1,440 MiB, and the
GPU expert pool reserved 1,500 MiB with zero promoted experts. The decode-rate
difference between the two exact runs is not claimed as a speed change; the
accepted gain is first-response latency from avoided prefill. This result
does not show faster one-token model math.

An initial run with dynamic GPU expert promotions did not meet the output
identity gate: the cold and cached 1,292-token texts differed after 50 tokens,
and two cached 128-token continuations also differed. Two cached 128-token
runs with promotions disabled matched, as did a separate cold/cached
128-token real-model pair. This motivated the explicit configuration guard;
the initial run is excluded from the exactness result above.

The accepted code changes do not alter general cold-prefill execution when
`--agent-state-cache` is omitted, and the replay planner is offline only.
Therefore the 64K benchmark was not rerun. The frozen 55.50 prefill tokens/s
and 44,620 expert rereads remain the long-context reference; no new 64K claim
is made.
