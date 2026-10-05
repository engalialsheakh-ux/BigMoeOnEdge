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
for future traces; the runtime default stays adaptive auto.
