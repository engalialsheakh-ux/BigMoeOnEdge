# Persistent agent prefix state

`bmoe-cli --agent-state-cache FILE` enables one persistent checkpoint of a prompt's
hybrid sequence state. It is optional and works with one-shot runs and `--session`.
When a later request starts with the same token sequence, the engine restores the
checkpoint and prefills only the remaining suffix. A process restart can reuse the
same file. The file contains llama.cpp's sequence state and the checkpoint tokens;
it contains no model weights or expert-cache entries.

The persistent checkpoint is taken at the largest 512-token boundary before the final prompt
token for prompts longer than 1024 tokens. Between 129 and 1024 tokens the
boundary is 64 tokens. For prompts extending beyond 4096 tokens, the session also keeps
one in-memory checkpoint at 4096 when the rolling boundary lies later. A change between
these boundaries reuses the earlier checkpoint; a change before both runs a cold prefill.
Only the rolling checkpoint survives a process restart.
For multi-turn chat, ordinary live-context continuation still applies when
`clear_kv=false`; this file is used automatically for fresh requests.

The file is accepted only when its exact saved tokens match the start of the new
tokenized prompt, its checksum is valid, and the model identity and relevant
context, KV, placement, routing, and prefill settings match. Model identity uses
the canonical path, file size, modification time, and sampled contents. Standard
model replacement invalidates the file. If the file is incompatible or corrupt,
the engine performs a cold prefill and replaces the checkpoint. Greedy sampling
and exact, unspeculated routing are required. A snapshot larger than 1 GiB is
not kept, to protect RAM on a 16 GB machine.

With a GPU expert pool, active promotions now replay in deterministic token
order. Every independent `clear_kv=true` request drains outstanding uploads,
clears logical GPU residency without reallocating the pool, resets the
per-request reuse history, and rediscovers the first one-token graph's GPU
cache branch. The uploader still transfers expert bundles in the background
between tokens; the next decode waits for those transfers before selecting
CPU or GPU experts. Thus transfer completion cannot change placement halfway
through a decode. The checkpoint contains no expert weights, and a checkpoint
made with `--gpu-expert-no-promote` can be restored with promotions enabled:
promotion is decode-only and does not alter the saved prefix state.

The earlier unrestricted path diverged after 50 output tokens on a long run.
Tracing repeated short restores isolated two causes: the second request kept
GPU residency and graph-discovery state from the first, so its first GPU
execution decision differed before routing diverged; later promotion decisions
then followed that changed routing. `--gpu-expert-no-promote` remains an
all-miss control. The deterministic mode retains the full configured GPU pool
and exact model top-k; its transfers can add a small token-boundary wait.

The checkpoint captures both attention KV and recurrent/SSM state through
llama.cpp's public sequence-state API. With MTP, it also captures the draft context's
sequence state and the driver's carried hidden row. MTP snapshot files use `BMOEAS03`;
older draft snapshots are rejected and rebuilt because they lack the driver row.
A focused gate mutates the state, restores
it twice, checks recurrent-state bytes, and compares subsequent logits. The
integration gate checks in-process reuse, restart reuse, a changed prefix,
an incompatible execution setting, and a corrupt file. Creating a checkpoint
splits the cold prompt at its boundary;
floating-point execution can therefore differ from an unsplit prompt. The exact
comparison for cache correctness is between the checkpoint-producing cold run
and the same run after restore. No experts are skipped and no model math changes.

For the local hybrid Q4_0 K/V CUDA configuration, a short 1,351-token prompt
produced a 68.5 MiB checkpoint after 1,024 tokens. Capture took 0.184 s; restore
took 0.013 s in-process and 0.173 s from disk after restart. The same 327-token
suffix and 16-token deterministic continuation matched in all three runs.
This initial diagnostic predates deterministic GPU-residency resets and is not
used as an exactness result.
Prefill took 28.134 s cold, 6.669 s with an in-process checkpoint, and 7.516 s
after restart. The second run also had a warmer expert cache, so the difference
is not attributable to state reuse alone. These are short-run observations, not
the final agent-workload benchmark.

The [agent-latency endgame report](bench-data/2026-10-06-agent-endgame/README.md)
records the earlier 17,663-token, 1,292-output-token all-miss baseline.
The [deterministic GPU-cache experiment](bench-data/2026-10-06-deterministic-gpu-agent-cache/README.md)
records the new replay and throughput comparison.

Example:

```powershell
$model = 'MODEL.gguf'
$checkpoint = 'D:\cache\agent-prefix.bin'
& build-cuda/cli/Release/bmoe-cli.exe --session --model $model `
    --ctx-size 262144 --ubatch 64 --moe-stream --n-gpu-layers -1 `
    --gpu-expert-cache-mb 1500 --gpu-expert-window 4 `
    --cache-type-k q4_0 --cache-type-v q4_0 `
    --agent-state-cache $checkpoint
```

Use an existing writable directory for `FILE`. The engine logs checkpoint
size, capture time, restored token count, and restore time to stderr.
