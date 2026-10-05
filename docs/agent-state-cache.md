# Persistent agent prefix state

`bmoe-cli --agent-state-cache FILE` enables one persistent checkpoint of a prompt's
hybrid sequence state. It is optional and works with one-shot runs and `--session`.
When a later request starts with the same token sequence, the engine restores the
checkpoint and prefills only the remaining suffix. A process restart can reuse the
same file. The file contains llama.cpp's sequence state and the checkpoint tokens;
it contains no model weights or expert-cache entries.

The checkpoint is taken at the largest 512-token boundary before the final prompt
token for prompts longer than 1024 tokens. Between 129 and 1024 tokens the
boundary is 64 tokens. There is only one checkpoint in memory and on disk. This
limits memory use, but a change before the saved boundary causes a cold prefill.
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

When a GPU expert pool is enabled, use `--gpu-expert-no-promote` with this
checkpoint feature. Dynamic CPU/GPU expert placement can change rounding and
make two greedy continuations of the same state diverge: a long local run
matched for 50 tokens before the outputs differed, and two identical restored
runs also differed. With promotions disabled, a real-model cold/restore pair
and two repeated restores matched over 128 generated tokens. The pool's
configured size remains unchanged; it stays empty for this exactness mode.
The CLI rejects a cache plus active GPU promotions rather than silently
claiming a deterministic continuation. This exactness requirement can reduce
decode throughput relative to dynamic promotions.

The checkpoint captures both attention KV and recurrent/SSM state through
llama.cpp's public sequence-state API. A focused gate mutates the state, restores
it twice, checks recurrent-state bytes, and compares subsequent logits. The
integration gate checks in-process reuse, restart reuse, a changed prefix, and a
corrupt file. Creating a checkpoint splits the cold prompt at its boundary;
floating-point execution can therefore differ from an unsplit prompt. The exact
comparison for cache correctness is between the checkpoint-producing cold run
and the same run after restore. No experts are skipped and no model math changes.

For the local hybrid Q4_0 K/V CUDA configuration, a short 1,351-token prompt
produced a 68.5 MiB checkpoint after 1,024 tokens. Capture took 0.184 s; restore
took 0.013 s in-process and 0.173 s from disk after restart. The same 327-token
suffix and 16-token deterministic continuation matched in all three runs.
This initial diagnostic used dynamic GPU promotions, so it is not the final
exactness mode.
Prefill took 28.134 s cold, 6.669 s with an in-process checkpoint, and 7.516 s
after restart. The second run also had a warmer expert cache, so the difference
is not attributable to state reuse alone. These are short-run observations, not
the final agent-workload benchmark.

The [agent-latency endgame report](bench-data/2026-10-06-agent-endgame/README.md)
contains the final 17,663-token, 1,292-output-token cold/cached comparison in
the deterministic GPU placement mode.

Example:

```powershell
$model = 'MODEL.gguf'
$checkpoint = 'D:\cache\agent-prefix.bin'
& build-cuda/cli/Release/bmoe-cli.exe --session --model $model `
    --ctx-size 262144 --ubatch 64 --moe-stream --n-gpu-layers -1 `
    --gpu-expert-cache-mb 1500 --gpu-expert-no-promote `
    --cache-type-k q4_0 --cache-type-v q4_0 `
    --agent-state-cache $checkpoint
```

Use an existing writable directory for `FILE`. The engine logs checkpoint
size, capture time, restored token count, and restore time to stderr.
