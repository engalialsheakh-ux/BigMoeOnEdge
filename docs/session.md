# Session mode

A fresh `bmoe-cli` process per prompt re-pays two fixed costs every time: the model load (tens
of seconds for a >RAM model) and the expert-cache warm-up ramp (the LRU cache starts empty, so
the first tokens miss often and read far more from flash than the steady state — see
[benchmarks.md](benchmarks.md)). Streaming was meant to avoid exactly this kind of repeated work.

Session mode amortises both. `Session` (`core/include/bmoe/session.h`) loads the model, discovers
the MoE expert tensors, and initialises the expert source **once**; each `generate()` then runs a
prompt against that resident state, and the expert LRU cache normally **survives between calls**,
so a second prompt starts warm unless a GPU cache resize resets its residency.

```cpp
std::string err;
auto s = Session::open(cfg, err);      // model load + capture + source.init — once
s->generate({.prompt = "..."});        // prefill + decode; cache filled
s->generate({.prompt = "..."});        // starts warm — no reload, no cold ramp
```

`run()` (`runtime.h`) is a thin one-shot wrapper over `Session` — open, one generate, close — so
the byte-identity gates exercise the same machinery an interactive session uses. Gates **S1/S2**
assert that a second, warm generate produces output identical to the cold one-shot reference:
warming the cache changes latency, never the bytes.

## Independent prompts vs multi-turn chat

Session JSON generation also accepts structured `messages`, `tools`, and `tool_choice`
with `--chatml`. These replace the single `prompt` for that request, use the GGUF chat
template, and return parsed assistant tool calls in `BMOE_DONE`. A client can send the
assistant call and matching tool result back as chat messages for the next turn.

`GenerateRequest::clear_kv` selects between two modes:

- **`true` (new chat / independent prompt):** the KV cache and the engine-held conversation are
  dropped before the prompt runs. The expert cache normally stays warm; a GPU cache resize resets
  its residency. This is the one-shot path — `run()` always uses it, and the byte-identity gates
  exercise it.
- **`false` (continue):** the prompt continues the current conversation. In chat mode the Session
  owns the conversation (`chat_history`) and re-renders the model's chat template over the *whole*
  history each turn, so the caller sends only the new user message — not the running transcript.

**KV prefix reuse.** Re-rendering the full history would re-tokenize the entire conversation, but
most of it is already decoded into the KV. Each turn the engine diffs the freshly rendered tokens
against `kv_tokens` (the tokens currently in the KV, in order), keeps the common prefix, removes the
divergent tail with `llama_memory_seq_rm`, and prefills only the suffix. So a follow-up turn pays
for its own tokens, not a full re-prefill — which matters because prefill is the slow phase on
device. `BMOE_DONE.n_prompt` reports the tokens actually prefilled this turn; `n_past` is the total
context length after it.

**Persistent fresh-request prefix reuse.** `SessionConfig::agent_state_cache_path` optionally
keeps one exact hybrid sequence checkpoint in memory and on disk. A fresh request whose tokenized
prompt starts with the saved tokens restores attention KV and recurrent state, then prefills only
its suffix. The model/config identity and a checksum reject stale or corrupt files; a miss runs
the normal cold path. This complements live chat continuation. The persistent checkpoint is
consulted only for `clear_kv=true`. With a GPU expert pool, dynamic expert promotion must be
committed between decode tokens; each independent request starts with empty logical GPU residency
and fresh graph discovery. See [agent-state-cache.md](agent-state-cache.md).

The session also keeps one in-memory checkpoint at token 4096 for longer fresh prompts.
If a later request changes after that point but before the rolling checkpoint, it restores
the earlier exact state and evaluates the remaining suffix. With MTP, both checkpoints
include the target sequence, draft sequence, and the draft driver's carried hidden row.

**Fallbacks and costs.** SWA-style memory (e.g. Gemma) can refuse a partial `seq_rm`; the engine
then clears the KV and re-prefills the whole prompt for that turn (correct, just slower). With
thinking **on**, the template strips the previous turn's reasoning on re-render, so the rendered
prefix diverges at the last answer and up to one answer's worth of tokens is re-prefilled per turn;
with thinking **off** (the app default) the re-fed suffix is just the new user turn plus a few
wrapper tokens.

**Reasoning is returned, not discarded.** On a thinking model the Session parses the reasoning span
out of the raw stream and carries it in its own field (`TokenMetrics`/`RunResult::reasoning`,
`delta_reasoning` on `BMOE_PROGRESS`, `reasoning` on `BMOE_DONE`) rather than dropping it. The answer text stays free of
it either way, so the byte-identity gates are unaffected; a caller that wants to show the thinking
reads the separate field. The parser wiring lives in `core/src/engine/chat_parse.cpp`
(see [seam.md](seam.md)).

## Cancel

`Session::cancel()` is thread-safe and interrupts an in-flight `generate()` at the next decode
boundary via llama's abort callback (installed unconditionally at open, so it works in serial and
overlap alike). It leaves the model and cache intact; the returned `RunResult` has `cancelled =
true`. In chat mode a cancel **rolls the turn back** to the reused prefix (dropping this turn's KV
and un-appending the user message) so prior turns stay usable and the conversation can continue.
Cancel is distinct from a fatal streaming error, which is sticky and ends the session.

## Fixed context

`n_ctx` and `n_batch` are baked into the llama context at `open()`, before any prompt is known, so
size them for the longest prompt + generation the session will serve. A request that would overflow
`n_ctx` is rejected without tearing the session down.

With Elastic KV enabled, `n_ctx` remains the logical limit while the physical KV allocation changes
in configured chunks. If KV growth needs dedicated VRAM, the session drains expert uploads, saves
the target and MTP draft sequence state plus the MTP driver's carried hidden row, and shrinks the GPU
expert pool before continuing. The pool
keeps its per-layer dummy and real slots; if even that minimum cannot fit above the configured
safety rail, growth fails and the prior sequence is restored. A later request whose prompt and
requested output fit at least two chunks below the current physical capacity can shrink KV at the
request boundary. The released memory can regrow the expert pool toward its original budget.
Context and pool changes do not reload the model or restart the session. CUDA allocation rounding
is checked against the dedicated-VRAM safety rail after allocation, with 64 MiB reserved for later
graph allocations. `physical_kv_tokens()` and
`gpu_expert_cache_bytes()` report the current allocations.

## CLI and app

`bmoe-cli --session` exposes this over a line protocol (requests on stdin, `BMOE_*` responses on
stdout — see [telemetry.md](telemetry.md)). The Android example runs one such process per model:
the first prompt loads the model, later prompts reuse the warm process, and the session is freed on
an explicit **Unload** or after an idle timeout. Changing the model or any streaming setting
reopens the session; changing only the prompt, `n_predict`, or the thinking toggle does not.
