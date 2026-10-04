# The seam: how we hook llama.cpp without forking it

The serial streamer connects to llama.cpp through two public mechanisms. Optional
overlap and mixed CPU/CUDA expert execution use the pinned fork extensions below;
quantized CUDA FlashAttention is a backend change in the same fork. This file
documents the contracts so they can be re-verified when the submodule is updated.

## 1. The eval-callback

`llama_context_params.cb_eval` / `cb_eval_user_data` (public) install a function called by
`ggml_backend_sched` for every graph node:

- `callback(node, ask=true, ud)` is called for each node. Returning true isolates that
  node: the scheduler computes it alone, `ggml_backend_synchronize`s, then calls
  `callback(node, ask=false, ud)`.
- Returning false groups the node with its neighbours for normal computation (no non-ask
  callback).

We use both phases:

**Capture phase** (one warm-up decode). `ask` is called for every node, so we scan each
node's `src[]` for expert weight tensors (`blk.<il>.<suffix>.weight`, where the suffixes
come from the arch's recipe — `ffn_{gate,up,down}_exps` for the split layout, a fused
`ffn_gate_up_exps` for others) and record the live `ggml_tensor*`. We return false
throughout — capture observes, it does not isolate. `ggml_tensor` is a public struct, so
reading `->name`, `->ne`, `->nb` and writing `->data` is public API surface.

**Stream phase** (real generation). We return true for `ffn_moe_topk-<il>`. The
non-ask callback then hands us that node with the selected expert ids materialized; we
gather them (stride-aware) and trigger the slice reads.

Two optional jobs ask for more: the route trace and
[cache-aware dropping](expert-dropping.md) also want each layer's `ffn_moe_weights*-<il>` chain,
which is another barrier per node but no new kind of access — same public struct, same read of
`->data`.

The two lossy routing policies go one step further, and they are the only places the engine
**writes into** a graph tensor's contents rather than repointing `->data` at its own buffer.
Dropping, at the terminal node of the weight chain, zeroes a dropped slot's weight and repoints
that slot's expert id; [substitution](cache-aware-substitution.md), at the `ffn_moe_topk` node,
rewrites the selected ids toward resident experts. In both cases the tensors are scratch the graph
produced and has not yet consumed, so this alters the values flowing through the run —
deliberately, that is what a lossy policy *is* — and never llama.cpp's own state, its weights, or
its control flow. It stays inside the same callback contract; nothing is patched.

## 2. gguf offsets

`gguf_init_from_file(..., no_alloc=true)` + `gguf_get_data_offset` +
`gguf_get_tensor_offset` (all public) give each tensor's absolute byte offset in the file,
without loading any tensor data. We match these to the captured tensors by name.

## 3. The expert-ready hook (fork extension)

Sections 1 and 2 are enough for the *serial* streamer: block on the expert reads, then let
the layer compute. Overlapping the two — reading a token's experts while the same token's
expert matmuls are running — needs a wait point that no public API exposes. That is the one
place where the released BigMoeOnEdge pin carries a llama.cpp extension.

**What it is.** A single optional hook, ~25 lines, living on the fork branch
`bmoe/expert-ready-hook` of `Helldez/llama.cpp` as a single commit on top of the upstream
pin. It adds nothing to the model files and changes no data layout; it is a callback the
CPU MoE kernel invokes.

**Exact API and call site.**

```c
// ggml-cpu.h
void ggml_cpu_set_expert_ready_hook(ggml_expert_ready_hook_t hook, void * user_data);
```

`ggml_compute_forward_mul_mat_id` calls the hook at the top of its per-expert loop, right
after the "expert not routed this token" skip, before it consumes that expert's weight
slice. Every compute thread calls it for every routed expert; the hook may block. There is
**no barrier inside the expert loop**, so a thread blocking on one expert cannot deadlock
the threadpool — other threads proceed to the experts whose slices are already resident.
The streamer's hook blocks until the requested expert slice has been read in, then returns.
When no hook is registered (stock upstream, or `--overlap` off) the call is a single null
check — zero cost.

**Why it exists.** The topk eval-callback (section 1) is the only public hook near routing,
and it can only fire *before* the expert matmuls of a layer run — it cannot pause partway
through them. Overlapping expert reads with expert compute requires a per-expert wait point
*inside* the kernel, which the public API does not provide. Hence the extension.

**Graceful degradation.** CMake probes `ggml-cpu.h` for the hook symbol and, when present,
defines `BMOE_HAVE_EXPERT_READY_HOOK`. Built against stock upstream (symbol absent) the
whole project still compiles and runs — the serial streaming path is unchanged; only
`--overlap` is affected, and it fails with a clear runtime error instead of silently
falling back.

**Sunset condition.** The released fork carries this hook until upstream exposes an
equivalent per-expert readiness callback. The pinned mixed expert-cache extension below
is an additional opt-in fork delta and must be reviewed before any submodule bump.

## Experimental mixed CPU/CUDA expert cache

The pinned GPU-cache fork revision extends the `ggml-cpu.c` `MUL_MAT_ID` kernel with an
optional `src[3]` I32 table indexed by original expert ID. `op_params[0]` gives the dummy
GPU slot. A routed ID whose table entry differs from that dummy is a GPU hit: thread zero
zeros the corresponding CPU destination row and omits it from the kernel's expert work
list. The existing expert-ready hook therefore runs only for CPU misses. A null `src[3]`
keeps the released kernel behavior. The standalone `cpu_expert_skip` gate checks the zero
rows, exact miss output, and hook calls with distinct expert weights.

The local `llama_set_expert_cache_layers` extension gives the one-token split
gate/up/down SiLU graph stable external weight and mapping tensors. The graph runs
the CPU miss path and a CUDA hit path, adds their rows, then applies the original
router weights. Other graph forms keep the original CPU expert path. The session
attaches the descriptors only for one-token decode, keeps their metadata alive until
after `llama_context`, and changes a layer's map at the top-k barrier before either
expert branch consumes it. Selected hit slots stay pinned through `llama_decode()`;
the map is cleared before unpinning. Promotion uses only an already-resident RAM
bundle, copied into bounded staging after the token, and publishes the GPU slot only
after all projections upload. The focused `gpu_expert_mixed` gate covers all miss,
mixed, all hit, multithreaded CPU execution, and a changing map.

The CUDA fallback `MUL_MAT_ID` sorter in the prior pin assumed each selected
slot occurred once. Multiple misses share dummy slot zero, so the fork delta
retains every occurrence when sorting the rows. The all-miss case in
`gpu_expert_mixed` catches a regression. The mixed-graph bridge is commit
`1647b34c3` on the Helldez/llama.cpp fork.

The pinned fork also includes commit `82c26e6a9` on local branch
`bmoe/exact-quantized-fa-baseline`. It adds eligible Q4_0/Q8_0 CUDA
FlashAttention tile expansion and dispatch diagnostics. This is a CUDA backend
implementation change, not a new BigMoeOnEdge API seam. The CUDA operator gates
and the [baseline freeze](bench-data/2026-10-04-baseline-freeze/README.md) record
the verified shape and runtime configuration. The fork commit and the parent
submodule pin are local until separately reviewed for publication.

## The chat glue: llama.cpp `common` (not the streaming seam)

Separate from the two streaming hooks above, `session.cpp` links llama.cpp's `common`
library for two things: rendering the model's own chat template and parsing reasoning output, and
(with `--mtp`) driving speculative decoding.
`common_chat_templates_init` / `common_chat_templates_apply` run the real Jinja template the
gguf ships (so Gemma's channel format, Qwen ChatML, etc. all format correctly, driven by the
model rather than hardcoded), and `common_chat_parse` extracts a reasoning model's thinking so
it can be reported apart from the answer. The parser-params wiring lives in its own translation
unit, `chat_parse.cpp` — the PEG parser arena has to be loaded explicitly or `common_chat_parse`
throws on the first token, which is how issue #49 stayed invisible; keeping it separate makes
that seam unit-testable without a model.

A second translation unit, `thinking_control.cpp`, crosses the same boundary for "thinking off".
`enable_thinking` is only a *request* to the template, and many templates never read it, so the
engine renders the template to find out (three renders at open, no model names involved) and, where
the flag is inert *and* reasoning is a structural section of the format, asks for a **continuation**
instead: the `continue_final_message` field of `common_chat_templates_inputs`, plus a synthetic
trailing assistant message, makes llama.cpp's own per-template handler emit that family's "reasoning
is over" span into the prompt. This is why no `<think>` or harmony channel marker appears anywhere in
`core/` — the markers stay upstream, where a submodule bump keeps them current.

Whether the continuation is *binding* is read off `common_chat_params::thinking_start_tag`/
`thinking_end_tag`: a model that declares a reasoning span owns it, so a pre-closed empty one is a
suggestion it can decline (LFM2.5 does), while a model that declares none separates reasoning
structurally and cannot. Both facts come from the loaded model, never from its name.
`tests/think_control_test.cpp` pins all of it against the vendored templates, again with no model.

### Speculative decoding

Only the MTP source crosses into `common`. The verify half of the loop — the wide batch, the argmax
acceptance, the KV rollback — is written against public `llama.h` alone, and so is the n-gram draft
source, which is why `--ngram` needs nothing from `common` at all. `core/include/bmoe/ngram_draft.h`
does not even include `llama.h`: it is written over `int32_t` token ids, which is what `llama_token`
is, so the drafting policy stays on the pure-policy side of the seam and is unit-tested with no model
and no native backend (`tests/ngram_test.cpp`). See [ngram.md](ngram.md).

`--mtp` reaches `common/speculative.h`, which is a header of the same `common` library and
includes only `llama.h` and `common.h`. The draft/verify orchestration — running the trained MTP
head, moving hidden states from the target to it, seeding the next draft from the accepted position
— is entirely upstream's; the engine supplies the loop around it and the two contexts it works on.
Worth stating explicitly because the internals it needs (`llama_set_embeddings_nextn` and the
`nextn` hidden-state getters) live in `src/llama-ext.h`, a *staging* header: they are used **inside**
`speculative.cpp`, which is already compiled into the `llama-common` the engine links, so nothing in
`core/` includes a private header and no in-tree patch is involved.

What the engine does own is the pair of contexts. Self-speculation is one model with two contexts
over it — the target, and a draft created with `ctx_type = LLAMA_CONTEXT_TYPE_MTP` — and the engine
builds the draft one itself rather than through `common_speculative_init_from_params`, for a reason
that belongs to this project: **the eval callback is per-context**. The streamer only sees the MTP
block's expert layer if the draft context carries the same `cb_eval`, and `init_from_params` derives
its context parameters from a full `common_params` with no way to inject one. See
[mtp.md](mtp.md).

Unlike the public-C-API streaming seam, `common` is **not a stable API** — it can change
between upstream versions. So a submodule bump may require updating this chat glue in
`session.cpp` / `chat_parse.cpp` / `thinking_control.cpp`; the build and gates catch a break at
compile time rather than at runtime (`tests/chat_parse_test.cpp` and
`tests/think_control_test.cpp` cover these seams directly). This
trade-off is deliberate and is also noted at the link site in the root `CMakeLists.txt`. The
gates themselves run with the template off (raw prompt), so they stay deterministic and are
unaffected by this dependency.

## The one ggml behaviour we depend on

That a node marked "needed" is computed and synchronized **before** the non-ask callback,
and that the batch containing the dependent expert matmul runs **after** the callback
returns. This is how `ggml_backend_sched` implements the eval-callback today
(`ggml/src/ggml-backend.cpp`). It is not a stability-guaranteed contract, so:

- the [byte-identity gates](../tests/moe_gates.cpp) assert lossless output, which fails
  loudly if the ordering ever changes;
- CI runs the gates on every submodule bump.

## Per-graph placement (`--prefill-device`)

The prefill device ([npu-prefill.md](npu-prefill.md)) adds no llama.cpp change, but it leans on these
behaviours of the public surface and two naming conventions. All are exercised by gates G16 and G17,
except where noted:

- **The scheduler assigns a backend per graph, from the weight's buffer.** `ggml_backend_sched` looks
  at each weight's `tensor->buffer` every time it splits a graph, so rebinding `buffer`, `data` and
  `extra` (and, for a weight carried to the device in another type, `type` and `nb`) between graphs
  moves the weight's ops. The rebind happens only between `llama_decode` calls.
- **Graph reuse skips that decision.** A graph shaped like the previous one is reused without being
  re-scheduled, so device graphs and CPU graphs are kept to different widths (`--prefill-min-tokens`).
  A llama.cpp change that reused graphs across shapes would break this; G16 would show it.
- **`ggml_backend_tensor_set` is where a backend lays weights out its own way.** The arena writes each
  expert through a per-expert view, and writes each slot tensor once whole at load so the backend
  records its layout for the tensor the op actually reads.
- **A buffer's usage is what a backend reads to decide how to hold it.** `ggml_backend_buffer_set_usage`
  marks a buffer `WEIGHTS` or not, and a backend may lay out and map a `WEIGHTS` buffer for its
  matmul alone (Hexagon with DMA64 maps it for DMA only, and most of its other kernels then refuse it
  at run time). So the engine marks only the matmul matrices `WEIGHTS` and puts every other layer weight
  (norms, biases, scales), as found from the ops the capture graph applies to it, in plain device
  memory.
- **Two context knobs.** `n_outputs_max` bounds the logit rows the context reserves, and toggling
  `llama_set_causal_attn` off and on makes the next decode redo the compute-buffer reservation, which
  the engine uses once, with the weights on the device, so the CPU does not keep a reservation for a
  graph it never runs.
- **Concurrent uploads into one buffer.** The arena's loader threads call `ggml_backend_tensor_set`
  on different views of the same slot buffer at once. ggml does not document this as thread-safe; it
  holds for the CPU, RPC and Hexagon backends today because each write touches only its own bytes.
  The gates cover the CPU (through RPC); on the NPU only a device run does.
- **Layer weights are named `blk.N.`** The engine selects the weights it moves by that prefix. It is
  the GGUF tensor naming every architecture in llama.cpp uses, not an API: a model that named its
  layers otherwise would keep them on the CPU, correct but not faster.
- **Model state tensors are named `cache_`.** The capture tells the KV and recurrent state apart from
  weights by the `cache_` prefix llama.cpp's memory modules give them, so the prefill can place the
  state where both sides can address it. That prefix is internal naming: if upstream renamed it the
  state would stay in CPU memory, again correct but slower, and `prefill_dev_nodes` in the telemetry
  would drop. Check it on each bump.
- **Device selection is explicit for ordinary decode.** With `--n-gpu-layers 0`, the engine gives
  llama.cpp an empty device list and disables operation offload, including in a CUDA-enabled build.
  With CUDA layers requested, it passes only the selected GPU and places routed expert tensors in
  CPU buffers through `tensor_buft_overrides`; `no_host=true` preserves file-backed expert storage.
  The prefill-device path still passes its selected accelerator independently. The CUDA placement
  is covered by a GPU gate and a real-model integration run; see [cuda-hybrid.md](cuda-hybrid.md).
- **A device that fails to open throws.** Hexagon opens its DSP session in `ggml_backend_dev_init`
  and throws when it cannot. The engine calls it once before the load, catches that, and keeps the run
  on the CPU; the backend caches the session, so the context's own init reuses it.

## Upgrading llama.cpp

Because the submodule pins the `bmoe/expert-ready-hook` fork branch (section 3), a bump
rebases that 1-commit branch onto the new upstream tag, re-pushes it, and re-pins:

```bash
# in a Helldez/llama.cpp checkout: rebase the single hook commit onto the new tag
git fetch upstream && git checkout bmoe/expert-ready-hook
git rebase <newer-upstream-tag> && git push --force-with-lease origin bmoe/expert-ready-hook

# in this repo: move the submodule to the rebased commit, rebuild, run the gates
cd third_party/llama.cpp && git fetch origin && git checkout <rebased-commit>
cd ../.. && git add third_party/llama.cpp && scripts/build-host.sh
cd build && ctest --output-on-failure     # gates must stay green
```

When the sunset condition lands (upstream ships the readiness callback) this collapses back
to a plain `git checkout <newer-tag>` against `ggml-org/llama.cpp` with no branch to carry.
Either way the gates are the enforcement. The fragility is not the public
API but the *internal naming conventions* the seam attaches to — the tensor suffixes and
the `ffn_moe_topk` node name are how llama.cpp happens to build MoE graphs today, not a
guaranteed contract, so upstream can rename or restructure them (Gemma 4's fused
`ffn_gate_up_exps` is one such evolution we absorbed with a recipe row). The gates are the
enforcement: a rename breaks byte-identity before merge instead of silently corrupting
output. Each supported architecture adds one more gate to keep green across a bump.

If a future release moves the two hooks (a stable expert-residency API, say) upstream,
this seam shrinks further or disappears — `core/` does not change.

Pinned submodule at the time of writing: `Helldez/llama.cpp` branch
`bmoe/expert-ready-hook-2609`, commit `dce9698`: the single expert-ready-hook commit
(section 3) on top of upstream `ggml-org/llama.cpp` master `965f897` of 2026-09-26 (530 commits
past `b10666`). On this base upstream's CPU `mul_mat_id` has a tiled path that reads an expert
before the classic loop does, so the hook fires ahead of it; G4 is what proves it still gates
every read. Each bump gets its own fork branch and the
previous ones stay, so every commit an old pin names remains reachable (see `.gitmodules` /
`git submodule status` for the current pin).
