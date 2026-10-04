# Exact mixed expert execution: design audit

This is the selected implementation direction for the pinned `dce969851` llama.cpp
tree. It is a design hypothesis until the synthetic graph tests prove scheduling,
mapping, and numerical behavior. The current CPU and hybrid CUDA paths remain the
reference when the new option is zero.

## Local constraints

- `Session::open` places routed expert tensors in file-backed CPU buffers with
  `LLM_FFN_EXPS_REGEX`, then `ExpertStreamSource` rebinds their data to exact GGUF
  slices. Strict mapping release is performed after source initialization.
- `RouterHook::c_eval` isolates `ffn_moe_topk`, gathers the selected original IDs,
  and starts the CPU source. The pinned CPU `mul_mat_id` expert-ready hook waits for
  requested slices. The CUDA top-k node may live on device.
- Qwen3.6 uses the generic `llm_graph_context::build_moe_ffn` funnel with separate
  gate, up, and down projections. The weighted expert outputs are summed there.
- The pinned scheduler copies inputs at the start of a split, before node callbacks
  inside that split. A host table changed at the top-k callback is unsafe if a copy
  for its consuming split has already been queued. A map used by `GET_ROWS` must
  live on the selected GPU and be updated after the top-k barrier, before that
  exact remap node executes. The test must prove this ordering under graph rebuilds
  and multiple ubatches; a view over the table must not evade the split boundary.

## Selected path

1. Add `--gpu-expert-cache-mb N`, default zero. Reject unsupported backend and
   architecture combinations explicitly. Keep the existing RAM cache independent.
2. Allocate an explicit-budget pool after the model/context capture and before
   mapping release. Keys are `(layer, original expert id)`. Each bundle owns all
   required projection slices in their original GGUF quantization and layout.
   A zero dummy slot per layer represents misses in the GPU graph. The pool and
   remap tensors must remain alive for every graph using them.
3. Register the model-scoped pool with a narrow, opt-in llama.cpp extension. The
   generic MoE graph builds a GPU hit branch using compact slot IDs, a CPU miss
   branch using original IDs, and one merge after the original weighted sum.
   The host mask makes CPU `MUL_MAT_ID` skip GPU hits before invoking the existing
   expert-ready hook. The GPU dummy slot yields zero for misses. The original
   top-k IDs and routing weights are never rewritten.
4. At the top-k callback, partition each draw against a READY pool snapshot and
   pin all hit slots until the consuming graph completes. Pass only misses to
   `ExpertStreamSource::load_layer`; the source remains the sole authority for
   exact miss bytes. Publish the GPU mapping for this graph/ubatch before its
   remap node runs. Assert `hits + misses == selected draws` and disjointness.
5. A miss may be copied from a stable owned staging bundle after it has been
   served on CPU. Uploads target RESERVED slots and are published only after a
   completion event. No slot can be evicted while pinned or uploading. The first
   correctness gate may use synchronous upload; asynchronous promotion follows
   only after its lifetime tests pass.

## Concrete seam on this checkout

- `core/include/bmoe/config.h`, `core/src/config.cpp`, and `cli/main.cpp` own the
  independent GPU budget and its explicit enablement. `Session::open` in
  `core/src/engine/session.cpp` captures the per-layer expert tensors before
  `ExpertStreamSource::init` consumes that layout. It can reserve the pool after
  the model and context compute buffers exist, then leave the existing
  `release_file_mappings` check authoritative.
- `core/src/moe/gpu_expert_pool.*` owns the bounded device buffer and per-layer
  projection/slot layout. A slot has EMPTY, RESERVED, and READY states and a
  generation counter. The graph must pin READY slots until the consuming work
  finishes. Publication must follow copy completion. A future asynchronous
  upload needs a stable, owned host staging bundle; a pointer into the source's
  evictable cache or released mapping is not sufficient.
- `core/src/moe/router_hook.cpp::c_eval` is the existing top-k barrier. It must
  snapshot hit slots and forward only original-ID misses to
  `ExpertStreamSource::load_layer` while preserving the original route tensor.
  The CPU `mul_mat_id` hook in pinned `ggml-cpu.c` remains the miss readiness
  barrier. The new map used by a GPU remap node must be published for that
  node's exact graph split, because `ggml_backend_sched_compute_splits` in
  pinned `ggml-backend.cpp` copies split inputs before its per-node callback.
- `llm_graph_context::build_moe_ffn` in pinned `llama-graph.cpp` is the shared
  graph construction point for Qwen3.6's split gate/up/down tensors. Its CPU
  branch keeps original expert IDs and weights; a GPU branch uses compact slot
  IDs and the zero dummy. Their expert outputs merge before the existing
  weighted sum. The exact merge order and numerical tolerance need synthetic
  and real-model evidence; a graph shape alone does not prove them.
- Prefill should bypass admission and eviction. Decode may look up READY slots
  after the top-k barrier. The cache is model-scoped and must be released before
  backend shutdown. No upload may outlive the host source or device buffer.

## Why this path

The [llama.cpp hybrid expert-cache draft](https://github.com/ggml-org/llama.cpp/pull/27861)
uses a GPU slot branch, a CPU miss branch, a zero dummy slot, and a weighted merge
without new CUDA matrix kernels. The [persistent pool discussion](https://github.com/ggml-org/llama.cpp/discussions/28248)
documents a real stale-table bug when a view let the remap consume a previous
ubatch's mapping. [BlobPager](https://github.com/ggml-org/llama.cpp/discussions/29234)
demonstrates a single-GPU exact mixed graph but uses fixed pins and a serial host
miss pass in its first version. Its result is design evidence, not a performance
prediction for this smaller GPU or this SSD-backed source.

The alternative of dispatching GPU work from inside the CPU `mul_mat_id` kernel
would keep the graph simpler, but would require per-projection cross-device
coordination and output transfer at every MoE layer. A graph-level branch allows
one weighted merge per layer and offers a path to concurrent CPU/GPU work. The
first graph version may serialize branches for correctness; profiling must then
show whether split synchronization prevents useful overlap.

## Safety and policy

- Initial fixed budgets are 512, 1024, 1536, and 2048 MiB. The sampled
  cache-off run peaked at 2273 MiB dedicated GPU use on a 6144 MiB device.
  Allocation must check current free dedicated memory and reserve at least
  768 MiB after persistent and graph allocations; measured shared GPU memory
  must remain zero for the inference process.
- Decode uses LRU as the reference policy. Prefill runs through the existing
  path without inserting into or evicting the decode pool. Whether prefill reads
  existing GPU hits is deferred until correctness and traces justify it.
- Pool bytes are copied from the stream-managed exact slices while their source
  lifetime is held, never from a released model-file mapping. The CPU path must
  not read unfilled hit slices. A failed upload or unsupported graph must fail
  explicitly instead of silently changing the requested mode.
- The extension must be inert at cache budget zero, preserve the pinned
  expert-ready hook, and leave Android and CPU-only builds unchanged.

## First falsification tests

1. Allocate 512 MiB without executing from it. Verify the size and VRAM rail,
   cache-off output equivalence, cleanup, and explicit oversized-budget failure.
2. Use distinct synthetic expert weights. Check exact GPU readback of every
   projection, a dummy slot, eviction/reuse, and a new graph/ubatch after reuse.
3. Force all-miss, one-hit, half-hit, all-hit, and alternating routes. Validate
   per-draw accounting and final numerical output against same-backend controls.
4. Run the real Qwen3.6 gate with strict mapping release and a cache budget.
   Do not accept a coherent-looking answer as proof of remap correctness.

No publication, commit, or submodule update is part of this audit.
