# Layer-tiled prefill transaction

The opt-in Windows CLI path `--prefill-layer-tile 512` schedules eight physical
64-token graphs per logical tile, one model layer at a time. The normal
`llama_decode` path retains its full-layer graph. The path is restricted to
Qwen3.5 MoE with causal inference, full CUDA placement, streamed experts, and
64-token ubatches. A final partial tile and partial microbatch are supported.

## Dependency and state proof

For each tile, layer 0 processes its microbatches in token order, then layer 1
does the same, through the last layer. Each layer sees the preceding layer's
complete tile activation. Attention K/V and recurrent state are per-layer:
their own microbatch order is unchanged. The attention mask uses the same
positions and KV cells as ordinary execution. Residual operations remain inside
their layer graph; norm and the output head run only after the final layer.
Tiles finish in token order before normal one-token decode resumes.

The hybrid memory context owns the same ubatches as ordinary decode. On layer 0,
`apply()` runs exactly once for each microbatch. The KV context records its
resolved `n_kv` while its existing sequence mapping, cell indices and positions
remain in the allocated batch descriptor. The recurrent context records `n_rs`,
`head`, `rs_z` and the source slot index for each state copy. Later layer ranges
select the recorded microbatch and bypass both memory contexts' `apply()`;
graph-input accessors read these small descriptors. KV tensor data, recurrent
tensor data and full model state are never copied for this purpose. The
`layer_start/layer_end` graph parameters are part of the graph reuse identity.
The full `[0,n_layers)` graph path remains the default.

The transaction is fail-stop. Arguments and shape/sequence assumptions are
validated before tiled device work where possible. Once execution starts, any
graph, copy or state error invalidates the context; subsequent decode calls
fail instead of resuming from partially modified KV or recurrent state. The
caller must discard the session. There is no tensor rollback or copy-on-write.

## Activation and graph memory

One reusable host F32 plane holds the hidden state for one 512-token tile.
For the measured model, `512 * 2048 * 4 = 4,194,304` bytes (4 MiB). A layer
reads each microbatch's slice before overwriting it with that layer's output.
No full-prompt or per-layer plane is allocated. The physical graph width stays
64, preserving the measured 70.3 MiB CUDA compute buffer, 1,440 MiB Q4 K/V
allocation and direct quantized FlashAttention dispatch. The 1,500 MiB GPU
expert pool is unchanged.

## Validation and evidence classes

**SIMULATED:** Reordering the recorded 3,401-token expert-access trace into
512-token tiles predicted 62,248.0 MiB physical expert reads, 76.87% below
the width-64 trace's 269,097.7 MiB. This was an I/O replay, not a speed
measurement; see the parent [README](README.md).

**MEASURED:** The tiny Qwen3.5 MoE hybrid gate compares ordinary and tiled
execution for 1, 64, 65, 130 and 520 prompt tokens, including exact serialized
KV-cell metadata, recurrent-cell metadata, sequence position, and post-tile
decode. CPU floating logits differ by at most about `2.75e-8` when the graph
is split; the gate uses a documented `1e-6` floating tolerance. Generated
output from the streamed tiny CUDA session matches exactly. The one short
real-model 3,401-token run also generated text exactly matching the frozen
width-64 reference; its detailed measurements are in the parent README.

**THEORETICAL:** The bounded host plane remains 4 MiB even for a 64K prompt:
the same plane is reused for each tile. This says nothing by itself about
64K throughput or total process memory.
