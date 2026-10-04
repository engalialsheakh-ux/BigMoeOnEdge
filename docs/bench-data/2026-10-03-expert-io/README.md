# Exact expert I/O: short-trace decision

This is a diagnostic of one 3,401-token prefill plus 32-token decode on the existing local
Qwen3.6-35B-A3B Q4_K_M Windows/CUDA model. It is **not** the final 64K benchmark, and its
traced tok/s is not a performance result. The run kept context 262144, ubatch 64, six CPU
threads, four direct I/O lanes, a 4400 MiB RAM cache, a 1500 MiB GPU expert cache, a 128 MiB
dedicated VRAM safety rail, full CUDA placement, GPU Q4_0 K/V, direct quantized MMA, and exact
top-8 routing. The access trace is written at the existing host-ID boundary and adds no CUDA
barrier. The trace and raw logs stay under the ignored local build directory.

The replay's **current** policy matched every traced `load_layer` call's lookup, hit, reread,
eviction, and logical read-byte counts. The prefill total was 275,909 lookups, 117,186 hits,
and 149,905 rereads, matching the runtime summary. Per-layer bundle sizes came from the
trace's GGUF projection metadata; all policies below used the same 4400 MiB limit and the
same exact demand sequence. Miss MiB is logical expert data, before 4096-byte direct-I/O
alignment. It is also the simple demand-stall proxy; it is not a wall-time prediction.

| Replay policy | Prefill miss MiB | Prefill rereads | Decode miss MiB | Decode rereads |
| --- | ---: | ---: | ---: | ---: |
| Current layer-aware prefill retention + LRU decode | 267,845.1 | 149,905 | 3,326.1 | 1,832 |
| Segmented probation/protected eviction (75% protected cap) | 267,845.1 | 149,905 | 3,358.1 | 1,851 |
| Frequency-ranked protection + capped segmentation | 267,892.3 | 149,933 | 3,358.1 | 1,851 |
| Layer quotas weighted by recent reuse | 269,520.8 | 150,898 | 3,329.4 | 1,834 |
| Decayed multi-batch expert history | 266,751.6 | 149,257 | 3,523.5 | 1,949 |

The history candidate saves 1,093.5 MiB during prefill but adds 197.4 MiB during decode:
roughly 0.33% net logical bytes saved over this short trace. It would also add persistent
frequency metadata and change the resident set entering decode. The capped segmented candidates
still slightly worsen decode. No candidate offers a material, well-supported benefit here, so the
existing RAM policy remains the default and no GPU admission rule changes. The current GPU
uploader already requires a RAM-resident bundle and does not read SSD solely to warm VRAM;
this short trace does not justify changing its reuse-window rule.

The GGUF keeps each layer's projection as one tensor with experts in adjacent fixed-size
slices. The local model has 256 experts and a 576 KiB slice in a sampled projection; its
sampled tensor start is 3584 bytes past a 4096-byte I/O boundary. Sorting adjacent misses
within each projection allows one exact contiguous read and copy for a group of up to 4 MiB.
The merger never reads a gap to save a submission, never crosses a projection or publication
wave, and publishes every included expert's ready flag only after the read completes.

| Demand-I/O geometry on this trace | Separate slices | Adjacent coalescing |
| --- | ---: | ---: |
| Aligned physical MiB | 272,986.6 | 272,445.8 |
| Read windows | 482,082 | 340,062 |
| Alignment over-read MiB | 1,815.4 | 1,274.6 |

The coalesced geometry **exactly matched** the short run's 272,445.8 physical MiB and
340,062 windows. It merged 142,020 logical ranges, 29.5% fewer read windows, while saving
540.8 MiB of alignment over-read (0.20% of physical traffic). Its mechanism is fewer
unbuffered submissions and less repeated boundary-block I/O; final throughput remains for
the user's untraced 64K run to determine.
