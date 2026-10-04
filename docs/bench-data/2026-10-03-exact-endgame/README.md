# Exact MoE endgame: offline and short diagnostic gates

This log continues from the local CUDA tree with exact top-8 routing, a 4400 MiB RAM
expert cache, a 1500 MiB GPU expert cache, a 128 MiB GPU safety rail, direct Q4_0
CUDA KV, and opt-in `--io-coalesce`. The long-context reference is 9.389 decode
tok/s and about 15.77 prefill tok/s at 65,507 prompt tokens. The reference run is
not reproduced here. Raw traces and model paths remain in ignored build output.

## What `--overlap` overlaps

The streamer starts four reader workers and publishes each projection's ready
flag after its read. The CPU `MUL_MAT_ID` expert-ready callback blocks only a
thread about to consume a missing slice; other compute threads can continue.
Thus `--overlap` hides some *storage read* time behind CPU expert computation.
It does not schedule the independent CPU and GPU expert branches concurrently.
The GPU cache is decode-only; the host top-k callback maps already-resident GPU
slots and CPU misses, then the graph computes the CPU branch, GPU branch, and
their exact sum. The current ggml scheduler executes backend splits in order
and synchronizes at callback boundaries.

## Per-layer GPU allocation

`scripts/gpu-layer-allocation-replay.py` reads the earlier real-model 256-token
route trace, including actual per-layer bundle bytes. It models the current
repeat-window-4 admission, one promotion per layer and token, and per-layer LRU.
The allocator uses the first 128 tokens to maximize hits minus a modest upload
charge under the *same* 1500 MiB budget; the last 128 tokens are held out.
The model's 1.6875 MiB bundles allow 848 real slots plus one dummy per layer.

| Layout | Full hits | Full misses | Held-out hits | Promotions | Uploaded MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current round-robin 1500 MiB layout | 35,733 | 46,187 | 17,680 | 5,625 | 9,492 |
| Trained nonuniform 1500 MiB layout | 36,034 | 45,886 | 17,530 | 5,568 | 9,396 |

The trained layout loses 150 held-out hits, increasing held-out slow-path
work by 0.64%. At the earlier 2048 MiB trace configuration, this simplified
replay predicted 42,623 hits versus 42,457 observed (0.39% high). It ignores
worker completion time and RAM residency, so smaller predicted differences
would not be trustworthy. Per-layer access counts, reuse medians, working sets,
hits, promotions, and evictions are reproducible with `--layer-csv`. **Decision:**
keep the uniform runtime allocation; the 5% miss or 3% critical-path gate fails.
This 256-token trace is not a surrogate for 64K routing.
The replay also exports a per-layer, per-slot curve (`--curves-csv`): 2,400
candidate points under the 1500 MiB constraint. Its avoided-work columns use
generous short-run upper bounds of 0.38 ms CPU work and 0.62 ms critical-path
time per avoided miss; the held-out hit loss dominates either weighting.

## CPU/GPU expert concurrency

The short eight-token `--compute-trace` diagnostic contained 320 mixed-branch
layer instances. Median per-layer CPU expert work was 3.518 ms and the GPU
branch was 0.355 ms *under per-node synchronization*. The median sum of
`min(CPU,GPU)` across 40 layers was 14.492 ms/token, an optimistic overlap
ceiling in the instrumented run. Its aggregate runtime was about 329 ms/token.
The graph and scheduler show the branches are serialized, so this is a real
opportunity in principle. The diagnostic changes kernel fusion and adds a
barrier per node, so its interval cannot establish a >=5% saving in an
untraced run. A separate scheduler would require a substantial llama.cpp
fork change, synchronized output buffers, and a proven reduction order.
An independent eight-token Nsight Systems CUDA trace captured seven stable
81-graph-launch decode cycles after graph warmup. The alternating short GPU
graph intervals summed to roughly 0.24–0.56 ms/token, while the other graph
intervals were much longer. Their assignment to the GPU expert branch is an
inference from graph order, not a kernel-labelled proof. This cold prompt had
few GPU expert hits, so it does not establish a steady 64K overlap ceiling.
**Decision:** no scheduler change on this evidence. The exact CPU/GPU timing
of a steady, untraced 64K decode remains unmeasured.

## RAM-only cross-layer GPU prediction

`scripts/gpu-route-predict-replay.py` deliberately assumes every predicted
expert is already in RAM and its GPU upload completes before demand. Real
queue pressure and RAM misses can only reduce this upper bound. The replay
uses the same 1500 MiB GPU pool and the same 256-token route trace.

| Predictor | GPU misses | Change vs current | Speculative uploads | Speculative MiB | Immediate precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current demand admission | 46,187 | baseline | 0 | 0 | — |
| Previous token, same layer | 46,097 | -0.20% | 10,005 | 16,883 | 28.4% |
| Previous layer, same token | 53,802 | +16.49% | 9,984 | 16,848 | 2.1% |
| Recent frequency | 46,225 | +0.08% | 2,359 | 3,981 | 13.3% |

**Decision:** do not add predictive GPU uploads. The best upper bound is
immaterial and more than doubles total upload volume in this simplified replay.
That previous-token predictor recalled only 5.8% of demand-time GPU misses.
No speculative SSD reads were made.

## Expert-centric prefill feasibility

The access trace and GGUF metadata show 40 routed layers, 256 experts per
layer, 1.6875 MiB per complete expert, a 432 MiB layer bank, and embedding
width 2048. The 4400 MiB RAM cache can hold ten complete banks. On the
3,401-token trace, current prefill reads 267,845 MiB logically and rereads
149,905 evicted bundles. The distinct selected bundles total only 14,880 MiB:
an ideal full-layer schedule has a **94.44% I/O lower-bound reduction** on
that trace. At 64K the reference's 3,334,981 rereads alone imply about
5,628,000 MiB of repeated logical bundle demand, subject to the trace's
exact per-bundle accounting.
The observed distinct bundles would require 14,901.4 MiB physically in 5,718
aligned windows. A pessimistic full-bank read of every expert at all 40 layers
would use 17,280 MiB logically, 17,296.5 MiB physically and 4,440 aligned
windows for each of 4K, 64K, or 256K, assuming a legal layer-major schedule.

| Prompt tokens | Two FP32 activation buffers | One expert bank | Combined |
| ---: | ---: | ---: | ---: |
| 4,096 | 64 MiB | 432 MiB | 496 MiB |
| 65,536 | 1,024 MiB | 432 MiB | 1,456 MiB |
| 262,144 | 4,096 MiB | 432 MiB | 4,528 MiB |

The simulator verifies every cache event against the trace, then scales its
short-run physical reads and stalls by token count as a **stationary-route
scenario**, not a long-context forecast. It contrasts that current schedule
with one hypothetical read of every expert bank:

| Tokens | Projected current physical MiB | Current read windows | Current rereads | Ideal full-bank physical MiB | Optimistic I/O-only stall saving |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 324,088 | 402,476 | 180,538 | 17,296.5 | 111 s |
| 65,536 | 5,185,413 | 6,439,620 | 2,888,613 | 17,296.5 | 1,874 s |
| 262,144 | 20,741,651 | 25,758,481 | 11,554,453 | 17,296.5 | 7,516 s |

The actual 65,507-token reference had 3,334,981 rereads, about 15% above
the stationary 65,536-token extrapolation. The stall figures assume all
physical-I/O reduction maps proportionally to the short run's unhidden stall;
they are optimistic bounds, not expected wall-clock gains. Any schedule
expressible through the current public API retains the full-model-per-ubatch
order, so its attainable reduction from *layer-major staging* is zero.
Keeping even two FP32 activation buffers and one bank would require 496 MiB
at 4K, 1,456 MiB at 64K, and 4,528 MiB at 256K in host RAM. The 256K case
exceeds the 4,400 MiB expert budget before other live allocations. No extra
GPU bank fits the fixed 1,500 MiB GPU expert budget without displacing the
existing cache, and every layer would require a token-order state barrier.

A layer-major schedule is mathematically plausible for a causal stack if
each attention/recurrent layer processes tokens in original order and carries
its own state forward. It cannot be expressed through the public `llama_decode`
API: each call builds the *whole* model graph for a microbatch, and intermediate
activations/recurrent state cannot be handed between separately scheduled
layers. A tile that still completes all 40 layers before the next tile keeps
the same cache-turnover problem. Implementing a true layer-major schedule
would require a new llama.cpp graph/state API and substantial scheduler
changes, well beyond an isolated fork delta. The 256K activation staging
lower bound also exceeds the RAM cache budget before other live allocations.
**Decision:** retain the exact current prefill schedule. The large theoretical
I/O gain is a research target, not an implementation proven safe here.

## Virtual sidecar layout

`scripts/expert-sidecar-virtual-replay.py` matches every recorded cache
decision and replays physical 4096-byte read windows. An optimistic layout
that packs all three projections of each expert into a single aligned bundle
would use 16.875 GiB of additional disk and change 340,062 current coalesced
read windows into 160,694 (-52.75%). Bytes fall only from 272,445.8 to
271,171.1 MiB (-0.47%). That one-read design delays the first projection
until the *whole* expert arrives, removing early publication under
`--overlap`. A one-expert/two-read layout preserves projection ordering but
only saves 5.49% of windows.

A more credible *static pair* layout places two coactive experts together:
both gate slices first, then both up/down slices. It can issue two reads for
a pair (or two reads for a singleton) and publish gate readiness before the
remaining projections. The first gate of a pair waits for a 1.125 MiB read
instead of a 0.5625 MiB read, which the service-time proxy below includes.
The pairs were trained from the first half of the prefill access trace and
frozen before evaluating the second half.

| Trace section | Current GGUF windows | Paired two-wave windows | Reduction | Four-lane service proxy |
| --- | ---: | ---: | ---: | ---: |
| Training prefill | 167,622 | 102,204 | 39.03% | 33,923.5 → 30,101.7 ms (-11.27%) |
| Held-out prefill | 166,563 | 115,428 | 30.70% | 33,663.0 → 30,534.0 ms (-9.30%) |
| Decode | 5,877 | 3,906 | 33.54% | 945.2 → 810.7 ms (-14.23%) |

The service proxy linearly interpolates the local median unbuffered-read
latencies measured at 512 KiB, 1 MiB, 2 MiB and 4 MiB (0.60, 0.91, 1.71,
3.26 ms) and divides summed service time by four lanes. It is not a runtime
stall prediction: it excludes queue contention, SSD caching, scatter copies,
and the exact time at which the first expert can compute. The 30.70% held-out
submission reduction clears the virtual-layout gate.

An exact, versioned prototype was built from the local model and tested through
an opt-in runtime path. It hashed the complete GGUF and sidecar payload,
validated every projection size and pair offset, fell back on missing/stale
files, and preserved the tiny gate's output and routing. The real 3,401-prompt
token/32-decode diagnostic produced identical text, cache hit/reread counts,
and GPU hit/miss counts with and without the sidecar. The physical-window count
matched the virtual replay exactly: 340,062 GGUF windows versus 221,538
sidecar windows (-34.85%), with 272,445.8 versus 271,171.1 MiB physically
read (-0.47%). This validates the replay geometry but not its latency proxy.

| Short run | Prefill seconds | Prefill tok/s | Decode tok/s | Session load seconds |
| --- | ---: | ---: | ---: | ---: |
| GGUF reference | 196.658 | 17.29 | 6.728 | 8.029 |
| GGUF repeat, same setup | 197.058 | 17.26 | 6.778 | 9.598 |
| Paired sidecar, initial scatter | 225.754 | 15.07 | 6.910 | 110.117 |
| Paired sidecar, direct bounce view | 219.687 | 15.48 | 6.418 | 109.745 |

The direct bounce view removed one memory copy and recovered about six prefill
seconds, but the corrected sidecar remained 11.5% slower in prefill and 5.3%
slower in this 32-token decode sample versus the repeated GGUF reference.
The sidecar's summed I/O lane time was lower (339.2 versus 374.3 seconds in
the repeat), but its unhidden prefill stall was similar (98.2 versus 102.3
seconds). Larger first-gate reads, scatter work, and scheduling effects can
consume a submission-count gain; this diagnostic does not isolate their shares.
Full SHA-256 verification added about 100 seconds to session startup. **Decision:**
reject the runtime sidecar and restore the GGUF reader. Keep only the isolated
builder, byte-layout test, and virtual replay so this negative result remains
reproducible. No model weights or GGUF bytes changed.

## DirectStorage to CUDA

The current GPU promotion source is an *already RAM-resident* expert bundle.
There are zero SSD-to-GPU demand transfers in that path. DirectStorage can
target a D3D12 buffer, and CUDA can import a shareable D3D12 resource, but
that would create a different cache-admission path requiring SSD reads that
the current policy deliberately avoids. The 3,401-token trace transferred
271,171 MiB of CPU-demand expert bytes versus 1,468 MiB of RAM-to-GPU
uploads; replacing the CPU demand reads with a GPU destination would be
wrong.

An [isolated DirectStorage/CUDA probe](../../../tools/dstorage-probe/README.md)
was built with Microsoft's DirectStorage SDK 1.3.0. It reads real expert
offsets from the existing access trace and checks the first returned buffer
at each size byte-for-byte against the GGUF. The Windows baseline uses
unbuffered direct reads followed by CUDA upload; its queued mode keeps eight
overlapped reads in flight, an optimistic comparison to the production
four-lane reader. The DirectStorage side uses a shared D3D12 buffer, imported
into CUDA, and waits on a shared fence before declaring data CUDA-ready.
Three local runs showed these *ranges* of median end-to-end latency (ms):

| Size | Single direct+CUDA | Single DirectStorage | Eight queued direct+CUDA | Eight queued DirectStorage |
| ---: | ---: | ---: | ---: | ---: |
| 512 KiB | 0.59–0.60 | 0.75–0.77 | 3.05–3.66 | 2.49–2.65 |
| 1 MiB | 0.89–0.96 | 0.94–2.50 | 4.79–5.56 | 3.64–3.98 |
| 2 MiB | 1.58–1.71 | 1.81–2.22 | 9.41–12.28 | 6.62–7.65 |
| 4 MiB | 3.10–3.39 | 2.59–2.97 | 17.37–24.39 | 12.75–13.43 |

The repeated eight-request 4 MiB DirectStorage runs reached about 2.0–2.5
GiB/s and cut median batch latency by 26–45% versus the optimistic overlapped
Windows baseline, but p95 varied and one DirectStorage batch reached 37.7 ms.
The latest eight-request 4 MiB sample took about 0.60 ms to enqueue/submit
and 15.3 ms afterward to become CUDA-ready, with 0.98 ms of sampled process
CPU time versus 6.84 ms in the baseline. Windows process CPU time has coarse
resolution for sub-millisecond operations. **Decision:** keep the prototype
isolated and do not integrate it: the production workload currently has no
GPU-destination SSD read to accelerate. The prerequisite is a justified
GPU-destination SSD demand path. See
[Microsoft's destination types](https://learn.microsoft.com/en-us/windows/win32/dstorage/dstorage/ne-dstorage-dstorage_request_destination_type)
and [NVIDIA's D3D12 external-memory rules](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html).

## Partial expert residency

For the model's split SiLU FFN, gate/up output channels can be partitioned
and the matching down-input channels summed, preserving the mathematical
expert function. Quantized down rows have 512 input channels, so a 50%
partition uses complete 256-channel blocks. A half-sized GPU bundle would
roughly double the number of resident expert identities at fixed VRAM, but
each GPU hit saves only *half* of an expert's CPU work, before transfers,
launches, reduction, or rounding differences. Replaying the same 81,920-route
trace at the effective full-bundle capacity of each fraction gives:

| GPU fraction per expert | Effective capacity | GPU hits | Full-expert-equivalent CPU work avoided |
| ---: | ---: | ---: | ---: |
| 100% (current) | 1500 MiB | 35,733 | 35,733 |
| 90% | 1667 MiB | 37,876 | 34,088 |
| 75% | 2000 MiB | 42,039 | 31,529 |
| 50% | 3000 MiB | 50,503 | 25,252 |
| 25% (absolute ceiling) | at most 6000 MiB | at most 81,920 | at most 20,480 |

At 50% residency, the modeled GPU executes half of each hit's FFN channels
and the CPU must still execute the other half; misses remain full CPU work.
The partial GPU upload bytes would shrink in proportion to the fraction,
but any PCIe traffic, extra launches, CPU/GPU synchronization and output
reduction add nonnegative cost. Even assigning zero cost to all of those,
every tested fraction saves less CPU work than the current full-bundle pool;
thus no positive wall-time improvement is predicted by this optimistic
model. The current graph serializes the CPU and GPU branches, making extra
per-expert reductions especially unattractive. **Decision:** do not implement
vertical residency; the >=5% decode-improvement gate fails before overhead.

## CUDA dispatch / graph capture

The Release CUDA build already has `GGML_CUDA_GRAPHS=ON`. In the earlier
untraced 3,401-token/32-decode diagnostic, the backend logged 283 graph
warmups and 131 warmup resets; dynamic graph properties prevent blanket
reuse. The per-node trace has 4,187 nodes per decode step, but it deliberately
synchronizes at every node; roughly 3,700–3,900 nodes per step then appear
under 100 microseconds each. This cannot be used to infer launch overhead in
the untraced fused schedule. An independent Nsight Systems CUDA trace on an
eight-token diagnostic found seven consecutive 81-graph-launch decode cycles
after warmup. Host `cudaGraphLaunch` API time in those cycles was 5.93–6.36
ms/token (one warmer cycle was 8.01 ms). The CUDA profile perturbs time, but
it directly measures this API cost. At least one host-visible routing barrier
per 40 MoE layers must remain; even the optimistic removal of all other 41
graph launches at the same mean API cost saves about 3.1 ms/token, or 2.9%
of the 106.5 ms/token 64K reference. Expanding graph capture would also need
stable buffer addresses across dynamic expert mappings and exact callback
timing. The earlier untraced 3,401-token run had 90.8 ms/token in its
*residual* compute term, which includes scheduler and other work, not launch
cost alone. **Decision:** no graph-capture or persistent-dispatch change;
the >=5% dispatch gate is not supported.

## Integration and benchmark gate

The paired sidecar passed its offline submission gate, then failed the real
short-run performance gate and was removed from the runtime. No new candidate
established a material exact-mode gain. Per the benchmark gate, no new 64K
benchmark was run. The existing 9.389 tok/s / about 15.77 tok/s reference
remains the recommended exact configuration.

The recommended command retains `-c 262144 -t 6 --ubatch 64 --moe-stream
--dense-weights anon --release-mmap --require-release-mmap --cache-mb 4400
--cache-ceil-mb 4400 --io-threads 4 --overlap --io-coalesce --n-gpu-layers -1
--gpu-expert-cache-mb 1500 --gpu-expert-window 4 --gpu-safety-mb 128
--cpu-affinity-mask 0xFFF --flash-attn auto --cache-type-k q4_0
--cache-type-v q4_0 --cuda-fa-telemetry`, with CUDA KV enabled and
`GGML_CUDA_FATTN_MMA_QUANT=auto`. It does not enable any rejected candidate.
There is no new measured 64K delta versus 9.389/15.77; the reference remains
the only full-length result.

The new durable artifacts are the GPU allocation and route-prediction replays,
the prefill feasibility and expert-compute analyzers, the virtual sidecar
replay and isolated builder with its byte-layout test, this report, and the
isolated [DirectStorage probe](../../../tools/dstorage-probe/README.md).
`tests/CMakeLists.txt` registers only the isolated sidecar-builder check.
The working tree also contains earlier uncommitted CUDA/KV/cache work; none
of it was committed or pushed here.
The short compute trace is diagnostic only; its token rate is not compared
to the reference. The copied 64K prompt has been tokenized from the precise
supplied text and yields **65,507 tokens**. The ignored
`build-cuda/expert-endgame-probe/final-command.ps1` contains the baseline
PowerShell command but was not executed.

## Verification and remaining bottleneck

- CUDA Release build succeeded after removing the rejected runtime path.
  `ctest --test-dir build-cuda -C Release --output-on-failure` passed **30/30**
  BigMoeOnEdge tests, including byte-identity, GPU cache, prefill, coalescing,
  and isolated sidecar-builder tests. A first invocation encountered 43 stale
  llama.cpp test registrations whose executables were not built because
  `LLAMA_BUILD_TESTS=OFF`; deleting only the outdated generated
  `build-cuda/third_party/llama.cpp/CTestTestfile.cmake` restored the intended
  30-test suite. The isolated DirectStorage probe built in Release and its
  byte check passed at all four transfer sizes. All new Python sources parsed,
  `git diff --check` passed, and clang-format 18 passed on changed C++ sources.
  No commit or push was made.
- The 64K reference's 15.77 prefill tok/s implies about 63.4 ms per prompt
  token, dominated by repeated expert demand across microbatches; its
  3,334,981 rereads alone represent about 5.37 TiB of logical bundles.
  The short access-traced prefill measured 267,845 MiB of logical misses and
  97.59 s of unhidden stall over 3,401 tokens; those short-run figures must
  not be substituted for the 64K reference.
- The 9.389 decode tok/s reference is about 106.5 ms/token. The short
  diagnostic at a different prompt measured 90.8 ms/token residual compute,
  44.9 ms/token expert stall, and 12.9 ms/token cache management. These terms
  describe that diagnostic's 148.6 ms/token, not the uninstrumented 64K
  breakdown. GPU graph launch overhead has an optimistic 3.1 ms/token
  removable bound; it cannot account for the dominant delay.
- The largest remaining *theoretical* exact-mode headroom is a legal
  layer-major prefill API that preserves attention/recurrent state and bounded
  activations. It is not available through the current public llama.cpp seam
  and would require an upstream architectural change and new exactness proof.
