"""Bound full-layer prefill staging and expert I/O from actual access metadata.

The lower I/O bound assumes every layer's selected expert bundle can be retained
for the entire prompt. It is not an executable llama.cpp schedule.
"""

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--embedding", type=int, required=True,
                        help="embedding width from this GGUF's metadata")
    parser.add_argument("--observed-tokens", type=int,
                        help="BMOE_DONE n_prompt for a stationary short-trace extrapolation")
    parser.add_argument("--observed-stall-s", type=float, default=0,
                        help="BMOE_DONE prefill_stall_s for an optimistic I/O-only stall bound")
    args = parser.parse_args()
    replay = SourceFileLoader(
        "expert_replay", str(Path(__file__).with_name("expert-access-replay.py"))
    ).load_module()
    n_expert, budget, overlap, units, ranges, batches = replay.read_trace(args.trace)
    cache = replay.Replay("current", n_expert, budget, overlap, units)
    unique = {layer: set() for layer in range(len(units))}
    prefill_batches = 0
    current_bytes = 0
    current_physical = current_windows = 0
    rereads = 0
    for index, batch in enumerate(batches):
        if cache.batch(batch) != batch.observed:
            raise ValueError(f"cache reference mismatch in batch {index}")
        if batch.phase != 0:
            continue
        prefill_batches += 1
        unique[batch.layer].update(expert for expert in batch.ids if 0 <= expert < n_expert)
        current_bytes += batch.observed[4]
        rereads += batch.observed[2]
        _, physical, _, windows = replay.estimate_io(cache.last_misses, ranges[batch.layer])
        current_physical += physical
        current_windows += windows
    lower = sum(len(ids) * units[layer] for layer, ids in unique.items())
    bank = max(units) * n_expert
    exact_unique_windows = exact_unique_physical = 0
    full_windows = full_physical = 0
    for layer in range(len(units)):
        _, bytes_unique, _, windows_unique = replay.estimate_io(unique[layer], ranges[layer])
        _, bytes_full, _, windows_full = replay.estimate_io(range(n_expert), ranges[layer])
        exact_unique_windows += windows_unique
        exact_unique_physical += bytes_unique
        full_windows += windows_full
        full_physical += bytes_full
    print(f"layers={len(units)} experts_per_layer={n_expert} "
          f"max_layer_bank_mib={bank/replay.MIB:.1f} "
          f"banks_fit_in_4400_mib={4400*replay.MIB//bank}")
    print(f"observed_prefill_batches={prefill_batches} observed_logical_mib={current_bytes/replay.MIB:.1f} "
          f"observed_physical_mib={current_physical/replay.MIB:.1f} observed_read_windows={current_windows} "
          f"observed_rereads={rereads} unique_expert_lower_bound_mib={lower/replay.MIB:.1f} "
          f"ideal_reduction_pct={100*(current_bytes-lower)/current_bytes:.2f}")
    print(f"observed_unique_geometry: physical_mib={exact_unique_physical/replay.MIB:.1f} "
          f"read_windows={exact_unique_windows}")
    print("tokens,two_fp32_activation_buffers_mib,one_layer_bank_mib,activation_plus_bank_mib,"
          "full_bank_logical_mib,full_bank_physical_mib,full_bank_read_windows,"
          "layer_transitions,causal_token_layer_steps")
    for tokens in (4096, 65536, 262144):
        activations = 2 * tokens * args.embedding * 4
        print(f"{tokens},{activations/replay.MIB:.1f},{bank/replay.MIB:.1f},"
              f"{(activations+bank)/replay.MIB:.1f},{sum(units)*n_expert/replay.MIB:.1f},"
              f"{full_physical/replay.MIB:.1f},{full_windows},{len(units)},{tokens*len(units)}")
    if args.observed_tokens:
        if args.observed_tokens <= 0 or args.observed_stall_s < 0:
            parser.error("observed tokens must be positive and observed stall nonnegative")
        print("stationary_short_trace_extrapolation: token_count,current_logical_mib,current_physical_mib,"
              "current_read_windows,current_rereads,ideal_full_bank_physical_mib,"
              "optimistic_io_only_stall_reduction_s,public_api_attainable_stall_reduction_s")
        for tokens in (4096, 65536, 262144):
            scale = tokens / args.observed_tokens
            current_stall = args.observed_stall_s * scale
            physical_ratio = min(1.0, full_physical / (current_physical * scale))
            print(f"{tokens},{current_bytes*scale/replay.MIB:.1f},"
                  f"{current_physical*scale/replay.MIB:.1f},{round(current_windows*scale)},"
                  f"{round(rereads*scale)},{full_physical/replay.MIB:.1f},"
                  f"{current_stall*(1-physical_ratio):.1f},0.0")


if __name__ == "__main__":
    main()
