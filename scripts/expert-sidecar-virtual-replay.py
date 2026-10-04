"""Virtual expert-local sidecar geometry over an exact expert access trace.

This computes an optimistic lower bound on submissions for a derived layout:
one 4 KiB-aligned, contiguous gate/up/down bundle per requested expert.
No sidecar is written and no runtime behavior is changed.
"""

import argparse
import json
from collections import Counter
from importlib.machinery import SourceFileLoader
from itertools import combinations
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--read-p50-ms", help="measured direct-read medians at 512,1024,2048,4096 KiB")
    parser.add_argument("--layout-json", type=Path, help="write deterministic per-layer expert pair ordering")
    args = parser.parse_args()
    replay_module = SourceFileLoader(
        "expert_replay", str(Path(__file__).with_name("expert-access-replay.py"))
    ).load_module()
    n_expert, budget, overlap, units, ranges, batches = replay_module.read_trace(args.trace)
    replay = replay_module.Replay("current", n_expert, budget, overlap, units)
    logical = merged_bytes = merged_reads = sidecar_bytes = sidecar_reads = 0
    misses_by_phase = [0, 0]
    observed_misses = []
    for index, batch in enumerate(batches):
        if replay.batch(batch) != batch.observed:
            raise ValueError(f"cache reference mismatch in batch {index}")
        misses = replay.last_misses
        _, physical, _, windows = replay_module.estimate_io(misses, ranges[batch.layer])
        observed_misses.append((batch.phase, batch.layer, tuple(misses), windows))
        merged_bytes += physical
        merged_reads += windows
        sidecar_reads += len(misses)
        misses_by_phase[batch.phase] += len(misses)
        logical += len(misses) * units[batch.layer]
        # Each bundle is a 4 KiB-aligned sidecar region. Sizes here are actual
        # GGUF projection byte counts; alignment is applied once per bundle.
        aligned = (units[batch.layer] + 4095) // 4096 * 4096
        sidecar_bytes += len(misses) * aligned
    layout_bytes = sum(n_expert * ((size + 4095) // 4096 * 4096) for size in units)
    mib = replay_module.MIB
    print(f"model_layers={len(units)} experts_per_layer={n_expert} "
          f"sidecar_size_gib={layout_bytes/(1024*mib):.3f}")
    print(f"prefill_misses={misses_by_phase[0]} decode_misses={misses_by_phase[1]}")
    print(f"current_coalesced_windows={merged_reads} current_physical_mib={merged_bytes/mib:.1f}")
    print(f"virtual_expert_local_windows={sidecar_reads} virtual_physical_mib={sidecar_bytes/mib:.1f} "
          f"logical_mib={logical/mib:.1f}")
    print(f"optimistic_submission_reduction_pct={100*(merged_reads-sidecar_reads)/merged_reads:.2f} "
          f"physical_byte_reduction_pct={100*(merged_bytes-sidecar_bytes)/merged_bytes:.2f}")
    # The overlap path publishes projection zero before projections one/two.
    # Reading the entire bundle at once delays that first publication. A
    # conservative expert-local layout therefore needs at least two reads.
    two_wave_reads = 2 * sidecar_reads
    print(f"two_wave_preserving_windows={two_wave_reads} "
          f"two_wave_submission_reduction_pct={100*(merged_reads-two_wave_reads)/merged_reads:.2f}")

    prefill_count = sum(phase == 0 for phase, _, _, _ in observed_misses)
    train_count = prefill_count // 2
    frequency = [Counter() for _ in units]
    coactivation = [Counter() for _ in units]
    for _, layer, misses, _ in observed_misses[:train_count]:
        chosen = sorted(set(misses))
        frequency[layer].update(chosen)
        coactivation[layer].update(combinations(chosen, 2))
    profile_order = []
    for layer in range(len(units)):
        remaining = set(range(n_expert))
        first = max(remaining, key=lambda expert: (frequency[layer][expert], -expert))
        order = [first]
        remaining.remove(first)
        while remaining:
            previous = order[-1]
            candidate = max(remaining, key=lambda expert: (
                coactivation[layer][tuple(sorted((previous, expert)))],
                frequency[layer][expert], -expert))
            order.append(candidate)
            remaining.remove(candidate)
        profile_order.append({expert: rank for rank, expert in enumerate(order)})
    if args.layout_json:
        with args.layout_json.open("w", encoding="utf-8") as stream:
            json.dump({"version": 1, "n_layer": len(units), "n_expert": n_expert,
                       "order": [[expert for expert, _ in sorted(ranks.items(), key=lambda pair: pair[1])]
                                 for ranks in profile_order]}, stream, separators=(",", ":"))
            stream.write("\n")

    def packed_windows(misses, ranks, bundle_size):
        slots = sorted(ranks[expert] for expert in misses)
        windows = 0
        i = 0
        while i < len(slots):
            windows += 1
            if (i + 1 < len(slots) and slots[i] % 2 == 0 and slots[i + 1] == slots[i] + 1
                    and 2 * bundle_size <= 4 * mib):
                i += 2
            else:
                i += 1
        return windows

    def current_window_sizes(misses, projections):
        sizes = []
        for base, unit, _ in projections:
            if not unit:
                continue
            begin = end = None
            for expert in sorted(misses):
                offset = base + expert * unit
                if end is not None and offset == end and offset + unit - begin <= 4 * mib:
                    end = offset + unit
                else:
                    if end is not None:
                        sizes.append(((end + 4095) // 4096 - begin // 4096) * 4096)
                    begin, end = offset, offset + unit
            if end is not None:
                sizes.append(((end + 4095) // 4096 - begin // 4096) * 4096)
        return sizes

    def paired_wave_sizes(misses, ranks, projections):
        slots = sorted(ranks[expert] for expert in misses)
        first = next(unit for _, unit, _ in projections if unit)
        rest = sum(unit for _, unit, _ in projections) - first
        sizes = []
        i = 0
        while i < len(slots):
            count = 2 if (i + 1 < len(slots) and slots[i] % 2 == 0
                          and slots[i + 1] == slots[i] + 1) else 1
            sizes.extend((count * first, count * rest))
            i += count
        return sizes

    latency_knots = None
    if args.read_p50_ms:
        values = [float(value) for value in args.read_p50_ms.split(",")]
        if len(values) != 4 or any(value <= 0 for value in values):
            parser.error("--read-p50-ms needs four positive comma-separated values")
        latency_knots = [(512 * 1024, values[0]), (mib, values[1]), (2 * mib, values[2]),
                         (4 * mib, values[3])]

    def read_service_ms(size):
        for (s0, t0), (s1, t1) in zip(latency_knots, latency_knots[1:]):
            if size <= s1:
                return t0 + (size - s0) * (t1 - t0) / (s1 - s0)
        return latency_knots[-1][1]

    for label, records in (("training_prefill", observed_misses[:train_count]),
                           ("heldout_prefill", observed_misses[train_count:prefill_count]),
                           ("decode", observed_misses[prefill_count:])):
        current = natural = trained = 0
        current_service = trained_service = 0.0
        for _, layer, misses, windows in records:
            current += windows
            natural += packed_windows(misses, {expert: expert for expert in range(n_expert)}, units[layer])
            trained += packed_windows(misses, profile_order[layer], units[layer])
            if args.read_p50_ms:
                current_sizes = current_window_sizes(misses, ranges[layer])
                trained_sizes = paired_wave_sizes(misses, profile_order[layer], ranges[layer])
                if len(current_sizes) != windows or len(trained_sizes) != 2 * packed_windows(
                        misses, profile_order[layer], units[layer]):
                    raise ValueError("window geometry mismatch")
                current_service += sum(read_service_ms(size) for size in current_sizes)
                trained_service += sum(read_service_ms(size) for size in trained_sizes)
        print(f"{label}: current_gguf_windows={current} natural_bundle_windows={natural} "
              f"trained_coactivation_bundle_windows={trained} "
              f"paired_two_wave_windows={2*trained} "
              f"pairing_submission_reduction_pct={100*(current-2*trained)/max(1,current):.2f}")
        if args.read_p50_ms:
            print(f"{label}_median_service_proxy: current_lane_ms={current_service/4:.1f} "
                  f"trained_lane_ms={trained_service/4:.1f} "
                  f"change_pct={100*(trained_service-current_service)/current_service:.2f}")


if __name__ == "__main__":
    main()
