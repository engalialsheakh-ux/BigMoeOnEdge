#!/usr/bin/env python3
"""Estimate wider physical prefill batches from a recorded exact access trace.

Groups consecutive graph chunks, preserving each token's routed IDs. This is a
cache and I/O geometry replay, not an inference or wall-time prediction.
"""

import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--base-width", type=int, default=64)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    args = parser.parse_args()

    access = runpy.run_path(str(Path(__file__).with_name("expert-access-replay.py")))
    mib = access["MIB"]
    n_expert, budget, overlap, units, ranges, batches = access["read_trace"](args.trace)
    prefill = [batch for batch in batches if batch.phase == 0]
    n_layer = len(units)
    if len(prefill) % n_layer or any(batch.layer != i % n_layer for i, batch in enumerate(prefill)):
        parser.error("trace must contain complete layer-ordered prefill chunks")
    chunks = [prefill[i:i + n_layer] for i in range(0, len(prefill), n_layer)]
    batch_type = access["Batch"]
    baseline = None
    for width in args.widths:
        if width < args.base_width or width % args.base_width:
            parser.error("widths must be multiples of the recorded base width")
        factor = width // args.base_width
        replay = access["Replay"]("current", n_expert, budget, overlap, units)
        windows = physical = distinct = whole_85 = 0
        for first in range(0, len(chunks), factor):
            group = chunks[first:first + factor]
            for layer in range(n_layer):
                ids = tuple(expert for chunk in group for expert in chunk[layer].ids)
                distinct_count = len(set(ids))
                distinct += distinct_count
                whole_85 += distinct_count * 100 >= 85 * n_expert
                batch = batch_type(0, layer, ids, ())
                actual = replay.batch(batch)
                if factor == 1 and actual != group[0][layer].observed:
                    raise ValueError(f"baseline replay disagrees at chunk {first}, layer {layer}: "
                                     f"{actual} != {group[0][layer].observed}")
                _, merged_bytes, _, merged_reads = access["estimate_io"](
                    replay.last_misses, ranges[layer])
                physical += merged_bytes
                windows += merged_reads
        stats = replay.counts[0]
        if baseline is None:
            baseline = stats["miss_bytes"], stats["rereads"], windows
        print(f"width={width} chunks={(len(chunks) + factor - 1) // factor} "
              f"distinct_mean={distinct / (n_layer * ((len(chunks) + factor - 1) // factor)):.1f} "
              f"whole_85_layers={whole_85} miss_mib={stats['miss_bytes'] / mib:.1f} "
              f"rereads={stats['rereads']} windows={windows} physical_mib={physical / mib:.1f} "
              f"miss_reduction_pct={100 * (baseline[0] - stats['miss_bytes']) / baseline[0]:.2f} "
              f"reread_reduction_pct={100 * (baseline[1] - stats['rereads']) / baseline[1]:.2f}")


if __name__ == "__main__":
    main()
