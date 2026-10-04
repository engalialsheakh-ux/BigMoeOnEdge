#!/usr/bin/env python3
"""Replay a recorded prefill trace in layer-major token tiles.

Each tile runs its original 64-token microbatches in layer order. Routing IDs
are unchanged. This is an I/O schedule simulation, not an executable model.
"""

import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--base-width", type=int, default=64)
    parser.add_argument("--tile-tokens", type=int, nargs="+", default=[64, 256, 512, 1024, 2048, 4096])
    args = parser.parse_args()

    access = runpy.run_path(str(Path(__file__).with_name("expert-access-replay.py")))
    mib = access["MIB"]
    n_expert, budget, overlap, units, ranges, batches = access["read_trace"](args.trace)
    prefill = [batch for batch in batches if batch.phase == 0]
    n_layer = len(units)
    if len(prefill) % n_layer or any(batch.layer != i % n_layer for i, batch in enumerate(prefill)):
        parser.error("trace must contain complete layer-ordered prefill chunks")
    chunks = [prefill[i:i + n_layer] for i in range(0, len(prefill), n_layer)]
    if not args.tile_tokens or args.tile_tokens[0] != args.base_width:
        parser.error("the first tile width must match the recorded microbatch width")

    baseline = None
    for tile in args.tile_tokens:
        if tile < args.base_width or tile % args.base_width:
            parser.error("tile widths must be positive multiples of the recorded microbatch width")
        factor = tile // args.base_width
        replay = access["Replay"]("current", n_expert, budget, overlap, units)
        windows = physical = 0
        for first in range(0, len(chunks), factor):
            group = chunks[first:first + factor]
            for layer in range(n_layer):
                for chunk in group:
                    batch = chunk[layer]
                    actual = replay.batch(batch)
                    if factor == 1 and actual != batch.observed:
                        raise ValueError(f"baseline replay disagrees at chunk {first}, layer {layer}: "
                                         f"{actual} != {batch.observed}")
                    _, merged_bytes, _, merged_reads = access["estimate_io"](
                        replay.last_misses, ranges[layer])
                    physical += merged_bytes
                    windows += merged_reads
        stats = replay.counts[0]
        if baseline is None:
            baseline = stats["miss_bytes"], stats["rereads"], physical, windows
        print(f"tile_tokens={tile} tiles={(len(chunks) + factor - 1) // factor} "
              f"miss_mib={stats['miss_bytes'] / mib:.1f} rereads={stats['rereads']} "
              f"physical_mib={physical / mib:.1f} windows={windows} "
              f"physical_reduction_pct={100 * (baseline[2] - physical) / baseline[2]:.2f} "
              f"window_reduction_pct={100 * (baseline[3] - windows) / baseline[3]:.2f}", flush=True)


if __name__ == "__main__":
    main()
