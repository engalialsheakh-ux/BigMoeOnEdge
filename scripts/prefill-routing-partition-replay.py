#!/usr/bin/env python3
"""Evaluate routing-set-aware prefill partitions on an exact expert-access trace.

The planner sees native routed IDs only after they have been recorded. It prices
candidate tile boundaries using the current replayed RAM residency, then
replans after each tile. This is an offline upper bound: obtaining future IDs
in a running layer would itself require extra routing work.
"""

import argparse
import math
import runpy
from pathlib import Path

MIB = 1024 * 1024


def load_trace(path, base_width):
    access = runpy.run_path(str(Path(__file__).with_name("expert-access-replay.py")))
    n_expert, budget, overlap, units, _, batches = access["read_trace"](path)
    prefill = [batch for batch in batches if batch.phase == 0]
    n_layer = len(units)
    if len(prefill) % n_layer or any(batch.layer != i % n_layer for i, batch in enumerate(prefill)):
        raise ValueError("trace must contain complete layer-ordered prefill microbatches")
    chunks = [prefill[i:i + n_layer] for i in range(0, len(prefill), n_layer)]
    if not chunks or len(chunks[0][0].ids) % base_width:
        raise ValueError("cannot infer expert top-k from the first microbatch")
    top_k = len(chunks[0][0].ids) // base_width
    if top_k == 0:
        raise ValueError("empty routed expert set")
    token_counts = []
    for chunk in chunks:
        count = len(chunk[0].ids)
        if count % top_k or any(len(batch.ids) != count for batch in chunk):
            raise ValueError("inconsistent routed IDs within a microbatch")
        token_counts.append(count // top_k)
    return access, n_expert, budget, overlap, units, chunks, token_counts


def segment(chunks, first, last, batch_type):
    for layer in range(len(chunks[0])):
        ids = tuple(expert for chunk in chunks[first:last] for expert in chunk[layer].ids)
        yield batch_type(0, layer, ids, ())


def replay_partition(access, n_expert, budget, overlap, units, chunks, boundaries):
    replay = access["Replay"]("current", n_expert, budget, overlap, units)
    for first, last in zip(boundaries, boundaries[1:]):
        for batch in segment(chunks, first, last, access["Batch"]):
            replay.batch(batch)
    stats = replay.counts[0]
    return stats["miss_bytes"], stats["rereads"], replay


def adaptive_boundaries(token_counts, max_tile):
    target = min(max_tile, max(512, 1 << (sum(token_counts) - 1).bit_length()))
    out = [0]
    while out[-1] < len(token_counts):
        end = out[-1]
        width = 0
        while end < len(token_counts) and width + token_counts[end] <= target:
            width += token_counts[end]
            end += 1
        out.append(max(end, out[-1] + 1))
    return out


def capacity_boundaries(chunks, token_counts, units, budget, max_tile):
    out = [0]
    while out[-1] < len(chunks):
        sets = [set() for _ in units]
        first = out[-1]
        width = 0
        end = first
        while end < len(chunks) and width + token_counts[end] <= max_tile:
            proposal = [seen | set(chunks[end][layer].ids) for layer, seen in enumerate(sets)]
            required = sum(len(seen) * unit for seen, unit in zip(proposal, units))
            if end > first and required > budget:
                break
            sets = proposal
            width += token_counts[end]
            end += 1
        out.append(max(end, first + 1))
    return out


def planned_boundaries(access, n_expert, budget, overlap, units, chunks, token_counts,
                       max_tile, embedding, activation_limit, ssd_mib_s, tile_ms):
    replay = access["Replay"]("current", n_expert, budget, overlap, units)
    out = [0]
    while out[-1] < len(chunks):
        first = out[-1]
        resident = [set() for _ in units]
        for layer, expert in replay.lru:
            resident[layer].add(expert)
        horizon = first
        width = 0
        while horizon < len(chunks) and width + token_counts[horizon] <= max_tile:
            width += token_counts[horizon]
            horizon += 1
        if horizon == first:
            raise ValueError("one recorded microbatch exceeds the maximum tile")
        # C(s,e) uses the currently resident RAM experts. GPU expert slots are
        # decode-only on this path, so prefill RAM-to-VRAM expert bytes are zero.
        # Compute, per-microbatch launches, and activation readback are fixed by
        # token count; only misses and a small per-tile overhead affect boundaries.
        dp = [math.inf] * (horizon + 1)
        choice = [0] * (horizon + 1)
        dp[horizon] = 0.0
        for start in range(horizon - 1, first - 1, -1):
            unions = [set() for _ in units]
            tokens = 0
            for end in range(start + 1, horizon + 1):
                tokens += token_counts[end - 1]
                if tokens * embedding * 4 > activation_limit or tokens > max_tile:
                    break
                for layer in range(len(units)):
                    unions[layer].update(chunks[end - 1][layer].ids)
                missing = sum(len(demand - have) * unit for demand, have, unit in zip(unions, resident, units))
                cost = missing / MIB / ssd_mib_s + tile_ms / 1000.0 + dp[end]
                if cost < dp[start]:
                    dp[start] = cost
                    choice[start] = end
        next_end = choice[first]
        if next_end <= first:
            raise ValueError("activation limit does not fit one microbatch")
        for batch in segment(chunks, first, next_end, access["Batch"]):
            replay.batch(batch)
        out.append(next_end)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--base-width", type=int, default=64)
    parser.add_argument("--max-tile", type=int, default=8192)
    parser.add_argument("--embedding", type=int, default=2048)
    parser.add_argument("--activation-limit-mib", type=float, default=64)
    parser.add_argument("--ssd-mib-s", type=float, default=1000)
    parser.add_argument("--tile-ms", type=float, default=1)
    args = parser.parse_args()
    if min(args.base_width, args.max_tile, args.embedding, args.activation_limit_mib, args.ssd_mib_s) <= 0:
        parser.error("width, embedding, activation and throughput must be positive")
    access, n_expert, budget, overlap, units, chunks, token_counts = load_trace(args.trace, args.base_width)
    policies = {
        "adaptive": adaptive_boundaries(token_counts, args.max_tile),
        "capacity": capacity_boundaries(chunks, token_counts, units, budget, args.max_tile),
        "cost_optimal": planned_boundaries(access, n_expert, budget, overlap, units, chunks, token_counts,
                                           args.max_tile, args.embedding, args.activation_limit_mib * MIB,
                                           args.ssd_mib_s, args.tile_ms),
    }
    for name, boundaries in policies.items():
        misses, rereads, _ = replay_partition(access, n_expert, budget, overlap, units, chunks, boundaries)
        peak = max(sum(token_counts[first:last]) * args.embedding * 4 / MIB
                   for first, last in zip(boundaries, boundaries[1:]))
        print(f"{name}: tiles={len(boundaries) - 1} boundaries={boundaries} "
              f"miss_mib={misses / MIB:.1f} rereads={rereads} activation_peak_mib={peak:.1f}")


if __name__ == "__main__":
    main()
