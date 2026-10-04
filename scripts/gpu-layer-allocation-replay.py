"""Offline, deterministic per-layer GPU expert-cache allocation replay.

Uses a real route trace, one end-of-token promotion per layer, repeat-window
admission, and per-layer LRU. It does not model worker completion time or RAM
residency, so its predictions must be checked against runtime counters.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MIB = 1024 * 1024


def read_trace(path):
    bundles = {}
    selected = defaultdict(list)
    observed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# layer="):
                fields = dict(field.split("=") for field in line[2:].split())
                bundles[int(fields["layer"])] = int(fields["expert_bytes"])
        stream.seek(0)
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        for row in reader:
            if int(row["phase"]) != 1:
                continue
            selected[int(row["layer"]), int(row["step"])].append(int(row["expert"]))
            observed += int(row["gpu_hit"])
    return bundles, selected, observed


def replay_layer(layer, steps, selected, capacity, window):
    resident = {}
    recent_miss = {}
    clock = 0
    hits = promotions = evictions = zero_reuse = 0
    halves = []
    reuse_distances = []
    last_access = {}
    for step_index, step in enumerate(steps):
        missing = []
        for expert in selected[layer, step]:
            clock += 1
            if expert in last_access:
                reuse_distances.append(step_index - last_access[expert])
            last_access[expert] = step_index
            if expert in resident:
                resident[expert][0] = clock
                resident[expert][1] += 1
                hits += 1
            else:
                missing.append(expert)
        for expert in missing:
            prior = recent_miss.get(expert)
            recent_miss[expert] = step_index
            if prior is None or step_index - prior > window:
                continue
            if expert in resident:
                continue
            if len(resident) >= capacity:
                victim = min(resident, key=lambda item: resident[item][0])
                zero_reuse += resident[victim][1] == 0
                del resident[victim]
                evictions += 1
            clock += 1
            resident[expert] = [clock, 0]
            promotions += 1
            break
        if step_index + 1 == len(steps) // 2:
            halves.append((hits, promotions, evictions))
    zero_reuse += sum(value[1] == 0 for value in resident.values())
    return hits, promotions, evictions, zero_reuse, halves[0], reuse_distances, len(resident)


def score(result):
    # An avoided CPU expert is worth one unit; uploading a bundle incurs a
    # conservative 0.1-unit charge. All traced layers use the same bundle size.
    return result[0] - 0.1 * result[1]


def summarize(name, allocation, curves, bundles, steps, selected, observed):
    totals = [0, 0, 0, 0]
    held = [0, 0, 0]
    reuse = []
    working = []
    for layer, capacity in sorted(allocation.items()):
        result = curves[layer][capacity]
        for i in range(4):
            totals[i] += result[i]
        for i in range(3):
            held[i] += result[i] - result[4][i]
        reuse.extend(result[5])
        working.append(result[6])
    used = sum((cap + 1) * bundles[layer] + 1024 for layer, cap in allocation.items())
    upload = sum(curves[layer][cap][1] * bundles[layer] for layer, cap in allocation.items())
    accesses = sum(len(selected[layer, step]) for layer in allocation for step in steps)
    print(f"{name}: gpu_mib={used/MIB:.2f} full_hits={totals[0]} full_misses={accesses-totals[0]} "
          f"held_hits={held[0]} held_promotions={held[1]} held_evictions={held[2]} "
          f"promotions={totals[1]} evictions={totals[2]} zero_reuse={totals[3]} "
          f"upload_mib={upload/MIB:.1f} trace_observed_hits={observed} "
          f"median_reuse_tokens={sorted(reuse)[len(reuse)//2]} resident={sum(working)}")
    return held


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budget-mb", type=int, default=1500)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--layer-csv", type=Path, help="write per-layer baseline and candidate diagnostics")
    parser.add_argument("--curves-csv", type=Path, help="write every evaluated per-layer capacity point")
    parser.add_argument("--cpu-cost-ms", type=float, help="observed upper-bound CPU work per avoided miss")
    parser.add_argument("--critical-cost-ms", type=float, help="observed upper-bound wall time per avoided miss")
    args = parser.parse_args()
    bundles, selected, observed = read_trace(args.trace)
    layers = sorted(bundles)
    steps = sorted({step for _, step in selected})
    assert steps and all((layer, step) in selected for layer in layers for step in steps)
    assert len(set(bundles.values())) == 1, "weighted heterogeneous bundles need a byte knapsack"
    bundle = bundles[layers[0]]
    slots = (args.budget_mb * MIB - sum(bundle + 1024 for _ in layers)) // bundle
    assert slots >= len(layers)
    baseline = {layer: 1 for layer in layers}
    for i in range(slots - len(layers)):
        baseline[layers[i % len(layers)]] += 1
    limit = min(60, slots)
    curves = {layer: {cap: replay_layer(layer, steps, selected, cap, args.window)
                      for cap in range(1, limit + 1)} for layer in layers}
    if args.curves_csv:
        with args.curves_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("layer", "real_slots", "cache_mib", "accesses", "hits", "misses",
                             "heldout_hits", "promotions", "upload_mib", "evictions", "zero_reuse",
                             "resident_bundles", "median_reuse_tokens", "avoided_cpu_work_upper_ms",
                             "avoided_critical_path_upper_ms"))
            for layer in layers:
                access_count = sum(len(selected[layer, step]) for step in steps)
                for cap, result in curves[layer].items():
                    distances = sorted(result[5])
                    writer.writerow((layer, cap, ((cap + 1) * bundle + 1024) / MIB, access_count,
                                     result[0], access_count - result[0], result[0] - result[4][0],
                                     result[1], result[1] * bundle / MIB, result[2], result[3],
                                     result[6], distances[len(distances) // 2] if distances else "",
                                     result[0] * args.cpu_cost_ms if args.cpu_cost_ms is not None else "",
                                     result[0] * args.critical_cost_ms if args.critical_cost_ms is not None else ""))
    # Exact multiple-choice knapsack on the first half: choose one capacity for
    # each layer. The cap count is small, so retain only reachable slot totals.
    dp = {0: (0.0, ())}
    for layer in layers:
        next_dp = {}
        for used, (value, choices) in dp.items():
            for cap, result in curves[layer].items():
                total = used + cap
                if total > slots:
                    continue
                train = result[4]
                candidate = value + score(train)
                if total not in next_dp or candidate > next_dp[total][0]:
                    next_dp[total] = candidate, choices + (cap,)
        dp = next_dp
    _, capacities = max(dp.values(), key=lambda item: item[0])
    optimized = dict(zip(layers, capacities))
    base_held = summarize("uniform", baseline, curves, bundles, steps, selected, observed)
    opt_held = summarize("trained_nonuniform", optimized, curves, bundles, steps, selected, observed)
    held_accesses = sum(len(selected[layer, step]) for layer in layers for step in steps[len(steps)//2:])
    print(f"heldout_hit_delta={opt_held[0]-base_held[0]} "
          f"heldout_miss_reduction_pct={100*(opt_held[0]-base_held[0])/max(1,held_accesses-base_held[0]):.3f}")
    print("allocation=" + ",".join(f"{layer}:{optimized[layer]}" for layer in layers))
    if args.layer_csv:
        with args.layer_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("layer", "accesses", "bundle_bytes", "uniform_slots", "candidate_slots",
                             "uniform_hits", "candidate_hits", "uniform_held_hits", "candidate_held_hits",
                             "uniform_promotions", "candidate_promotions", "uniform_evictions",
                             "candidate_evictions", "reuse_median_tokens", "resident_bundles"))
            for layer in layers:
                a = curves[layer][baseline[layer]]
                b = curves[layer][optimized[layer]]
                distances = sorted(a[5])
                writer.writerow((layer, sum(len(selected[layer, step]) for step in steps), bundles[layer],
                                 baseline[layer], optimized[layer], a[0], b[0], a[0] - a[4][0],
                                 b[0] - b[4][0], a[1], b[1], a[2], b[2],
                                 distances[len(distances) // 2] if distances else "", a[6]))


if __name__ == "__main__":
    main()
