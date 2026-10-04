"""Upper-bound replay of RAM-only speculative GPU promotions from real routes.

The speculative candidate is assumed RAM resident and instantly uploaded before
the target layer. This deliberately favors prediction; real queue and RAM checks
can only lower useful hits. Demand admission still uses the current repeat rule.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from importlib.machinery import SourceFileLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budget-mb", type=int, default=1500)
    args = parser.parse_args()
    replay = SourceFileLoader("gpu_layer_replay", str(Path(__file__).with_name("gpu-layer-allocation-replay.py"))).load_module()
    bundles, selected, _ = replay.read_trace(args.trace)
    layers = sorted(bundles)
    steps = sorted({step for _, step in selected})
    assert len(set(bundles.values())) == 1
    bundle = bundles[layers[0]]
    slots = (args.budget_mb * replay.MIB - len(layers) * (bundle + 1024)) // bundle
    capacities = {layer: 1 for layer in layers}
    for i in range(slots - len(layers)):
        capacities[layers[i % len(layers)]] += 1

    def run(kind):
        resident = {layer: {} for layer in layers}
        recent_miss = {}
        previous = {layer: [] for layer in layers}
        frequency = {layer: Counter() for layer in layers}
        hits = misses = promotions = evictions = speculative = useful = 0
        predictions = correct = predicted_miss = 0
        clock = 0
        for token, step in enumerate(steps):
            for layer in layers:
                ids = selected[layer, step]
                chosen = None
                if kind == "previous_token":
                    pool = previous[layer]
                elif kind == "previous_layer":
                    pool = selected[layer - 1, step] if layer > 0 else []
                elif kind == "frequency":
                    pool = [expert for expert, _ in frequency[layer].most_common(8)]
                else:
                    pool = []
                for expert in pool:
                    if expert not in resident[layer]:
                        chosen = expert
                        break
                if chosen is not None:
                    predictions += 1
                    if chosen in ids:
                        correct += 1
                        predicted_miss += 1
                    if len(resident[layer]) >= capacities[layer]:
                        victim = min(resident[layer], key=lambda item: resident[layer][item][0])
                        del resident[layer][victim]
                        evictions += 1
                    clock += 1
                    resident[layer][chosen] = [clock, 0, True]
                    promotions += 1
                    speculative += 1
                missing = []
                for expert in ids:
                    clock += 1
                    if expert in resident[layer]:
                        resident[layer][expert][0] = clock
                        resident[layer][expert][1] += 1
                        useful += resident[layer][expert][2]
                        resident[layer][expert][2] = False
                        hits += 1
                    else:
                        misses += 1
                        missing.append(expert)
                    frequency[layer][expert] += 1
                for expert in missing:
                    prior = recent_miss.get((layer, expert))
                    recent_miss[layer, expert] = token
                    if prior is None or token - prior > 4:
                        continue
                    if expert in resident[layer]:
                        continue
                    if len(resident[layer]) >= capacities[layer]:
                        victim = min(resident[layer], key=lambda item: resident[layer][item][0])
                        del resident[layer][victim]
                        evictions += 1
                    clock += 1
                    resident[layer][expert] = [clock, 0, False]
                    promotions += 1
                    break
                previous[layer] = ids
        print(f"{kind}: hits={hits} misses={misses} promotions={promotions} evictions={evictions} "
              f"predictions={predictions} correct_immediate={correct} precision={correct/max(1,predictions):.3f} "
              f"demand_miss_recall={correct/max(1,misses+correct):.3f} "
              f"useful_speculative={useful} wasted_or_unproven={speculative-useful} "
              f"speculative_upload_mib={speculative*bundle/replay.MIB:.1f} "
              f"total_upload_mib={promotions*bundle/replay.MIB:.1f}")
        return misses, promotions, speculative

    baseline = run("none")
    for candidate in ("previous_token", "previous_layer", "frequency"):
        miss, promotions, speculative = run(candidate)
        print(f"{candidate}_vs_baseline: miss_reduction_pct={100*(baseline[0]-miss)/baseline[0]:.3f} "
              f"additional_promotions={promotions-baseline[1]} speculative_promotions={speculative}")


if __name__ == "__main__":
    main()
