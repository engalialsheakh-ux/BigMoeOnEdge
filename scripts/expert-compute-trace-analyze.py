"""Estimate the serial CPU/GPU expert branches from --compute-trace output.

Trace callbacks synchronize after every node and perturb throughput. Reported
branch intervals are diagnostics and their min() is only an overlap upper bound.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.trace.open(encoding="utf-8") as stream:
        reader = csv.DictReader(row for row in stream if not row.startswith("#"))
        for row in reader:
            if int(row["phase"]) == 1:
                groups[int(row["step"])].append(row)
    per_step = defaultdict(lambda: [0, 0, 0])
    cpu_samples = []
    gpu_samples = []
    for step, rows in sorted(groups.items()):
        names = [row["name"] for row in rows]
        for layer in range(40):
            try:
                gate = names.index(f"ffn_moe_gate-{layer}")
                down = names.index(f"ffn_moe_down-{layer}")
                cache = names.index(f"ffn_moe_cache_slots-{layer}")
                merged = names.index(f"ffn_moe_cache_merged-{layer}")
            except ValueError:
                continue
            if not gate < down < cache < merged:
                raise ValueError(f"unexpected branch ordering in step {step}, layer {layer}")
            cpu = sum(int(row["wall_ns"]) for row in rows[gate:down + 1])
            gpu = sum(int(row["wall_ns"]) for row in rows[cache + 1:merged])
            cpu_samples.append(cpu)
            gpu_samples.append(gpu)
            per_step[step][0] += cpu
            per_step[step][1] += gpu
            per_step[step][2] += min(cpu, gpu)
    print(f"steps={len(per_step)} mixed_layers={len(cpu_samples)}")
    print(f"median_cpu_branch_ms={statistics.median(cpu_samples)/1e6:.3f} "
          f"median_gpu_branch_ms={statistics.median(gpu_samples)/1e6:.3f}")
    for step, (cpu, gpu, possible) in sorted(per_step.items()):
        print(f"step={step} cpu_ms={cpu/1e6:.3f} gpu_ms={gpu/1e6:.3f} "
              f"optimistic_overlap_ms={possible/1e6:.3f}")
    if per_step:
        print(f"median_optimistic_overlap_ms_per_token="
              f"{statistics.median(value[2] for value in per_step.values())/1e6:.3f}")


if __name__ == "__main__":
    main()
