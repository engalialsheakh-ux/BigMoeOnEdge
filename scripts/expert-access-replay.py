#!/usr/bin/env python3
"""Replay exact host-visible MoE demands from --expert-access-trace.

The trace is written by the streamer after its normal router barrier. It records the
current cache's counters per load_layer call, so the reference replay must match before
an alternative can be trusted. This tool uses only Python's standard library.
"""

import argparse
import collections
import dataclasses
import struct

MIB = 1024 * 1024


def read_exact(file, size):
    value = file.read(size)
    if len(value) != size:
        raise ValueError("truncated expert access trace")
    return value


@dataclasses.dataclass
class Batch:
    phase: int
    layer: int
    ids: tuple
    observed: tuple


def read_trace(path):
    with open(path, "rb") as file:
        if read_exact(file, 4) != b"BMOA":
            raise ValueError("bad expert access trace magic")
        version, n_layer, n_expert, budget, flags = struct.unpack("<IIIQI", read_exact(file, 24))
        if version != 3 or not n_layer or not n_expert or not budget or flags & ~1:
            raise ValueError("unsupported expert access trace header")
        units, ranges = [], []
        for _ in range(n_layer):
            unit, = struct.unpack("<Q", read_exact(file, 8))
            projection = [struct.unpack("<QQi", read_exact(file, 20)) for _ in range(3)]
            if unit != sum(p[1] for p in projection):
                raise ValueError("expert bundle size disagrees with projection ranges")
            units.append(unit)
            ranges.append(projection)
        batches = []
        while marker := file.read(1):
            if marker != b"B":
                raise ValueError("bad expert access trace batch marker")
            phase, layer, count, lookups, hits, rereads, evictions, logical_bytes = struct.unpack(
                "<BHIIIIIQ", read_exact(file, 31))
            if phase > 1 or layer >= n_layer or count > 1_000_000:
                raise ValueError("invalid expert access trace batch")
            ids = struct.unpack("<" + "i" * count, read_exact(file, 4 * count))
            batches.append(Batch(phase, layer, ids, (lookups, hits, rereads, evictions, logical_bytes)))
    return n_expert, budget, bool(flags & 1), units, ranges, batches


class Replay:
    def __init__(self, policy, n_expert, budget, overlap, units):
        self.policy = policy
        self.n_expert = n_expert
        self.budget = budget
        self.overlap = overlap
        self.units = units
        self.lru = collections.OrderedDict()
        self.bytes = 0
        self.retained = set()
        self.probation = set()
        self.hot = collections.OrderedDict()
        self.hot_bytes = 0
        self.retained_bytes = 0
        self.probation_bytes = 0
        self.ever_evicted = set()
        self.frequency = collections.Counter()
        self.history = collections.Counter()
        self.layer_batches = collections.Counter()
        self.phase = None
        self.counts = collections.defaultdict(lambda: collections.Counter())
        self.layer_counts = collections.defaultdict(lambda: collections.Counter())
        self.last_misses = []
        self.quota = self._prefill_quota()
        self.base_quota = self.quota[:]
        self.layer_hit_ema = [0.5] * len(units)
        self.last_prefill_layer = -1

    def _prefill_quota(self):
        banks = [u * self.n_expert for u in self.units]
        total = sum(banks)
        headroom = min(self.budget, max(banks, default=0))
        pool, remaining = self.budget - headroom, self.budget - headroom
        quota = []
        for unit, bank in zip(self.units, banks):
            share = pool * bank // total if total else 0
            q = min(bank, remaining, share // unit * unit) if unit else 0
            quota.append(q)
            remaining -= q
        return quota

    def _drop(self, key, stat):
        self.lru.pop(key)
        cost = self.units[key[0]]
        self.bytes -= cost
        if key in self.retained:
            self.retained.remove(key)
            self.retained_bytes -= cost
        if key in self.probation:
            self.probation.remove(key)
            self.probation_bytes -= cost
        if key in self.hot:
            self.hot.pop(key)
            self.hot_bytes -= cost
        self.ever_evicted.add(key)
        stat["evictions"] += 1

    def _victim(self, exclude):
        if self.phase == 1 and self.policy in ("segmented", "frequency"):
            for key in reversed(self.lru):
                if key not in exclude and key not in self.retained and key in self.probation:
                    return key
        for key in reversed(self.lru):
            if key not in exclude and key not in self.retained:
                return key
        return None

    def _evict_to(self, target, demand, stat):
        while self.bytes > target:
            victim = self._victim(demand)
            if victim is None:
                break
            self._drop(victim, stat)

    def _prefill_plan(self, batch, stat):
        il = batch.layer
        if self.policy == "layer_aware" and il < self.last_prefill_layer:
            weights = [q * (0.5 + hit) for q, hit in zip(self.base_quota, self.layer_hit_ema)]
            total = sum(weights)
            pool = sum(self.base_quota)
            self.quota = [min(int(1.5 * base), int(pool * weight / total) // unit * unit)
                          if unit and total else 0
                          for base, weight, unit in zip(self.base_quota, weights, self.units)]
        self.last_prefill_layer = il
        freq = collections.Counter(e for e in batch.ids if 0 <= e < self.n_expert)
        last = {e: i for i, e in enumerate(batch.ids) if 0 <= e < self.n_expert}
        ranked = sorted(freq, key=lambda e: (-freq[e], -last[e], e))
        for key in [k for k in self.retained if k[0] == il]:
            self.retained.remove(key)
            if key in self.lru:
                self.retained_bytes -= self.units[il]
        n_keep = min(len(ranked), self.quota[il] // self.units[il]) if self.units[il] else 0
        if self.policy == "frequency":
            ranked.sort(key=lambda e: (-(freq[e] + min(self.frequency[il, e], 3)), -last[e], e))
        if self.policy == "history":
            if self.layer_batches[il] and self.layer_batches[il] % 8 == 0:
                for e in range(self.n_expert):
                    self.history[il, e] //= 2
            ranked = sorted(
                set(freq) | {e for e in range(self.n_expert) if self.history[il, e]},
                key=lambda e: (-(2 * freq[e] + self.history[il, e]), -last.get(e, -1), e))
        for e in ranked[:n_keep]:
            key = (il, e)
            self.retained.add(key)
            if key in self.lru:
                self.retained_bytes += self.units[il]
        if self.policy == "history":
            self.layer_batches[il] += 1
            for e, count in freq.items():
                self.history[il, e] += count
        demand = {(il, e) for e in freq}
        incoming = sum(self.units[il] for key in demand if key not in self.lru)
        self._evict_to(max(0, self.budget - incoming), demand, stat)
        return demand

    def batch(self, batch):
        if batch.phase != self.phase:
            self.phase = batch.phase
            self.retained.clear()
            self.retained_bytes = 0
            self.probation.clear()
            self.probation_bytes = 0
            self.hot.clear()
            self.hot_bytes = 0
            if batch.phase == 1 and self.policy in ("segmented", "frequency"):
                self.probation.update(self.lru)
                self.probation_bytes = self.bytes
        il = batch.layer
        stat = collections.Counter()
        self.last_misses = []
        demand = self._prefill_plan(batch, stat) if batch.phase == 0 else {
            (il, e) for e in batch.ids if 0 <= e < self.n_expert}
        seen = set()
        order = sorted(demand) if self.overlap else [(il, e) for e in batch.ids if (il, e) in demand]
        for key in order:
            if key in seen:
                if not self.overlap and key in self.lru:
                    self.lru.move_to_end(key, last=False)
                continue
            seen.add(key)
            stat["lookups"] += 1
            cost = self.units[il]
            if key in self.lru:
                stat["hits"] += 1
                stat["hit_bytes"] += cost
                if not self.overlap:
                    self.lru.move_to_end(key, last=False)
                if batch.phase == 1 and self.policy in ("segmented", "frequency"):
                    if key in self.probation:
                        self.probation.remove(key)
                        self.probation_bytes -= cost
                        self.hot[key] = None
                        self.hot.move_to_end(key, last=False)
                        self.hot_bytes += cost
                        while self.hot_bytes > 3 * self.budget // 4:
                            cold, _ = self.hot.popitem(last=True)
                            cold_cost = self.units[cold[0]]
                            self.hot_bytes -= cold_cost
                            self.probation.add(cold)
                            self.probation_bytes += cold_cost
                    elif key in self.hot:
                        self.hot.move_to_end(key, last=False)
            else:
                self.last_misses.append(key[1])
                if key in self.ever_evicted:
                    stat["rereads"] += 1
                self.lru[key] = None
                self.lru.move_to_end(key, last=False)
                self.bytes += cost
                stat["miss_bytes"] += cost
                if key in self.retained:
                    self.retained_bytes += cost
                else:
                    stat["transient_loads"] += 1
                if batch.phase == 1 and self.policy in ("segmented", "frequency"):
                    self.probation.add(key)
                    self.probation_bytes += cost
            self.frequency[key] += 1
        if self.overlap:
            for e in batch.ids:
                key = (il, e)
                if key in self.lru:
                    self.lru.move_to_end(key, last=False)
        if batch.phase == 0:
            self._evict_to(self.budget, demand, stat)
            if self.policy == "layer_aware" and stat["lookups"]:
                self.layer_hit_ema[il] = 0.8 * self.layer_hit_ema[il] + 0.2 * stat["hits"] / stat["lookups"]
        else:
            self._evict_to(self.budget, demand, stat)
        if self.bytes > self.budget:
            raise ValueError(f"{self.policy} exceeds the fixed cache budget after layer {il}: "
                             f"{self.bytes / MIB:.1f} > {self.budget / MIB:.1f} MiB")
        stat["retained_bytes"] = self.retained_bytes
        stat["probation_bytes"] = (self.probation_bytes if batch.phase == 1 and
                                   self.policy in ("segmented", "frequency") else self.bytes - self.retained_bytes)
        self.counts[batch.phase].update({k: v for k, v in stat.items() if k not in ("retained_bytes", "probation_bytes")})
        self.counts[batch.phase]["retained_bytes_peak"] = max(
            self.counts[batch.phase]["retained_bytes_peak"], stat["retained_bytes"])
        self.counts[batch.phase]["probation_bytes_peak"] = max(
            self.counts[batch.phase]["probation_bytes_peak"], stat["probation_bytes"])
        self.layer_counts[batch.phase, il].update({k: v for k, v in stat.items()
                                                   if k not in ("retained_bytes", "probation_bytes")})
        return tuple(stat[k] for k in ("lookups", "hits", "rereads", "evictions", "miss_bytes"))


def estimate_io(misses, projections, alignment=4096, max_read=4 * MIB):
    """Exact Windows aligned-window geometry for individual and adjacent-slice reads."""
    separate_bytes = merged_bytes = separate_reads = merged_reads = 0
    for base, unit, _file in projections:
        if not unit:
            continue
        group_start = group_end = None
        for expert in sorted(misses):
            offset = base + expert * unit
            begin = offset // alignment * alignment
            end = (offset + unit + alignment - 1) // alignment * alignment
            separate_bytes += end - begin
            separate_reads += 1
            if group_end is not None and offset == group_end and offset + unit - group_start <= max_read:
                group_end = offset + unit
            else:
                if group_end is not None:
                    merged_bytes += ((group_end + alignment - 1) // alignment * alignment) - \
                                    (group_start // alignment * alignment)
                    merged_reads += 1
                group_start, group_end = offset, offset + unit
        if group_end is not None:
            merged_bytes += ((group_end + alignment - 1) // alignment * alignment) - \
                            (group_start // alignment * alignment)
            merged_reads += 1
    return separate_bytes, merged_bytes, separate_reads, merged_reads


def report(label, replay):
    for phase in (0, 1):
        stat = replay.counts[phase]
        lookups = stat["lookups"]
        print(f"{label:12} {'prefill' if phase == 0 else 'decode':7} "
              f"hit={stat['hits'] / lookups * 100 if lookups else 0:5.1f}% "
              f"hit_mib={stat['hit_bytes'] / MIB:10.1f} miss_mib={stat['miss_bytes'] / MIB:10.1f} "
              f"rereads={stat['rereads']:8} evictions={stat['evictions']:8} "
              f"admission_rejects={stat['admission_rejects']:8} "
              f"transient_loads={stat['transient_loads']:8} "
              f"retained_peak_mib={stat['retained_bytes_peak'] / MIB:7.1f} "
              f"probation_peak_mib={stat['probation_bytes_peak'] / MIB:7.1f} "
              f"stall_proxy_mib={stat['miss_bytes'] / MIB:10.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--budget-mb", type=int, default=4400)
    parser.add_argument("--policies", default="current,segmented,frequency,layer_aware,history",
                        help="comma-separated subset of current,segmented,frequency,layer_aware,history")
    args = parser.parse_args()
    n_expert, recorded_budget, overlap, units, _ranges, batches = read_trace(args.trace)
    budget = args.budget_mb * MIB
    if budget != recorded_budget:
        parser.error(f"trace budget is {recorded_budget / MIB:g} MiB, requested {args.budget_mb} MiB")
    requested = args.policies.split(",")
    if any(policy not in ("current", "segmented", "frequency", "layer_aware", "history") for policy in requested):
        parser.error("unknown cache policy")
    policies = ["current"] + [policy for policy in requested if policy != "current"]
    for policy in policies:
        replay = Replay(policy, n_expert, budget, overlap, units)
        io_totals = [0, 0, 0, 0]
        for index, batch in enumerate(batches):
            actual = replay.batch(batch)
            if policy == "current" and actual != batch.observed:
                raise SystemExit(f"reference mismatch at batch {index} layer {batch.layer}: "
                                 f"observed={batch.observed} replay={actual}")
            if policy == "current":
                values = estimate_io(replay.last_misses, _ranges[batch.layer])
                for i, value in enumerate(values):
                    io_totals[i] += value
        report(policy, replay)
        if policy == "current":
            print(f"I/O geometry: separate_mib={io_totals[0] / MIB:.1f} "
                  f"merged_mib={io_totals[1] / MIB:.1f} "
                  f"separate_windows={io_totals[2]} merged_windows={io_totals[3]}")
            print("Per-layer miss MiB (current policy):")
            for (phase, layer), stat in sorted(replay.layer_counts.items()):
                print(f"  phase={phase} layer={layer} miss_mib={stat['miss_bytes'] / MIB:.1f} "
                      f"rereads={stat['rereads']}")


if __name__ == "__main__":
    main()
