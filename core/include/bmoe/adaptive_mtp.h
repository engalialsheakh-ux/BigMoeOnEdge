#pragma once

#include <algorithm>
#include <array>

namespace bmoe {

struct MtpVerifyCost {
    double wall_ms = 0.0;
    double stall_ms = 0.0;
    double unique_experts = 0.0;
    double cache_misses = 0.0;
    double cpu_fallback = 0.0;
    double draft_ms = 0.0;
};

// The confidence floor stays in llama.cpp; this controller never adds an unverified target pass.
struct AdaptiveMtpWidth {
    static constexpr int max_width = 8;
    static constexpr int probe_interval = 8;
    struct History {
        int samples = 0;
        int last_observation = 0;
        double compute_ms = 0.0;
        double draft_ms = 0.0;
        double accepted = 0.0;
        double unique_experts = 0.0;
        double cache_misses = 0.0;
        double cpu_fallback = 0.0;
    };

    int width = 2;
    int streak = 0;
    int increases = 0;
    int decreases = 0;
    int max_requested = 0;
    int last_proposed = 0;
    int cost_truncations = 0;
    int cost_probes = 0;
    int observations = 0;
    int last_probe = 0;
    bool explore_cost = false;
    std::array<History, max_width + 1> history{};
    double recent_miss_rate = 0.0;
    double recent_stall_ms_per_miss = 0.0;

    int request(int ceiling, int room, bool explore = false) {
        explore_cost = explore;
        last_proposed = std::min({width, ceiling, room, max_width});
        int chosen = last_proposed;
        while (chosen > 1 && history[chosen].samples >= 2 && history[chosen - 1].samples >= 2) {
            const auto & wide = history[chosen];
            const auto & narrow = history[chosen - 1];
            const auto predicted_cost = [&](const History & h) {
                return std::max(0.001, h.compute_ms + (explore_cost ? h.draft_ms : 0.0) +
                                           h.unique_experts * recent_miss_rate * recent_stall_ms_per_miss);
            };
            if ((1.0 + wide.accepted) * predicted_cost(narrow) > (1.0 + narrow.accepted) * predicted_cost(wide)) break;
            --chosen;
        }
        // A cold-cache observation must not permanently exclude a wider batch after the cache warms.
        // Probe only the adjacent width, within the acceptance cap, and charge it to an ordinary pass.
        if (explore_cost && chosen < last_proposed && observations - last_probe >= probe_interval &&
            observations - history[chosen + 1].last_observation >= probe_interval) {
            ++chosen;
            ++cost_probes;
            last_probe = observations;
        }
        if (chosen < last_proposed) ++cost_truncations;
        max_requested = std::max(max_requested, chosen);
        return chosen;
    }

    void observe(int requested, int drafted, int accepted, int ceiling, MtpVerifyCost cost = {}) {
        ++observations;
        if (drafted > 0 && drafted <= max_width && cost.wall_ms > 0.0) {
            History & h = history[drafted];
            const bool stale = explore_cost && observations - h.last_observation >= probe_interval;
            const auto smooth = [&](double & value, double sample) {
                value = h.samples == 0 || stale ? sample : 0.75 * value + 0.25 * sample;
            };
            smooth(h.compute_ms, std::max(0.0, cost.wall_ms - cost.stall_ms));
            smooth(h.draft_ms, std::max(0.0, cost.draft_ms));
            smooth(h.accepted, double(accepted));
            smooth(h.unique_experts, cost.unique_experts);
            smooth(h.cache_misses, cost.cache_misses);
            smooth(h.cpu_fallback, cost.cpu_fallback);
            ++h.samples;
            h.last_observation = observations;
            if (cost.unique_experts > 0.0)
                recent_miss_rate = 0.75 * recent_miss_rate + 0.25 * cost.cache_misses / cost.unique_experts;
            if (cost.cache_misses > 0.0)
                recent_stall_ms_per_miss = 0.75 * recent_stall_ms_per_miss + 0.25 * cost.stall_ms / cost.cache_misses;
        }
        if (requested <= 0) return;
        if (drafted == 0 || accepted < drafted) {
            streak = 0;
            const int lower = drafted > 0 ? accepted + 1 : width - 1;
            const int next = std::max(1, std::min(width, lower));
            if (next < width) ++decreases;
            width = next;
        } else if (drafted == requested && accepted == drafted && requested == width) {
            if (++streak >= 3) {
                if (width < ceiling) {
                    ++width;
                    ++increases;
                }
                streak = 0;
            }
        } else if (!(explore_cost && drafted == requested && accepted == drafted && requested < width)) {
            // p_min stopped the draft early; an accepted short prefix does not justify widening.
            // Successful cost-limited requests preserve evidence from wider probes without adding to it.
            streak = 0;
        }
        width = std::min(width, ceiling);
    }
};

} // namespace bmoe
