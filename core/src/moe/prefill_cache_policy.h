#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmoe::detail {

struct PrefillBudget {
    size_t headroom = 0;
    std::vector<size_t> quota;
};

// A complete active bank can coexist with the retained pool; each quota is a whole expert.
inline PrefillBudget prefill_budget(const std::vector<size_t> & entry_bytes, int n_expert, size_t budget) {
    PrefillBudget out;
    out.quota.assign(entry_bytes.size(), 0);
    size_t largest = 0, total = 0;
    for (size_t unit : entry_bytes) {
        const size_t bank = unit * (size_t) n_expert;
        largest = std::max(largest, bank);
        total += bank;
    }
    out.headroom = std::min(budget, largest);
    if (total == 0) return out;
    const size_t pool = budget - out.headroom;
    size_t remaining = pool;
    for (size_t il = 0; il < entry_bytes.size(); ++il) {
        const size_t unit = entry_bytes[il];
        if (unit == 0) continue;
        const size_t bank = unit * (size_t) n_expert;
        const size_t share = (size_t) ((long double) pool * bank / total);
        out.quota[il] = std::min({bank, remaining, share / unit * unit});
        remaining -= out.quota[il];
    }
    return out;
}

// ids are token-major. Frequency wins; a later occurrence breaks ties, then expert id.
inline void rank_prefill_ids(const int32_t * ids,
                             int n_ids,
                             int n_expert,
                             std::vector<int> & frequency,
                             std::vector<int> & last,
                             std::vector<int> & ranked) {
    std::fill(frequency.begin(), frequency.end(), 0);
    std::fill(last.begin(), last.end(), -1);
    ranked.clear();
    for (int i = 0; i < n_ids; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert) continue;
        if (frequency[e]++ == 0) ranked.push_back(e);
        last[e] = i;
    }
    std::sort(ranked.begin(), ranked.end(), [&](int a, int b) {
        if (frequency[a] != frequency[b]) return frequency[a] > frequency[b];
        if (last[a] != last[b]) return last[a] > last[b];
        return a < b;
    });
}

} // namespace bmoe::detail
