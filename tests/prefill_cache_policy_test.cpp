#include "prefill_cache_policy.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    const auto equal = bmoe::detail::prefill_budget({64, 64, 64, 64}, 8, 1536);
    if (equal.headroom != 512 || equal.quota != std::vector<size_t>({256, 256, 256, 256})) return 1;
    const auto clamped = bmoe::detail::prefill_budget({64, 128, 0}, 8, 400);
    if (clamped.headroom != 400 || clamped.quota != std::vector<size_t>({0, 0, 0})) return 2;
    const auto unequal = bmoe::detail::prefill_budget({64, 128, 0}, 8, 2048);
    if (unequal.headroom != 1024 || unequal.quota != std::vector<size_t>({320, 640, 0})) return 3;

    // Three tokens, duplicate IDs, and two frequency-two experts. Expert 3 occurs later.
    const int32_t ids[] = {3, 9, 11, 20, 9, 17, 22, 31, 3, 40, 41, 42};
    std::vector<int> frequency(64), last(64), ranked;
    bmoe::detail::rank_prefill_ids(ids, 12, 64, frequency, last, ranked);
    if (ranked.size() != 10 || ranked[0] != 3 || ranked[1] != 9 || frequency[3] != 2 || frequency[9] != 2 ||
        ranked[2] != 42 || ranked[3] != 41)
        return 4;
    std::puts("prefill quota and frequency/recency selection passed");
    return 0;
}
