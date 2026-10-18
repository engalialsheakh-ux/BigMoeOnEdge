#include "bmoe/hybrid_miss.h"

int main() {
    constexpr uint64_t mib = 1024 * 1024;
    const bmoe::HybridMissCosts costs{0.8, 0.3 / double(mib), 0.0, 1};
    if (bmoe::choose_gpu_misses(8, 4, 4, mib, costs) != 4) return 1;
    if (bmoe::choose_gpu_misses(8, 2, 4, mib, costs) != 2) return 2;
    if (bmoe::choose_gpu_misses(8, 4, 1, mib, costs) != 1) return 3;
    if (bmoe::choose_gpu_misses(8, 4, 4, mib, {0.1, 2.0 / double(mib), 0.0, 1}) != 0) return 4;
    if (bmoe::choose_gpu_misses(8, 4, 4, mib, {0.8, 0.0, 0.0, 1}) != 0) return 5;
    if (bmoe::choose_gpu_misses(8, 4, 4, mib, {0.8, 0.3 / double(mib), 0.0, 6}) != 2) return 6;
}
