#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bmoe {

// Rates are observed in this session. A zero rate keeps misses on CPU until calibration exists.
struct HybridMissCosts {
    double cpu_ms_per_expert = 0.0;
    double h2d_ms_per_byte = 0.0;
    double gpu_ms_per_expert = 0.0;
    int cpu_expert_capacity = 1;
};

inline int
choose_gpu_misses(int misses, int host_resident, int available_slots, uint64_t expert_bytes, HybridMissCosts costs) {
    if (misses <= 0 || host_resident <= 0 || available_slots <= 0 || expert_bytes == 0 ||
        costs.cpu_ms_per_expert <= 0.0 || costs.h2d_ms_per_byte <= 0.0 || costs.cpu_expert_capacity <= 0)
        return 0;
    const int maximum = std::min({misses, host_resident, available_slots});
    const auto cpu_ms = [&](int gpu_count) {
        return double(misses - gpu_count) * costs.cpu_ms_per_expert / costs.cpu_expert_capacity;
    };
    int chosen = 0;
    double best_ms = cpu_ms(0);
    for (int gpu_count = 1; gpu_count <= maximum; ++gpu_count) {
        const double gpu_ms =
            gpu_count * (double(expert_bytes) * costs.h2d_ms_per_byte + std::max(0.0, costs.gpu_ms_per_expert));
        const double wall_ms = std::max(cpu_ms(gpu_count), gpu_ms);
        if (wall_ms < best_ms) {
            best_ms = wall_ms;
            chosen = gpu_count;
        }
    }
    return chosen;
}

} // namespace bmoe
