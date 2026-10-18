#include "gpu_expert_pool.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {
struct Probe {
    bmoe::GpuExpertPool * pool;
    const float * weights;
    size_t expert_bytes;
    int slots;
    int calls = 0;
    int fills = 0;
};

void stage(void * user_data) {
    auto & probe = *static_cast<Probe *>(user_data);
    const int32_t * list = probe.pool->host_fill_list(0);
    const int count = list[2 * probe.slots];
    ++probe.calls;
    probe.fills += count;
    for (int i = 0; i < count; ++i) {
        const int expert = list[2 * i];
        if (expert < 0 || expert >= 3 || list[2 * i + 1] <= 0 || list[2 * i + 1] >= probe.slots ||
            !probe.pool->stage_projection(
                0, i, 0, reinterpret_cast<const char *>(probe.weights) + size_t(expert) * probe.expert_bytes,
                probe.expert_bytes)) {
            std::fprintf(stderr, "GPU LRU stage received an invalid fill\n");
            std::abort();
        }
    }
}
} // namespace

int main() {
    llama_backend_init();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = ggml_backend_dev_get(i);
            break;
        }
    if (!device) return 77;
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_dev_init(device, nullptr),
                                                                               ggml_backend_free);
    ggml_init_params params{1024 * 1024, nullptr, true};
    auto weights_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    auto graph_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    if (!backend || !weights_ctx || !graph_ctx) return 1;

    constexpr int n_expert = 3;
    constexpr int n_used = 2;
    constexpr int n_in = 4;
    constexpr int n_out = 3;
    std::array<float, n_expert * n_in * n_out> weights{};
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = 0.01f * float(i + 1);
    bmoe::LayerExperts layer;
    layer.bound = true;
    layer.proj[0].tensor = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_F32, n_in, n_out, n_expert);
    layer.proj[0].nb2 = layer.proj[0].tensor->nb[2];
    bmoe::GpuExpertPool pool;
    std::string error;
    if (!pool.init(device, 1, n_expert, {layer}, error, 128, true) || !pool.host_stage_available()) {
        std::fprintf(stderr, "GPU LRU pool: %s\n", error.c_str());
        return 1;
    }
    ggml_tensor * state = pool.make_device_state(weights_ctx.get(), 0);
    ggml_tensor * stage_weights = pool.make_host_stage_tensor(weights_ctx.get(), 0, 0);
    ggml_tensor * cached = pool.make_weight_tensor(weights_ctx.get(), 0, 0);
    if (!state || !stage_weights || !cached) return 1;
    Probe probe{&pool, weights.data(), size_t(layer.proj[0].nb2), pool.layers()[0].slot_count};

    ggml_tensor * bank = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, n_in, n_out, n_expert);
    ggml_tensor * input = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, n_in, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_I32, n_used, 1);
    ggml_tensor * decision = ggml_lru_decide(graph_ctx.get(), ids, state, n_expert, probe.slots);
    ggml_tensor * slots = ggml_view_2d(graph_ctx.get(), decision, n_used, 1, n_used * sizeof(int32_t), 0);
    ggml_tensor * fills = ggml_view_1d(graph_ctx.get(), decision, 2 * probe.slots + 1, n_used * sizeof(int32_t));
    fills = ggml_lru_stage(graph_ctx.get(), fills, pool.host_fill_list(0), stage, &probe);
    ggml_tensor * filled = ggml_lru_fill(graph_ctx.get(), fills, stage_weights, cached);
    ggml_tensor * actual = ggml_mul_mat_id(graph_ctx.get(), filled, input, slots);
    ggml_tensor * reference = ggml_mul_mat_id(graph_ctx.get(), bank, input, ids);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, actual);
    auto graph_buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(graph_ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!graph_buffer) return 1;
    ggml_backend_tensor_set(bank, weights.data(), 0, sizeof(weights));
    const std::array<float, n_in> x{0.1f, 0.2f, 0.3f, 0.4f};
    ggml_backend_tensor_set(input, x.data(), 0, sizeof(x));

    const std::array<std::array<int32_t, n_used>, 3> rounds{{{2, 0}, {2, 0}, {1, 2}}};
    const std::array<int, 3> expected_fills{2, 2, 3};
    for (size_t round = 0; round < rounds.size(); ++round) {
        ggml_backend_tensor_set(ids, rounds[round].data(), 0, sizeof(rounds[round]));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
        std::array<float, n_out * n_used> want{}, got{};
        ggml_backend_tensor_get(reference, want.data(), 0, sizeof(want));
        ggml_backend_tensor_get(actual, got.data(), 0, sizeof(got));
        for (size_t i = 0; i < want.size(); ++i)
            if (!std::isfinite(got[i]) || std::fabs(want[i] - got[i]) > 1e-5f) {
                std::fprintf(stderr, "GPU LRU mismatch round %zu index %zu: %.9f vs %.9f\n", round, i, want[i], got[i]);
                return 1;
            }
        if (probe.calls != int(round + 1) || probe.fills != expected_fills[round]) return 1;
    }
    const auto stats = pool.device_stats();
    if (stats.routed != 6 || stats.fills != 3) return 1;
    std::puts("GPU LRU remapping, host staging, hit/miss fill, and GPU matmul match resident weights");
    return 0;
}
