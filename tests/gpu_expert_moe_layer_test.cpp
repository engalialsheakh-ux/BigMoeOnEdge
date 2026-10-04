#include "gpu_expert_pool.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

int main() {
    llama_backend_init();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = dev;
            break;
        }
    }
    if (!device) {
        llama_backend_free();
        return 77;
    }
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_dev_init(device, nullptr),
                                                                               ggml_backend_free);
    ggml_init_params params{1024 * 1024, nullptr, true};
    auto weights_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    auto graph_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    if (!backend || !weights_ctx || !graph_ctx) return 1;

    constexpr int n_expert = 3;
    constexpr int n_used = 3;
    constexpr int n_embd = 4;
    constexpr int n_ff = 3;
    bmoe::LayerExperts layer;
    layer.bound = true;
    std::array<ggml_tensor *, 3> resident{};
    std::array<std::vector<float>, 3> bank;
    for (int ip = 0; ip < 3; ++ip) {
        const int input_dim = ip == 2 ? n_ff : n_embd;
        const int output_dim = ip == 2 ? n_embd : n_ff;
        layer.proj[ip].tensor = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_F32, input_dim, output_dim, n_expert);
        layer.proj[ip].nb2 = layer.proj[ip].tensor->nb[2];
        resident[ip] = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, input_dim, output_dim, n_expert);
        bank[ip].resize(size_t(input_dim * output_dim * n_expert));
        for (int e = 0; e < n_expert; ++e)
            for (int row = 0; row < output_dim; ++row)
                for (int col = 0; col < input_dim; ++col)
                    bank[ip][size_t(e * input_dim * output_dim + row * input_dim + col)] =
                        0.005f * float((ip + 1) * (e + 1) * (row + 1) * (col + 1));
    }

    bmoe::GpuExpertPool pool;
    std::string error;
    if (!pool.init(device, 1, n_expert, {layer}, error)) {
        std::fprintf(stderr, "GPU MoE pool init failed: %s\n", error.c_str());
        return 1;
    }
    std::array<bmoe::GpuExpertPool::Lease, n_expert> leases;
    for (int expert = 0; expert < n_expert; ++expert) {
        leases[size_t(expert)] = pool.reserve(0, expert);
        if (!leases[size_t(expert)]) return 1;
        for (int ip = 0; ip < 3; ++ip) {
            const size_t floats_per_expert = size_t(layer.proj[ip].nb2) / sizeof(float);
            if (!pool.upload(leases[size_t(expert)], ip, bank[ip].data() + size_t(expert) * floats_per_expert,
                             size_t(layer.proj[ip].nb2)))
                return 1;
        }
        if (!pool.publish(leases[size_t(expert)])) return 1;
    }
    std::array<ggml_tensor *, 3> cached{};
    for (int ip = 0; ip < 3; ++ip)
        cached[ip] = pool.make_weight_tensor(weights_ctx.get(), 0, ip);
    if (!cached[0] || !cached[1] || !cached[2]) return 1;

    ggml_tensor * input = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, n_embd, 1, 1);
    ggml_tensor * original_ids = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_I32, n_used, 1);
    ggml_tensor * slot_ids = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_I32, n_used, 1);
    ggml_tensor * route_weights = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, 1, n_used, 1);
    auto build_layer = [&](const std::array<ggml_tensor *, 3> & w, ggml_tensor * ids) {
        ggml_tensor * up = ggml_mul_mat_id(graph_ctx.get(), w[0], input, ids);
        ggml_tensor * gate = ggml_mul_mat_id(graph_ctx.get(), w[1], input, ids);
        ggml_tensor * activated = ggml_swiglu_split(graph_ctx.get(), gate, up);
        ggml_tensor * down = ggml_mul_mat_id(graph_ctx.get(), w[2], activated, ids);
        ggml_tensor * weighted = ggml_mul(graph_ctx.get(), down, route_weights);
        ggml_tensor * sum = ggml_view_2d(graph_ctx.get(), weighted, n_embd, 1, weighted->nb[2], 0);
        for (int i = 1; i < n_used; ++i) {
            ggml_tensor * draw =
                ggml_view_2d(graph_ctx.get(), weighted, n_embd, 1, weighted->nb[2], size_t(i) * weighted->nb[1]);
            sum = ggml_add(graph_ctx.get(), sum, draw);
        }
        return sum;
    };
    ggml_tensor * reference = build_layer(resident, original_ids);
    ggml_tensor * actual = build_layer(cached, slot_ids);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, actual);
    auto graph_buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(graph_ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!graph_buffer) return 1;

    for (int ip = 0; ip < 3; ++ip)
        ggml_backend_tensor_set(resident[ip], bank[ip].data(), 0, bank[ip].size() * sizeof(float));
    const std::array<float, n_embd> x{0.25f, 0.5f, 0.75f, 1.0f};
    const std::array<int32_t, n_used> experts{2, 0, 1};
    const std::array<int32_t, n_used> slots{leases[2].slot, leases[0].slot, leases[1].slot};
    const std::array<float, n_used> weights{0.2f, 0.3f, 0.5f};
    ggml_backend_tensor_set(input, x.data(), 0, sizeof(x));
    ggml_backend_tensor_set(original_ids, experts.data(), 0, sizeof(experts));
    ggml_backend_tensor_set(slot_ids, slots.data(), 0, sizeof(slots));
    ggml_backend_tensor_set(route_weights, weights.data(), 0, sizeof(weights));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
    std::array<float, n_embd> want{}, got{};
    ggml_backend_tensor_get(reference, want.data(), 0, sizeof(want));
    ggml_backend_tensor_get(actual, got.data(), 0, sizeof(got));
    for (int i = 0; i < n_embd; ++i) {
        if (!std::isfinite(got[size_t(i)]) || std::fabs(want[size_t(i)] - got[size_t(i)]) > 1e-5f) {
            std::fprintf(stderr, "all-hit GPU MoE differs at %d: resident=%.9f cached=%.9f\n", i, want[size_t(i)],
                         got[size_t(i)]);
            return 1;
        }
    }
    graph_buffer.reset();
    graph_ctx.reset();
    weights_ctx.reset();
    pool.reset();
    backend.reset();
    llama_backend_free();
    std::puts("all-hit CUDA MoE gate/up/down and weighted sum match resident GPU graph");
    return 0;
}
