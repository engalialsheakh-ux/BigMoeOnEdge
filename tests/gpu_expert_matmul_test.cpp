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
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
            break;
        }
    }
    if (!gpu) {
        llama_backend_free();
        return 77;
    }
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_dev_init(gpu, nullptr),
                                                                               ggml_backend_free);
    if (!backend) {
        llama_backend_free();
        return 1;
    }
    ggml_init_params params{1024 * 1024, nullptr, true};
    auto weights_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    auto graph_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    if (!weights_ctx || !graph_ctx) return 1;

    ggml_tensor * original = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_F32, 4, 3, 3);
    bmoe::LayerExperts layer;
    layer.bound = true;
    layer.proj[0].tensor = original;
    layer.proj[0].nb2 = original->nb[2];
    bmoe::GpuExpertPool pool;
    std::string error;
    if (!pool.init(gpu, 1, 3, {layer}, error)) {
        std::fprintf(stderr, "GPU expert pool init: %s\n", error.c_str());
        return 1;
    }
    std::array<bmoe::GpuExpertPool::Lease, 3> leases;
    for (int expert = 0; expert < 3; ++expert) {
        std::array<float, 12> weights;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col)
                weights[size_t(row * 4 + col)] = float((expert + 1) * (row + 1) * (col + 1));
        leases[size_t(expert)] = pool.reserve(0, expert);
        if (!leases[size_t(expert)] || !pool.upload(leases[size_t(expert)], 0, weights.data(), sizeof(weights)) ||
            !pool.publish(leases[size_t(expert)]))
            return 1;
    }
    ggml_tensor * cached = pool.make_weight_tensor(weights_ctx.get(), 0, 0);
    ggml_tensor * input = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, 4, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_I32, 3, 1);
    ggml_tensor * output = ggml_mul_mat_id(graph_ctx.get(), cached, input, ids);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, output);
    auto graph_buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(graph_ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!cached || !graph_buffer) return 1;

    const std::array<float, 4> x{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<int32_t, 3> selected{leases[2].slot, leases[0].slot, leases[1].slot};
    ggml_backend_tensor_set(input, x.data(), 0, sizeof(x));
    ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
    std::array<float, 9> actual{};
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    for (size_t draw = 0; draw < selected.size(); ++draw) {
        const int expert = std::array<int, 3>{2, 0, 1}[draw];
        for (int row = 0; row < 3; ++row) {
            const float expected = 30.0f * float((expert + 1) * (row + 1));
            const float got = actual[draw * 3 + size_t(row)];
            if (std::fabs(expected - got) > 1e-5f) {
                std::fprintf(stderr, "GPU expert matmul mismatch: draw=%zu row=%d expected=%.6f got=%.6f\n", draw, row,
                             expected, got);
                return 1;
            }
        }
    }
    graph_buffer.reset();
    graph_ctx.reset();
    weights_ctx.reset();
    pool.reset();
    backend.reset();
    llama_backend_free();
    std::puts("GPU expert slot matmul matches distinct reference experts");
    return 0;
}
