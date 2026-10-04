#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

int main() {
    auto backend =
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) return 1;
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    ggml_init_params params{1024 * 1024, nullptr, true};
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    if (!ctx) return 1;
    ggml_tensor * bank = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 3, 3);
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, 1);
    ggml_tensor * map = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3);
    ggml_tensor * reference = ggml_mul_mat_id(ctx.get(), bank, input, ids);
    ggml_tensor * masked = ggml_mul_mat_id(ctx.get(), bank, input, ids);
    masked->src[3] = map;
    const int32_t dummy = 0;
    std::memcpy(masked->op_params, &dummy, sizeof(dummy));
    ggml_cgraph * graph_reference = ggml_new_graph(ctx.get());
    ggml_cgraph * graph_masked = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph_reference, reference);
    ggml_build_forward_expand(graph_masked, masked);
    auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer) return 1;

    std::array<float, 36> weights;
    for (int expert = 0; expert < 3; ++expert)
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col)
                weights[size_t(expert * 12 + row * 4 + col)] = float((expert + 1) * (row + 1) * (col + 1));
    const std::array<float, 4> x{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<int32_t, 3> selected{2, 0, 1};
    const std::array<int32_t, 3> slots{0, 1, 2}; // expert zero is the only miss
    ggml_backend_tensor_set(bank, weights.data(), 0, sizeof(weights));
    ggml_backend_tensor_set(input, x.data(), 0, sizeof(x));
    ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
    ggml_backend_tensor_set(map, slots.data(), 0, sizeof(slots));
    if (ggml_backend_graph_compute(backend.get(), graph_reference) != GGML_STATUS_SUCCESS) return 1;
    std::array<float, 9> want{}, got{};
    ggml_backend_tensor_get(reference, want.data(), 0, sizeof(want));

    std::array<int, 3> ready_calls{};
    ggml_cpu_set_expert_ready_hook([](const ggml_tensor *, int expert,
                                      void * user) { ++(*static_cast<std::array<int, 3> *>(user))[size_t(expert)]; },
                                   &ready_calls);
    const auto status = ggml_backend_graph_compute(backend.get(), graph_masked);
    ggml_cpu_set_expert_ready_hook(nullptr, nullptr);
    if (status != GGML_STATUS_SUCCESS) return 1;
    ggml_backend_tensor_get(masked, got.data(), 0, sizeof(got));
    for (int draw = 0; draw < 3; ++draw)
        for (int row = 0; row < 3; ++row) {
            const float expected = draw == 1 ? want[size_t(draw * 3 + row)] : 0.0f;
            if (std::fabs(got[size_t(draw * 3 + row)] - expected) > 1e-6f) {
                std::fprintf(stderr, "CPU skip mismatch at draw %d row %d: %.6f vs %.6f\n", draw, row,
                             got[size_t(draw * 3 + row)], expected);
                return 1;
            }
        }
    if (ready_calls != std::array<int, 3>{1, 0, 0}) {
        std::fprintf(stderr, "expert-ready hook reached GPU hits: counts=%d,%d,%d\n", ready_calls[0], ready_calls[1],
                     ready_calls[2]);
        return 1;
    }
    std::puts("CPU expert skip produces exact zero hit rows and waits only for misses");
    return 0;
}
