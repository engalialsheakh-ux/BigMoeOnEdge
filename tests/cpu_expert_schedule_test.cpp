#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <thread>

struct ReadyProbe {
    std::atomic<bool> first_waiting{false};
    std::atomic<bool> second_progressed{false};
};

static void expert_ready(const ggml_tensor *, int expert, void * user_data) {
    auto & probe = *static_cast<ReadyProbe *>(user_data);
    if (expert == 0) {
        probe.first_waiting.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    } else if (expert == 1 && probe.first_waiting.load(std::memory_order_acquire)) {
        probe.second_progressed.store(true, std::memory_order_release);
    }
}

int main() {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    if (!cpu) return 1;
    ggml_backend_cpu_set_n_threads(cpu.get(), 6);
    ggml_init_params params{1024 * 1024, nullptr, true};
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    if (!ctx) return 2;
    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 4, 8);
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 8, 1);
    ggml_tensor * result = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()), ggml_backend_buffer_free);
    if (!buffer) return 3;
    std::array<float, 4 * 4 * 8> values{};
    std::array<float, 4> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    std::array<int32_t, 8> selected{};
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = float(i % 9 + 1) / 16.0f;
    for (int i = 0; i < 8; ++i)
        selected[size_t(i)] = i;
    ggml_backend_tensor_set(weights, values.data(), 0, sizeof(values));
    ggml_backend_tensor_set(input, input_values.data(), 0, sizeof(input_values));
    ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));

    ReadyProbe probe;
    ggml_cpu_set_expert_ready_hook(&expert_ready, &probe);
    const auto status = ggml_backend_graph_compute(cpu.get(), graph);
    ggml_cpu_set_expert_ready_hook(nullptr, nullptr);
    if (status != GGML_STATUS_SUCCESS || !probe.second_progressed.load(std::memory_order_acquire)) return 4;
    std::array<float, 4 * 8> output{};
    ggml_backend_tensor_get(result, output.data(), 0, sizeof(output));
    for (int expert = 0; expert < 8; ++expert)
        for (int row = 0; row < 4; ++row) {
            float expected = 0.0f;
            for (int col = 0; col < 4; ++col)
                expected += values[size_t(expert * 16 + row * 4 + col)] * input_values[size_t(col)];
            if (std::fabs(output[size_t(expert * 4 + row)] - expected) > 1e-5f) return 5;
        }
}
