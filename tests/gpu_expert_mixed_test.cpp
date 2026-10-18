#include "gpu_expert_pool.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

int main() {
    llama_backend_init();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto candidate = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = candidate;
            break;
        }
    }
    if (!device) {
        llama_backend_free();
        return 77;
    }

    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    auto gpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_dev_init(device, nullptr),
                                                                           ggml_backend_free);
    if (!cpu || !gpu) return 1;
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    ggml_init_params params{1024 * 1024, nullptr, true};
    auto meta = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    auto cpu_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    auto gpu_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    if (!meta || !cpu_ctx || !gpu_ctx) return 1;

    constexpr int n_expert = 3;
    constexpr int n_used = 2;
    constexpr int n_tokens = 2;
    constexpr int n_embd = 4;
    constexpr int n_ff = 3;
    const std::array<int32_t, n_used * n_tokens> selected{2, 0, 0, 2};
    bmoe::LayerExperts layer;
    layer.bound = true;
    std::array<std::vector<float>, 3> values;
    std::array<ggml_tensor *, 3> cpu_bank{}, gpu_bank{}, cached{};
    for (int p = 0; p < 3; ++p) {
        const int input_dim = p == 2 ? n_ff : n_embd;
        const int output_dim = p == 2 ? n_embd : n_ff;
        layer.proj[p].tensor = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, input_dim, output_dim, n_expert);
        layer.proj[p].nb2 = layer.proj[p].tensor->nb[2];
        cpu_bank[p] = ggml_new_tensor_3d(cpu_ctx.get(), GGML_TYPE_F32, input_dim, output_dim, n_expert);
        gpu_bank[p] = ggml_new_tensor_3d(gpu_ctx.get(), GGML_TYPE_F32, input_dim, output_dim, n_expert);
        values[p].resize(size_t(input_dim * output_dim * n_expert));
        for (int e = 0; e < n_expert; ++e)
            for (int row = 0; row < output_dim; ++row)
                for (int col = 0; col < input_dim; ++col)
                    values[p][size_t(e * input_dim * output_dim + row * input_dim + col)] =
                        0.013f * float(1 + p * 7 + e * 11 + row * 3 + col * 5 + e * row);
    }

    bmoe::GpuExpertPool pool;
    std::string error;
    if (!pool.init(device, 1, n_expert, {layer}, error)) {
        std::fprintf(stderr, "mixed pool init failed: %s\n", error.c_str());
        return 1;
    }
    std::array<bmoe::GpuExpertPool::Lease, n_expert> leases;
    for (int e = 0; e < n_expert; ++e) {
        leases[size_t(e)] = pool.reserve(0, e);
        if (!leases[size_t(e)]) return 1;
        for (int p = 0; p < 3; ++p) {
            const size_t slice = size_t(layer.proj[p].nb2);
            if (!pool.upload(leases[size_t(e)], p, values[p].data() + size_t(e) * slice / sizeof(float), slice))
                return 1;
        }
        if (!pool.publish(leases[size_t(e)])) return 1;
    }
    for (int p = 0; p < 3; ++p)
        cached[p] = pool.make_weight_tensor(gpu_ctx.get(), 0, p);
    ggml_tensor * host_map = pool.make_host_table(cpu_ctx.get(), 0);
    ggml_tensor * device_map = pool.make_device_table(gpu_ctx.get(), 0);
    if (!cached[0] || !cached[1] || !cached[2] || !host_map || !device_map) return 1;

    ggml_tensor * cpu_input = ggml_new_tensor_3d(cpu_ctx.get(), GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * cpu_ids = ggml_new_tensor_2d(cpu_ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
    auto cpu_layer = [&](bool skip) {
        ggml_tensor * up = ggml_mul_mat_id(cpu_ctx.get(), cpu_bank[1], cpu_input, cpu_ids);
        ggml_tensor * gate = ggml_mul_mat_id(cpu_ctx.get(), cpu_bank[0], cpu_input, cpu_ids);
        ggml_tensor * activated = ggml_swiglu_split(cpu_ctx.get(), gate, up);
        ggml_tensor * down = ggml_mul_mat_id(cpu_ctx.get(), cpu_bank[2], activated, cpu_ids);
        if (skip) {
            const int32_t dummy = 0;
            for (ggml_tensor * tensor : {up, gate, down}) {
                tensor->src[3] = host_map;
                std::memcpy(tensor->op_params, &dummy, sizeof(dummy));
            }
        }
        return down;
    };
    ggml_tensor * cpu_reference = cpu_layer(false);
    ggml_tensor * cpu_masked = cpu_layer(true);
    ggml_cgraph * cpu_reference_graph = ggml_new_graph(cpu_ctx.get());
    ggml_cgraph * cpu_masked_graph = ggml_new_graph(cpu_ctx.get());
    ggml_build_forward_expand(cpu_reference_graph, cpu_reference);
    ggml_build_forward_expand(cpu_masked_graph, cpu_masked);
    auto cpu_buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(cpu_ctx.get(), cpu.get()), ggml_backend_buffer_free);
    if (!cpu_buffer) return 1;

    ggml_tensor * gpu_input = ggml_new_tensor_3d(gpu_ctx.get(), GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * gpu_ids = ggml_new_tensor_2d(gpu_ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * repeated_map =
        ggml_view_3d(gpu_ctx.get(), device_map, 1, n_expert, n_tokens, device_map->nb[1], device_map->nb[2], 0);
    ggml_tensor * slot_ids = ggml_get_rows(gpu_ctx.get(), repeated_map, gpu_ids);
    slot_ids = ggml_reshape_2d(gpu_ctx.get(), slot_ids, n_used, n_tokens);
    auto gpu_layer = [&](const std::array<ggml_tensor *, 3> & bank, ggml_tensor * ids) {
        ggml_tensor * up = ggml_mul_mat_id(gpu_ctx.get(), bank[1], gpu_input, ids);
        ggml_tensor * gate = ggml_mul_mat_id(gpu_ctx.get(), bank[0], gpu_input, ids);
        ggml_tensor * activated = ggml_swiglu_split(gpu_ctx.get(), gate, up);
        return ggml_mul_mat_id(gpu_ctx.get(), bank[2], activated, ids);
    };
    ggml_tensor * gpu_reference = gpu_layer(gpu_bank, gpu_ids);
    ggml_tensor * gpu_masked = gpu_layer(cached, slot_ids);
    ggml_cgraph * gpu_reference_graph = ggml_new_graph(gpu_ctx.get());
    ggml_cgraph * gpu_masked_graph = ggml_new_graph(gpu_ctx.get());
    ggml_build_forward_expand(gpu_reference_graph, gpu_reference);
    ggml_build_forward_expand(gpu_masked_graph, gpu_masked);
    auto gpu_buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(gpu_ctx.get(), gpu.get()), ggml_backend_buffer_free);
    if (!gpu_buffer) return 1;

    for (int p = 0; p < 3; ++p) {
        ggml_backend_tensor_set(cpu_bank[p], values[p].data(), 0, values[p].size() * sizeof(float));
        ggml_backend_tensor_set(gpu_bank[p], values[p].data(), 0, values[p].size() * sizeof(float));
    }
    const std::array<float, n_embd * n_tokens> input{0.2f, 0.7f, -0.4f, 1.1f, -0.3f, 0.8f, 0.5f, -0.6f};
    ggml_backend_tensor_set(cpu_input, input.data(), 0, sizeof(input));
    ggml_backend_tensor_set(gpu_input, input.data(), 0, sizeof(input));
    ggml_backend_tensor_set(cpu_ids, selected.data(), 0, sizeof(selected));
    ggml_backend_tensor_set(gpu_ids, selected.data(), 0, sizeof(selected));
    if (ggml_backend_graph_compute(cpu.get(), cpu_reference_graph) != GGML_STATUS_SUCCESS) return 1;
    if (ggml_backend_graph_compute(gpu.get(), gpu_reference_graph) != GGML_STATUS_SUCCESS) return 1;
    std::array<float, n_embd * n_used * n_tokens> cpu_want{}, gpu_want{};
    ggml_backend_tensor_get(cpu_reference, cpu_want.data(), 0, sizeof(cpu_want));
    ggml_backend_tensor_get(gpu_reference, gpu_want.data(), 0, sizeof(gpu_want));
    float cpu_gpu_delta = 0.0f;
    for (size_t i = 0; i < cpu_want.size(); ++i)
        cpu_gpu_delta = std::max(cpu_gpu_delta, std::fabs(cpu_want[i] - gpu_want[i]));
    const float mixed_tolerance = 2.0f * cpu_gpu_delta + 1e-5f;

    auto run = [&](const char * name, bool hit_first, bool hit_second) {
        std::array<int32_t, 3> map{};
        std::vector<bmoe::GpuExpertPool::Lease> pins;
        for (int route = 0; route < n_used; ++route) {
            const bool hit = route == 0 ? hit_first : hit_second;
            if (!hit) continue;
            const int expert = selected[size_t(route)];
            auto lease = pool.pin(0, expert);
            if (!lease) return false;
            pins.push_back(lease);
            map[size_t(expert)] = lease.slot;
        }
        if (!pool.update_mapping(0, map.data(), n_expert)) return false;
        std::array<std::atomic<int>, 3> ready_calls{};
        ggml_cpu_set_expert_ready_hook(
            [](const ggml_tensor *, int expert, void * user) {
                (*static_cast<std::array<std::atomic<int>, 3> *>(user))[size_t(expert)].fetch_add(1);
            },
            &ready_calls);
        const auto cpu_status = ggml_backend_graph_compute(cpu.get(), cpu_masked_graph);
        ggml_cpu_set_expert_ready_hook(nullptr, nullptr);
        if (cpu_status != GGML_STATUS_SUCCESS) return false;
        if (ggml_backend_graph_compute(gpu.get(), gpu_masked_graph) != GGML_STATUS_SUCCESS) return false;
        std::vector<float> cpu_rows(size_t(n_embd * n_used * n_tokens));
        std::vector<float> gpu_rows(size_t(n_embd * n_used * n_tokens));
        ggml_backend_tensor_get(cpu_masked, cpu_rows.data(), 0, cpu_rows.size() * sizeof(float));
        ggml_backend_tensor_get(gpu_masked, gpu_rows.data(), 0, gpu_rows.size() * sizeof(float));
        for (int route = 0; route < n_used * n_tokens; ++route) {
            const int expert = selected[size_t(route)];
            const bool hit = expert == selected[0] ? hit_first : hit_second;
            const int calls = ready_calls[size_t(expert)].load();
            if ((hit && calls != 0) || (!hit && calls == 0)) {
                std::fprintf(stderr, "%s: expert-ready count wrong for %d: %d\n", name, expert, calls);
                return false;
            }
            for (int row = 0; row < n_embd; ++row) {
                const size_t i = size_t(route * n_embd + row);
                if ((hit && std::fabs(cpu_rows[i]) > 1e-6f) || (!hit && std::fabs(gpu_rows[i]) > 1e-6f)) {
                    std::fprintf(stderr, "%s: duplicate or nonzero dummy row %zu\n", name, i);
                    return false;
                }
                const float expected = hit ? gpu_want[i] : cpu_want[i];
                if (!std::isfinite(cpu_rows[i] + gpu_rows[i]) ||
                    std::fabs(cpu_rows[i] + gpu_rows[i] - expected) > mixed_tolerance) {
                    std::fprintf(stderr, "%s: mixed row %zu differs: %.8f vs %.8f (limit %.8f)\n", name, i,
                                 cpu_rows[i] + gpu_rows[i], expected, mixed_tolerance);
                    return false;
                }
            }
        }
        for (auto lease : pins)
            if (!pool.unpin(lease)) return false;
        return true;
    };

    // Fix the CPU-vs-CUDA tolerance from the two full references before mixed cases.
    if (!run("all miss", false, false) || !run("all hit", true, true) || !run("mixed A", true, false) ||
        !run("mixed B", false, true))
        return 1;
    gpu_buffer.reset();
    cpu_buffer.reset();
    gpu_ctx.reset();
    cpu_ctx.reset();
    meta.reset();
    pool.reset();
    gpu.reset();
    cpu.reset();
    llama_backend_free();
    std::printf("mixed CUDA/CPU MoE passed; CPU-vs-CUDA tolerance fixed at %.8f\n", mixed_tolerance);
    return 0;
}
