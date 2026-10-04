#include "gpu_expert_pool.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <array>
#include <cstdint>
#include <cstring>
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
        std::puts("GPU expert pool test skipped: no GPU backend");
        return 77;
    }

    ggml_tensor weight{};
    ggml_tensor second_weight{};
    bmoe::LayerExperts layer;
    layer.bound = true;
    layer.proj[0].tensor = &weight;
    layer.proj[0].nb2 = 400 * 1024;
    layer.proj[1].tensor = &second_weight;
    layer.proj[1].nb2 = 8 * 1024;
    const std::vector<bmoe::LayerExperts> layers{layer};
    std::vector<uint8_t> first_bytes(400 * 1024), second_bytes(8 * 1024), readback(first_bytes.size());
    for (size_t i = 0; i < first_bytes.size(); ++i)
        first_bytes[i] = uint8_t((i * 37 + 11) & 0xff);
    for (size_t i = 0; i < second_bytes.size(); ++i)
        second_bytes[i] = uint8_t((i * 19 + 7) & 0xff);
    for (int repeat = 0; repeat < 3; ++repeat) {
        bmoe::GpuExpertPool pool;
        std::string error;
        if (!pool.init(gpu, 1, 3, layers, error, repeat == 0 ? 128 : 768) || pool.allocated_bytes() != 1024 * 1024 ||
            pool.capacity() != 1) {
            std::fprintf(stderr, "GPU expert pool allocation failed: %s\n", error.c_str());
            llama_backend_free();
            return 1;
        }
        const auto first = pool.reserve(0, 0);
        if (!first || pool.pin(0, 0) || pool.publish(first) ||
            !pool.upload(first, 0, first_bytes.data(), first_bytes.size()) || pool.publish(first) ||
            !pool.upload(first, 1, second_bytes.data(), second_bytes.size()) || !pool.publish(first) ||
            !pool.read_back(first, 0, readback.data(), readback.size()) ||
            std::memcmp(readback.data(), first_bytes.data(), readback.size()) != 0) {
            std::fputs("incomplete upload became visible or exact readback failed\n", stderr);
            llama_backend_free();
            return 1;
        }
        const auto pinned = pool.pin(0, 0);
        const std::array<int32_t, 3> mapped{first.slot, 0, 0};
        ggml_init_params meta_params{64 * 1024, nullptr, true};
        ggml_context * meta = ggml_init(meta_params);
        ggml_tensor * device_map = pool.make_device_table(meta, 0);
        ggml_tensor * host_map = pool.make_host_table(meta, 0);
        std::array<int32_t, 3> got{};
        if (!pinned || !meta || !device_map || !host_map || !pool.update_mapping(0, mapped.data(), 3)) {
            std::fputs("pinned slot mapping failed\n", stderr);
            llama_backend_free();
            return 1;
        }
        ggml_backend_tensor_get(device_map, got.data(), 0, sizeof(got));
        if (got != mapped || std::memcmp(host_map->data, mapped.data(), sizeof(got)) != 0 || pool.reserve(0, 1)) {
            std::fputs("device and host maps diverged or in-flight slot was evicted\n", stderr);
            llama_backend_free();
            return 1;
        }
        ggml_free(meta);
        if (!pool.unpin(pinned)) {
            std::fputs("in-flight slot was evicted or not released\n", stderr);
            llama_backend_free();
            return 1;
        }
        const auto second = pool.reserve(0, 1);
        if (!second || second.slot != first.slot || second.generation == first.generation || pool.unpin(pinned) ||
            pool.pin(0, 1) || !pool.upload(second, 0, first_bytes.data(), first_bytes.size()) ||
            !pool.upload(second, 1, second_bytes.data(), second_bytes.size()) || !pool.publish(second)) {
            std::fputs("slot reuse or generation protection failed\n", stderr);
            llama_backend_free();
            return 1;
        }
        const auto next = pool.pin(0, 1);
        if (!next || !pool.unpin(next)) {
            std::fputs("reused slot was not pinnable\n", stderr);
            llama_backend_free();
            return 1;
        }
        pool.reset();
        if (!pool.init(gpu, 1, 3, layers, error)) {
            std::fprintf(stderr, "GPU expert pool reopen failed: %s\n", error.c_str());
            llama_backend_free();
            return 1;
        }
        const auto fresh = pool.reserve(0, 0);
        if (!fresh || fresh.generation == first.generation || pool.publish(first) || pool.pin(0, 3)) {
            std::fputs("old lease survived a pool reset or invalid expert was admitted\n", stderr);
            llama_backend_free();
            return 1;
        }
    }
    llama_backend_free();
    std::puts("GPU expert slot lifetime and repeated allocation passed");
    return 0;
}
