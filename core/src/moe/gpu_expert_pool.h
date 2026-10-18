#pragma once

#include "expert_stream_source.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace bmoe {

// Owns one bounded device allocation. Slot zero in each bound layer is reserved for
// the future zero expert; ordinary slots cannot be reused while pinned or uploading.
class GpuExpertPool {
public:
    struct Lease {
        int layer = -1;
        int slot = -1;
        uint64_t generation = 0;
        explicit operator bool() const { return layer >= 0; }
    };

    struct Projection {
        size_t offset = 0;
        size_t expert_bytes = 0;
        const ggml_tensor * shape = nullptr;
    };

    struct Layer {
        Projection proj[MoeRecipe::max_exps];
        size_t bundle_bytes = 0;
        size_t device_map_offset = 0;
        size_t host_map_offset = 0;
        size_t device_state_offset = 0;
        int slot_count = 0; // includes the zero slot
        unsigned required_mask = 0;
    };

    GpuExpertPool() = default;
    ~GpuExpertPool();
    GpuExpertPool(const GpuExpertPool &) = delete;
    GpuExpertPool & operator=(const GpuExpertPool &) = delete;

    bool init(ggml_backend_dev_t device,
              int budget_mb,
              int n_expert,
              const std::vector<LayerExperts> & experts,
              std::string & error,
              int safety_mb = 768,
              bool enable_device_owned = false);
    void reset();
    // Call only after uploads and graph pins have drained; keeps the device allocation.
    void clear_residency();

    // An upload owns a RESERVED lease until publish() or abandon(). A graph owns a
    // READY lease from pin() through unpin(); eviction skips both kinds of lease.
    Lease reserve(int layer, int expert);
    bool upload(Lease lease, int projection, const void * bytes, size_t size);
    bool read_back(Lease lease, int projection, void * bytes, size_t size) const;
    ggml_tensor * make_weight_tensor(ggml_context * ctx, int layer, int projection) const;
    ggml_tensor * make_device_table(ggml_context * ctx, int layer) const;
    ggml_tensor * make_host_table(ggml_context * ctx, int layer) const;
    ggml_tensor * make_device_state(ggml_context * ctx, int layer) const;
    ggml_tensor * make_host_stage_tensor(ggml_context * ctx, int layer, int projection) const;
    int32_t * host_fill_list(int layer) const;
    bool stage_projection(int layer, int fill_index, int projection, const void * bytes, size_t size);
    bool host_stage_available() const { return host_stage_ != nullptr; }
    struct DeviceStats {
        uint64_t routed = 0;
        uint64_t fills = 0;
    };
    DeviceStats device_stats() const;
    bool update_mapping(int layer, const int32_t * slots, int count);
    bool publish(Lease lease);
    bool abandon(Lease lease);
    Lease pin(int layer, int expert);
    bool unpin(Lease lease);

    size_t allocated_bytes() const { return allocated_bytes_; }
    size_t minimum_bytes() const { return minimum_bytes_; }
    size_t planned_bytes() const { return planned_bytes_; }
    size_t capacity() const { return capacity_; }
    int expert_count() const { return n_expert_; }
    uint64_t evictions() const { return evictions_; }
    size_t resident_count() const;
    int available_slots(int layer) const;
    const std::vector<Layer> & layers() const { return layers_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }

private:
    enum class State { Empty, Reserved, Ready };
    struct Slot {
        State state = State::Empty;
        int expert = -1;
        uint64_t generation = 0;
        uint64_t last_use = 0;
        unsigned pins = 0;
        unsigned uploaded_mask = 0;
    };
    bool valid(Lease lease) const;
    ggml_tensor projection_tensor(int layer, int slot, int projection) const;
    void reset_device_state(int layer);

    mutable std::mutex mu_;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_buffer_t host_maps_ = nullptr;
    ggml_backend_buffer_t host_stage_ = nullptr;
    char * stage_host_base_ = nullptr;
    char * stage_device_base_ = nullptr;
    size_t stage_request_stride_ = 0;
    size_t stage_projection_offset_[MoeRecipe::max_exps]{};
    size_t stage_projection_stride_[MoeRecipe::max_exps]{};
    size_t allocated_bytes_ = 0;
    size_t minimum_bytes_ = 0;
    size_t planned_bytes_ = 0;
    size_t capacity_ = 0;
    size_t alignment_ = 1;
    size_t map_stride_ = 0;
    size_t state_stride_ = 0;
    static constexpr int map_batch_max_ = 16;
    std::vector<int32_t> map_upload_;
    int n_expert_ = 0;
    uint64_t next_generation_ = 0;
    uint64_t clock_ = 0;
    uint64_t evictions_ = 0;
    std::vector<Layer> layers_;
    std::vector<std::vector<Slot>> slots_;
};

} // namespace bmoe
