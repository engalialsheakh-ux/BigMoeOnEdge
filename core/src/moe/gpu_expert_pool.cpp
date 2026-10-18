#include "gpu_expert_pool.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace bmoe {
namespace {
constexpr size_t mib = 1024ull * 1024ull;

size_t aligned(size_t n, size_t a) {
    return n + (a - n % a) % a;
}
} // namespace

GpuExpertPool::~GpuExpertPool() {
    reset();
}

bool GpuExpertPool::init(ggml_backend_dev_t device,
                         int budget_mb,
                         int n_expert,
                         const std::vector<LayerExperts> & experts,
                         std::string & error,
                         int safety_mb,
                         bool enable_device_owned) {
    reset();
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU || budget_mb <= 0 || n_expert <= 0) {
        error = "GPU expert pool needs a GPU device, a positive budget, and expert metadata";
        return false;
    }
    if (safety_mb < 128) {
        error = "GPU expert safety rail must be at least 128 MiB";
        return false;
    }
    auto buft = ggml_backend_dev_buffer_type(device);
    if (!buft) {
        error = "GPU device has no device buffer type";
        return false;
    }
    const size_t alignment = std::max<size_t>(128, ggml_backend_buft_get_alignment(buft));
    alignment_ = alignment;
    n_expert_ = n_expert;
    map_stride_ = aligned(size_t(n_expert) * size_t(map_batch_max_) * sizeof(int32_t), alignment);
    state_stride_ = aligned((size_t(n_expert) + 2 * size_t(n_expert + 1) + 3) * sizeof(int32_t), alignment);
    map_upload_.resize(size_t(n_expert) * size_t(map_batch_max_));
    const size_t budget = size_t(budget_mb) * mib;
    layers_.resize(experts.size());
    slots_.resize(experts.size());
    size_t minimum = 0;
    for (size_t il = 0; il < experts.size(); ++il) {
        if (!experts[il].bound) continue;
        for (int ip = 0; ip < MoeRecipe::max_exps; ++ip) {
            const auto & ref = experts[il].proj[ip];
            if (!ref.tensor) continue;
            if (!ref.nb2 || ref.nb2 > std::numeric_limits<size_t>::max() - layers_[il].bundle_bytes) {
                error = "invalid expert projection size";
                reset();
                return false;
            }
            layers_[il].proj[ip].expert_bytes = size_t(ref.nb2);
            layers_[il].proj[ip].shape = ref.tensor;
            layers_[il].required_mask |= 1u << ip;
            layers_[il].bundle_bytes += aligned(size_t(ref.nb2), alignment);
        }
        if (!layers_[il].bundle_bytes) {
            error = "bound expert layer has no projection bytes";
            reset();
            return false;
        }
        // Slot zero is the dummy. A real slot is also required to make this a cache.
        if (minimum > budget || map_stride_ + state_stride_ > budget - minimum ||
            layers_[il].bundle_bytes > (budget - minimum - map_stride_ - state_stride_) / 2) {
            error = "GPU expert budget cannot hold a dummy and one real slot per layer";
            reset();
            return false;
        }
        minimum += map_stride_ + state_stride_ + 2 * layers_[il].bundle_bytes;
        layers_[il].slot_count = 2;
    }
    if (!minimum) {
        error = "GPU expert pool found no bound MoE layers";
        reset();
        return false;
    }
    minimum_bytes_ = minimum;
    size_t used = minimum;
    bool grew = true;
    while (grew) {
        grew = false;
        for (Layer & layer : layers_) {
            if (!layer.slot_count || layer.slot_count > n_expert || layer.bundle_bytes > budget - used) continue;
            ++layer.slot_count;
            used += layer.bundle_bytes;
            grew = true;
        }
    }
    size_t offset = 0;
    for (size_t il = 0; il < layers_.size(); ++il) {
        Layer & layer = layers_[il];
        if (!layer.slot_count) continue;
        slots_[il].resize(size_t(layer.slot_count));
        for (Projection & proj : layer.proj) {
            if (!proj.expert_bytes) continue;
            offset = aligned(offset, alignment);
            proj.offset = offset;
            offset += aligned(proj.expert_bytes, alignment) * size_t(layer.slot_count);
        }
        layer.device_map_offset = aligned(offset, alignment);
        offset = layer.device_map_offset + map_stride_;
        layer.device_state_offset = aligned(offset, alignment);
        offset = layer.device_state_offset + state_stride_;
    }
    if (offset > budget) {
        error = "GPU expert slot layout exceeds its budget";
        reset();
        return false;
    }
    size_t free_before = 0, total = 0;
    ggml_backend_dev_memory(device, &free_before, &total);
    const size_t safety_rail = size_t(safety_mb) * mib;
    auto memory_report = [&](size_t total_bytes, size_t remaining_bytes, bool allocated) {
        char report[256];
        const double remaining_mib = allocated ? double(remaining_bytes) / mib : double(free_before) / mib - budget_mb;
        std::snprintf(report, sizeof(report),
                      "total_dedicated_mib=%.1f free_dedicated_mib=%.1f requested_mib=%d safety_mib=%d %s=%.1f",
                      double(total_bytes) / mib, double(free_before) / mib, budget_mb, safety_mb,
                      allocated ? "remaining_after_allocation_mib" : "projected_remaining_mib", remaining_mib);
        return std::string(report);
    };
    if (!total || free_before < budget || free_before - budget < safety_rail) {
        error = "insufficient dedicated GPU memory for expert cache and safety rail: " + memory_report(total, 0, false);
        reset();
        return false;
    }
    buffer_ = ggml_backend_buft_alloc_buffer(buft, budget);
    if (!buffer_) {
        error = "GPU expert cache allocation failed: " + memory_report(total, 0, false);
        reset();
        return false;
    }
    size_t free_after = 0;
    ggml_backend_dev_memory(device, &free_after, &total);
    if (!total || free_after < safety_rail) {
        error = "GPU expert allocation left less than the configured safety rail: " +
                memory_report(total, free_after, true);
        reset();
        return false;
    }
    std::fprintf(stderr, "bmoe-gpu-cache: %s\n", memory_report(total, free_after, true).c_str());
    allocated_bytes_ = ggml_backend_buffer_get_size(buffer_);
    size_t host_bytes = 0;
    for (Layer & layer : layers_) {
        if (!layer.slot_count) continue;
        layer.host_map_offset = host_bytes;
        host_bytes += map_stride_;
    }
    host_maps_ = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), host_bytes);
    if (!host_maps_) {
        error = "GPU expert host mapping allocation failed";
        reset();
        return false;
    }
    auto * reg = ggml_backend_dev_backend_reg(device);
    using mapped_pointer_fn = void * (*) (ggml_backend_dev_t, void *);
    using mapped_alloc_fn = ggml_backend_buffer_t (*)(size_t);
    auto mapped_pointer = reg ? reinterpret_cast<mapped_pointer_fn>(
                                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mapped_host_pointer"))
                              : nullptr;
    auto mapped_alloc = reg ? reinterpret_cast<mapped_alloc_fn>(
                                  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_alloc_mapped_host_buffer"))
                            : nullptr;
    if (enable_device_owned && mapped_pointer && mapped_alloc) {
        stage_request_stride_ = aligned((2 * size_t(n_expert_ + 1) + 1) * sizeof(int32_t), alignment);
        size_t stage_bytes = stage_request_stride_ * layers_.size();
        size_t max_slots = 0;
        for (const Layer & layer : layers_)
            max_slots = std::max(max_slots, size_t(layer.slot_count));
        for (int p = 0; p < MoeRecipe::max_exps; ++p) {
            size_t max_bytes = 0;
            for (const Layer & layer : layers_)
                if (layer.slot_count) max_bytes = std::max(max_bytes, layer.proj[p].expert_bytes);
            stage_projection_stride_[p] = aligned(max_bytes, alignment);
            stage_projection_offset_[p] = stage_bytes;
            stage_bytes += stage_projection_stride_[p] * max_slots;
        }
        host_stage_ = mapped_alloc(stage_bytes);
        if (host_stage_) {
            stage_host_base_ = static_cast<char *>(ggml_backend_buffer_get_base(host_stage_));
            stage_device_base_ = static_cast<char *>(mapped_pointer(device, stage_host_base_));
        }
        if (!stage_device_base_) {
            if (host_stage_) ggml_backend_buffer_free(host_stage_);
            host_stage_ = nullptr;
            stage_host_base_ = nullptr;
        }
    }
    planned_bytes_ = offset;
    for (const Layer & layer : layers_)
        capacity_ += size_t(std::max(0, layer.slot_count - 1));
    for (size_t il = 0; il < layers_.size(); ++il) {
        if (!layers_[il].slot_count) continue;
        for (int ip = 0; ip < MoeRecipe::max_exps; ++ip) {
            const size_t bytes = layers_[il].proj[ip].expert_bytes;
            if (!bytes) continue;
            ggml_tensor tensor = projection_tensor(int(il), 0, ip);
            ggml_backend_tensor_memset(&tensor, 0, 0, bytes);
        }
        std::vector<int32_t> misses(size_t(n_expert), 0);
        if (!update_mapping(int(il), misses.data(), n_expert)) {
            error = "GPU expert mapping initialization failed";
            reset();
            return false;
        }
        reset_device_state(int(il));
    }
    return true;
}

void GpuExpertPool::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    if (host_stage_) ggml_backend_buffer_free(host_stage_);
    host_stage_ = nullptr;
    stage_host_base_ = stage_device_base_ = nullptr;
    stage_request_stride_ = 0;
    std::fill_n(stage_projection_offset_, MoeRecipe::max_exps, 0);
    std::fill_n(stage_projection_stride_, MoeRecipe::max_exps, 0);
    if (host_maps_) ggml_backend_buffer_free(host_maps_);
    host_maps_ = nullptr;
    if (buffer_) ggml_backend_buffer_free(buffer_);
    buffer_ = nullptr;
    allocated_bytes_ = minimum_bytes_ = planned_bytes_ = capacity_ = 0;
    alignment_ = 1;
    map_stride_ = 0;
    state_stride_ = 0;
    map_upload_.clear();
    n_expert_ = 0;
    clock_ = 0;
    evictions_ = 0;
    layers_.clear();
    slots_.clear();
}

bool GpuExpertPool::valid(Lease lease) const {
    return lease.layer >= 0 && size_t(lease.layer) < slots_.size() && lease.slot > 0 &&
           size_t(lease.slot) < slots_[size_t(lease.layer)].size() &&
           slots_[size_t(lease.layer)][size_t(lease.slot)].generation == lease.generation;
}

ggml_tensor GpuExpertPool::projection_tensor(int layer, int slot, int projection) const {
    const Projection & proj = layers_[size_t(layer)].proj[projection];
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_I8;
    tensor.ne[0] = int64_t(proj.expert_bytes);
    tensor.ne[1] = tensor.ne[2] = tensor.ne[3] = 1;
    tensor.nb[0] = 1;
    tensor.nb[1] = tensor.nb[2] = tensor.nb[3] = proj.expert_bytes;
    tensor.buffer = buffer_;
    tensor.data = static_cast<char *>(ggml_backend_buffer_get_base(buffer_)) + proj.offset +
                  size_t(slot) * aligned(proj.expert_bytes, alignment_);
    return tensor;
}

void GpuExpertPool::reset_device_state(int layer) {
    const Layer & layout = layers_[size_t(layer)];
    std::vector<int32_t> state(size_t(n_expert_) + 2 * size_t(layout.slot_count) + 3, 0);
    std::fill(state.begin(), state.begin() + n_expert_ + layout.slot_count, -1);
    state[size_t(n_expert_) + 2 * size_t(layout.slot_count)] = 1;
    ggml_tensor bytes{};
    bytes.type = GGML_TYPE_I8;
    bytes.ne[0] = int64_t(state.size() * sizeof(int32_t));
    bytes.ne[1] = bytes.ne[2] = bytes.ne[3] = 1;
    bytes.nb[0] = 1;
    bytes.nb[1] = bytes.nb[2] = bytes.nb[3] = size_t(bytes.ne[0]);
    bytes.buffer = buffer_;
    bytes.data = static_cast<char *>(ggml_backend_buffer_get_base(buffer_)) + layout.device_state_offset;
    ggml_backend_tensor_set(&bytes, state.data(), 0, state.size() * sizeof(int32_t));
}

void GpuExpertPool::clear_residency() {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t il = 0; il < slots_.size(); ++il) {
        auto & layer = slots_[il];
        for (size_t i = 1; i < layer.size(); ++i) {
            Slot & slot = layer[i];
            slot.state = State::Empty;
            slot.expert = -1;
            slot.generation = ++next_generation_;
            slot.last_use = 0;
            slot.pins = 0;
            slot.uploaded_mask = 0;
        }
        if (!layer.empty()) reset_device_state(int(il));
    }
}

GpuExpertPool::Lease GpuExpertPool::reserve(int layer, int expert) {
    std::lock_guard<std::mutex> lock(mu_);
    if (layer < 0 || size_t(layer) >= slots_.size() || expert < 0 || expert >= n_expert_) return {};
    auto & slots = slots_[size_t(layer)];
    if (slots.size() < 2) return {};
    for (size_t i = 1; i < slots.size(); ++i)
        if (slots[i].state != State::Empty && slots[i].expert == expert) return {};
    size_t best = 0;
    for (size_t i = 1; i < slots.size(); ++i) {
        if (slots[i].state == State::Empty) {
            best = i;
            break;
        }
        if (slots[i].state == State::Ready && !slots[i].pins && (!best || slots[i].last_use < slots[best].last_use))
            best = i;
    }
    if (!best) return {};
    Slot & slot = slots[best];
    if (slot.state == State::Ready) ++evictions_;
    slot.state = State::Reserved;
    slot.expert = expert;
    slot.generation = ++next_generation_;
    slot.last_use = ++clock_;
    slot.uploaded_mask = 0;
    return {layer, int(best), slot.generation};
}

bool GpuExpertPool::upload(Lease lease, int projection, const void * bytes, size_t size) {
    ggml_tensor tensor{};
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!valid(lease) || projection < 0 || projection >= MoeRecipe::max_exps || !bytes) return false;
        Slot & slot = slots_[size_t(lease.layer)][size_t(lease.slot)];
        const Projection & proj = layers_[size_t(lease.layer)].proj[projection];
        if (slot.state != State::Reserved || !proj.expert_bytes || size != proj.expert_bytes ||
            (slot.uploaded_mask & (1u << projection)))
            return false;
        tensor = projection_tensor(lease.layer, lease.slot, projection);
    }
    ggml_backend_tensor_set(&tensor, bytes, 0, size);
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid(lease) || slots_[size_t(lease.layer)][size_t(lease.slot)].state != State::Reserved) return false;
    Slot & slot = slots_[size_t(lease.layer)][size_t(lease.slot)];
    slot.uploaded_mask |= 1u << projection;
    return true;
}

bool GpuExpertPool::read_back(Lease lease, int projection, void * bytes, size_t size) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid(lease) || projection < 0 || projection >= MoeRecipe::max_exps || !bytes) return false;
    const Slot & slot = slots_[size_t(lease.layer)][size_t(lease.slot)];
    const Projection & proj = layers_[size_t(lease.layer)].proj[projection];
    if (slot.state != State::Ready || !proj.expert_bytes || size != proj.expert_bytes) return false;
    ggml_tensor tensor = projection_tensor(lease.layer, lease.slot, projection);
    ggml_backend_tensor_get(&tensor, bytes, 0, size);
    return true;
}

ggml_tensor * GpuExpertPool::make_weight_tensor(ggml_context * ctx, int layer, int projection) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ctx || !buffer_ || layer < 0 || size_t(layer) >= layers_.size() || projection < 0 ||
        projection >= MoeRecipe::max_exps)
        return nullptr;
    const Layer & layout = layers_[size_t(layer)];
    const Projection & proj = layout.proj[projection];
    if (!layout.slot_count || !proj.shape || !proj.expert_bytes) return nullptr;
    const ggml_tensor * src = proj.shape;
    if (src->ne[0] <= 0 || src->ne[1] <= 0 || src->ne[2] < 1 || src->ne[3] != 1 ||
        size_t(src->nb[2]) != proj.expert_bytes)
        return nullptr;
    ggml_tensor * weight = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], layout.slot_count);
    if (!weight) return nullptr;
    weight->nb[2] = aligned(proj.expert_bytes, alignment_);
    weight->nb[3] = weight->nb[2] * size_t(layout.slot_count);
    weight->buffer = buffer_;
    weight->data = static_cast<char *>(ggml_backend_buffer_get_base(buffer_)) + proj.offset;
    return weight;
}

ggml_tensor * GpuExpertPool::make_device_table(ggml_context * ctx, int layer) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ctx || !buffer_ || layer < 0 || size_t(layer) >= layers_.size() || !layers_[size_t(layer)].slot_count)
        return nullptr;
    ggml_tensor * table = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, n_expert_, map_batch_max_);
    if (!table) return nullptr;
    table->buffer = buffer_;
    table->data = static_cast<char *>(ggml_backend_buffer_get_base(buffer_)) + layers_[size_t(layer)].device_map_offset;
    return table;
}

ggml_tensor * GpuExpertPool::make_host_table(ggml_context * ctx, int layer) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ctx || !host_maps_ || layer < 0 || size_t(layer) >= layers_.size() || !layers_[size_t(layer)].slot_count)
        return nullptr;
    ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_, 1);
    if (!table) return nullptr;
    table->buffer = host_maps_;
    table->data =
        static_cast<char *>(ggml_backend_buffer_get_base(host_maps_)) + layers_[size_t(layer)].host_map_offset;
    return table;
}

ggml_tensor * GpuExpertPool::make_device_state(ggml_context * ctx, int layer) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ctx || !buffer_ || layer < 0 || size_t(layer) >= layers_.size() || !layers_[size_t(layer)].slot_count)
        return nullptr;
    const Layer & layout = layers_[size_t(layer)];
    ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert_ + 2 * layout.slot_count + 3);
    if (!state) return nullptr;
    state->buffer = buffer_;
    state->data = static_cast<char *>(ggml_backend_buffer_get_base(buffer_)) + layout.device_state_offset;
    return state;
}

ggml_tensor * GpuExpertPool::make_host_stage_tensor(ggml_context * ctx, int layer, int projection) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ctx || !host_stage_ || layer < 0 || size_t(layer) >= layers_.size() || projection < 0 ||
        projection >= MoeRecipe::max_exps)
        return nullptr;
    const Projection & proj = layers_[size_t(layer)].proj[projection];
    if (!proj.shape || !stage_projection_stride_[projection] ||
        stage_projection_stride_[projection] < proj.expert_bytes)
        return nullptr;
    const ggml_tensor * src = proj.shape;
    ggml_tensor * tensor =
        ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], layers_[size_t(layer)].slot_count);
    if (!tensor) return nullptr;
    tensor->nb[2] = stage_projection_stride_[projection];
    tensor->nb[3] = tensor->nb[2] * size_t(layers_[size_t(layer)].slot_count);
    tensor->data = stage_device_base_ + stage_projection_offset_[projection];
    return tensor;
}

int32_t * GpuExpertPool::host_fill_list(int layer) const {
    if (!host_stage_ || layer < 0 || size_t(layer) >= layers_.size() || !layers_[size_t(layer)].slot_count)
        return nullptr;
    return reinterpret_cast<int32_t *>(stage_host_base_ + size_t(layer) * stage_request_stride_);
}

bool GpuExpertPool::stage_projection(int layer, int fill_index, int projection, const void * bytes, size_t size) {
    if (!host_stage_ || !bytes || layer < 0 || size_t(layer) >= layers_.size() || fill_index < 0 ||
        fill_index >= layers_[size_t(layer)].slot_count || projection < 0 || projection >= MoeRecipe::max_exps ||
        size != layers_[size_t(layer)].proj[projection].expert_bytes || size > stage_projection_stride_[projection])
        return false;
    std::memcpy(stage_host_base_ + stage_projection_offset_[projection] +
                    size_t(fill_index) * stage_projection_stride_[projection],
                bytes, size);
    return true;
}

GpuExpertPool::DeviceStats GpuExpertPool::device_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    DeviceStats stats;
    if (!buffer_) return stats;
    for (const Layer & layer : layers_) {
        if (!layer.slot_count) continue;
        const size_t offset =
            layer.device_state_offset + (size_t(n_expert_) + 2 * size_t(layer.slot_count) + 1) * sizeof(int32_t);
        ggml_tensor counters{};
        counters.type = GGML_TYPE_I32;
        counters.ne[0] = 2;
        counters.ne[1] = counters.ne[2] = counters.ne[3] = 1;
        counters.nb[0] = sizeof(int32_t);
        counters.nb[1] = counters.nb[2] = counters.nb[3] = 2 * sizeof(int32_t);
        counters.buffer = buffer_;
        counters.data = static_cast<char *>(ggml_backend_buffer_get_base(buffer_)) + offset;
        int32_t values[2]{};
        ggml_backend_tensor_get(&counters, values, 0, sizeof(values));
        stats.routed += uint32_t(values[0]);
        stats.fills += uint32_t(values[1]);
    }
    return stats;
}

bool GpuExpertPool::update_mapping(int layer, const int32_t * slots, int count) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!buffer_ || !host_maps_ || layer < 0 || size_t(layer) >= layers_.size() || !layers_[size_t(layer)].slot_count ||
        !slots || count != n_expert_)
        return false;
    for (int expert = 0; expert < count; ++expert) {
        const int slot_id = slots[expert];
        if (slot_id < 0 || slot_id >= layers_[size_t(layer)].slot_count) return false;
        if (slot_id > 0) {
            const Slot & slot = slots_[size_t(layer)][size_t(slot_id)];
            if (slot.state != State::Ready || slot.expert != expert || !slot.pins) return false;
        }
    }
    const size_t bytes = size_t(count) * sizeof(int32_t);
    char * host =
        static_cast<char *>(ggml_backend_buffer_get_base(host_maps_)) + layers_[size_t(layer)].host_map_offset;
    std::memcpy(host, slots, bytes);
    ggml_tensor device_bytes{};
    device_bytes.type = GGML_TYPE_I8;
    for (int token = 0; token < map_batch_max_; ++token)
        std::memcpy(map_upload_.data() + size_t(token) * size_t(count), slots, bytes);
    const size_t device_bytes_count = bytes * size_t(map_batch_max_);
    device_bytes.ne[0] = int64_t(device_bytes_count);
    device_bytes.ne[1] = device_bytes.ne[2] = device_bytes.ne[3] = 1;
    device_bytes.nb[0] = 1;
    device_bytes.nb[1] = device_bytes.nb[2] = device_bytes.nb[3] = device_bytes_count;
    device_bytes.buffer = buffer_;
    device_bytes.data =
        static_cast<char *>(ggml_backend_buffer_get_base(buffer_)) + layers_[size_t(layer)].device_map_offset;
    ggml_backend_tensor_set(&device_bytes, map_upload_.data(), 0, device_bytes_count);
    return true;
}

bool GpuExpertPool::publish(Lease lease) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid(lease)) return false;
    Slot & slot = slots_[size_t(lease.layer)][size_t(lease.slot)];
    if (slot.state != State::Reserved || slot.uploaded_mask != layers_[size_t(lease.layer)].required_mask) return false;
    slot.state = State::Ready;
    slot.last_use = ++clock_;
    return true;
}

bool GpuExpertPool::abandon(Lease lease) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid(lease)) return false;
    Slot & slot = slots_[size_t(lease.layer)][size_t(lease.slot)];
    if (slot.state != State::Reserved) return false;
    slot.state = State::Empty;
    slot.expert = -1;
    return true;
}

GpuExpertPool::Lease GpuExpertPool::pin(int layer, int expert) {
    std::lock_guard<std::mutex> lock(mu_);
    if (layer < 0 || size_t(layer) >= slots_.size() || expert < 0 || expert >= n_expert_) return {};
    auto & slots = slots_[size_t(layer)];
    for (size_t i = 1; i < slots.size(); ++i) {
        Slot & slot = slots[i];
        if (slot.state == State::Ready && slot.expert == expert) {
            ++slot.pins;
            slot.last_use = ++clock_;
            return {layer, int(i), slot.generation};
        }
    }
    return {};
}

int GpuExpertPool::available_slots(int layer) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (layer < 0 || size_t(layer) >= slots_.size()) return 0;
    int count = 0;
    const auto & slots = slots_[size_t(layer)];
    for (size_t i = 1; i < slots.size(); ++i)
        if (slots[i].state == State::Empty || (slots[i].state == State::Ready && !slots[i].pins)) ++count;
    return count;
}

bool GpuExpertPool::unpin(Lease lease) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid(lease)) return false;
    Slot & slot = slots_[size_t(lease.layer)][size_t(lease.slot)];
    if (slot.state != State::Ready || !slot.pins) return false;
    --slot.pins;
    return true;
}

size_t GpuExpertPool::resident_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t count = 0;
    for (const auto & layer : slots_)
        for (const Slot & slot : layer)
            if (slot.state == State::Ready) ++count;
    return count;
}

} // namespace bmoe
