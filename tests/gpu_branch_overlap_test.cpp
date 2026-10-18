#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

// A deterministic asynchronous device: queued nodes execute only at a fence.
// This proves host work is submitted before the fence without timing a real GPU.
struct DeviceProbe {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_device device{};
    ggml_backend_buffer_type buft{};
    ggml_backend gpu{};
    ggml_backend host{};
    std::vector<ggml_tensor *> pending;
    bool accepts_host = false;
    bool shared_allocator = false;
    bool host_saw_pending = false;
    bool observed_pending = false;
    bool observe = false;
    bool abort_host = false;
    int observations = 0;

    struct Storage {
        std::vector<unsigned char> bytes;
        void * base;
        explicit Storage(size_t size) : bytes(size + 64) {
            base = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(bytes.data()) + 63) & ~uintptr_t(63));
        }
    };

    static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t size) {
        ggml_backend_buffer_i iface{};
        iface.free_buffer = [](ggml_backend_buffer_t buffer) { delete static_cast<Storage *>(buffer->context); };
        iface.get_base = [](ggml_backend_buffer_t buffer) { return static_cast<Storage *>(buffer->context)->base; };
        iface.set_tensor = [](ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset,
                              size_t size) { std::memcpy(static_cast<char *>(tensor->data) + offset, data, size); };
        iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset,
                              size_t size) {
            std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
        };
        iface.clear = [](ggml_backend_buffer_t buffer, uint8_t value) {
            std::memset(static_cast<Storage *>(buffer->context)->base, value, buffer->size);
        };
        return ggml_backend_buffer_init(type, iface, new Storage(size), size);
    }

    void synchronize() {
        for (ggml_tensor * node : pending) {
            if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE) continue;
            const float * a = static_cast<const float *>(node->src[0]->data);
            const float * b = node->src[1] ? static_cast<const float *>(node->src[1]->data) : nullptr;
            float * out = static_cast<float *>(node->data);
            for (int64_t i = 0; i < ggml_nelements(node); ++i) {
                if (node->op == GGML_OP_SQR)
                    out[i] = a[i] * a[i];
                else if (node->op == GGML_OP_ADD)
                    out[i] = a[i] + b[i];
                else
                    GGML_ABORT("unexpected synthetic device operation");
            }
        }
        pending.clear();
    }

    DeviceProbe() {
        ggml_backend_cpu_set_n_threads(cpu, 1);
        buft.device = &device;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "test-private-device"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 64; };
        buft.iface.alloc_buffer = allocate;
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "test-device"; };
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            auto & self = *static_cast<DeviceProbe *>(dev->context);
            return self.shared_allocator ? ggml_backend_cpu_buffer_type() : &self.buft;
        };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
            const auto & self = *static_cast<DeviceProbe *>(dev->context);
            return type == &self.buft || (self.accepts_host && ggml_backend_buft_is_host(type));
        };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * node) {
            return std::strncmp(node->name, "host_", 5) != 0 &&
                   (node->op == GGML_OP_SQR || node->op == GGML_OP_ADD || node->op == GGML_OP_VIEW ||
                    node->op == GGML_OP_RESHAPE || node->op == GGML_OP_NONE);
        };
        gpu.context = this;
        gpu.device = &device;
        gpu.iface.get_name = [](ggml_backend_t) { return "test-device"; };
        gpu.iface.synchronize = [](ggml_backend_t backend) {
            static_cast<DeviceProbe *>(backend->context)->synchronize();
        };
        gpu.iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph * graph) {
            auto & self = *static_cast<DeviceProbe *>(backend->context);
            ggml_tensor ** nodes = ggml_graph_nodes(graph);
            self.pending.insert(self.pending.end(), nodes, nodes + ggml_graph_n_nodes(graph));
            return GGML_STATUS_SUCCESS;
        };
        host.context = this;
        host.device = ggml_backend_get_device(cpu);
        host.iface.get_name = [](ggml_backend_t) { return "test-host"; };
        host.iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph * graph) {
            auto & self = *static_cast<DeviceProbe *>(backend->context);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                if (std::strcmp(ggml_graph_node(graph, i)->name, "host_work") == 0) {
                    self.host_saw_pending |= !self.pending.empty();
                    if (self.abort_host) return GGML_STATUS_ABORTED;
                }
            }
            return ggml_backend_graph_compute(self.cpu, graph);
        };
    }

    ~DeviceProbe() { ggml_backend_free(cpu); }

    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        auto & self = *static_cast<DeviceProbe *>(data);
        if (ask) return self.observe && std::strcmp(tensor->name, "device_branch") == 0;
        ++self.observations;
        self.observed_pending |= !self.pending.empty();
        return true;
    }
};

enum class Case { Default, Independent, Observed, SharedSource, SharedAllocator, Transfer, Abort };

bool run(Case test, const char * name) {
    DeviceProbe probe;
    probe.accepts_host = test == Case::SharedSource || test == Case::SharedAllocator;
    probe.shared_allocator = test == Case::SharedAllocator;
    probe.observe = test == Case::Observed;
    probe.abort_host = test == Case::Abort;
    ggml_init_params params{1024 * 1024, nullptr, true};
    auto tensors = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    auto shared = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    auto graph_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    ggml_tensor * input = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, 16);
    ggml_tensor * bias = ggml_new_tensor_1d(shared.get(), GGML_TYPE_F32, 16);
    auto input_buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(tensors.get(), &probe.gpu), ggml_backend_buffer_free);
    auto shared_buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(shared.get(), probe.cpu), ggml_backend_buffer_free);
    if (!input_buffer || !shared_buffer) return false;
    ggml_backend_t backends[]{&probe.gpu, &probe.host};
    auto sched = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>(
        ggml_backend_sched_new(backends, nullptr, 2, 128, false, false), ggml_backend_sched_free);
    if (test != Case::Default) ggml_backend_sched_set_branch_overlap(sched.get(), true);
    ggml_backend_sched_set_eval_callback(sched.get(), DeviceProbe::callback, &probe);
    auto graph = ggml_new_graph_custom(graph_ctx.get(), 128, false);

    ggml_tensor * host_input = input;
    if (test != Case::Transfer) {
        host_input = ggml_dup(graph_ctx.get(), input);
        ggml_set_name(host_input, "host_input");
        ggml_backend_sched_set_tensor_backend(sched.get(), host_input, &probe.host);
        ggml_build_forward_expand(graph, host_input);
    }
    ggml_tensor * device_result = ggml_sqr(graph_ctx.get(), input);
    ggml_backend_sched_set_tensor_backend(sched.get(), device_result, &probe.gpu);
    if (test == Case::SharedSource) {
        auto shared_view = ggml_view_1d(graph_ctx.get(), bias, 16, 0);
        device_result = ggml_add(graph_ctx.get(), device_result, shared_view);
        ggml_backend_sched_set_tensor_backend(sched.get(), device_result, &probe.gpu);
    }
    ggml_set_name(device_result, "device_branch");
    ggml_build_forward_expand(graph, device_result);
    auto host_result = ggml_scale(graph_ctx.get(), host_input, 3.0f);
    ggml_set_name(host_result, "host_work");
    ggml_backend_sched_set_tensor_backend(sched.get(), host_result, &probe.host);
    auto result = ggml_add(graph_ctx.get(), host_result, device_result);
    ggml_set_name(result, "joined_result");
    ggml_backend_sched_set_tensor_backend(sched.get(), result, &probe.gpu);
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    if (!ggml_backend_sched_alloc_graph(sched.get(), graph)) return false;

    const bool expected_overlap = test == Case::Independent || test == Case::Abort;
    for (int iteration = 0; iteration < 4; ++iteration) {
        std::array<float, 16> values{}, offsets{}, actual{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = float(int(i) - 7 + iteration) / 8.0f;
            offsets[i] = float(int(i) + iteration) / 16.0f;
        }
        ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
        ggml_backend_tensor_set(bias, offsets.data(), 0, sizeof(offsets));
        probe.host_saw_pending = false;
        const auto status = ggml_backend_sched_graph_compute_async(sched.get(), graph);
        if (status != (probe.abort_host ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS)) return false;
        if (probe.host_saw_pending != expected_overlap || probe.observed_pending || !probe.pending.empty()) {
            std::fprintf(stderr, "%s: overlap=%d expected=%d observation-pending=%d pending=%zu\n", name,
                         probe.host_saw_pending, expected_overlap, probe.observed_pending, probe.pending.size());
            return false;
        }
        if (probe.abort_host) continue;
        ggml_backend_tensor_get(result, actual.data(), 0, sizeof(actual));
        for (size_t i = 0; i < values.size(); ++i) {
            const float expected =
                values[i] * values[i] + 3.0f * values[i] + (test == Case::SharedSource ? offsets[i] : 0.0f);
            if (std::fabs(actual[i] - expected) > 1e-6f) {
                std::fprintf(stderr, "%s: numerical mismatch at iteration %d, element %zu\n", name, iteration, i);
                return false;
            }
        }
    }
    if (probe.observe && probe.observations != 4) return false;
    std::printf("%s: %s; overlap=%d; observations=%d\n", name, probe.abort_host ? "abort fenced" : "exact output",
                expected_overlap, probe.observations);
    return true;
}

} // namespace

int main() {
    return run(Case::Default, "default serial") && run(Case::Independent, "private independent branches") &&
                   run(Case::Observed, "observed device branch") && run(Case::SharedSource, "shared source view") &&
                   run(Case::SharedAllocator, "shared allocator") && run(Case::Transfer, "dependent input transfer") &&
                   run(Case::Abort, "host abort after device launch")
               ? 0
               : 1;
}
