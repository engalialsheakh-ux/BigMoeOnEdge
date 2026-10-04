#include "bmoe/route_trace.h"
#include "bmoe/runtime.h"
#include "bmoe/session.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Trace final : bmoe::IRouteTraceSink {
    bmoe::RouteTraceStatic info;
    std::vector<bmoe::RouteTraceRow> rows;

    void on_static(const bmoe::RouteTraceStatic & value) override { info = value; }
    void on_rows(const bmoe::RouteTraceRow * first, size_t count) override {
        rows.insert(rows.end(), first, first + count);
    }
};

bool check_prefill_routes(const Trace & trace, int n_prompt, double read_mib) {
    if (trace.info.n_expert != 256 || trace.info.n_expert_used != 8) return false;
    std::set<int> all;
    std::set<int> first;
    std::set<int> later;
    std::set<std::pair<int, int>> union_ids;
    std::vector<int> slots((size_t) n_prompt, 0);
    for (const auto & row : trace.rows) {
        if (row.phase != 0 || row.layer != 0) continue;
        if (row.step < 0 || row.step >= n_prompt || row.slot < 0 || row.slot >= 8 || row.expert < 0 ||
            row.expert >= 256)
            return false;
        ++slots[(size_t) row.step];
        all.insert(row.expert);
        (row.step == 0 ? first : later).insert(row.expert);
    }
    for (const auto & row : trace.rows)
        if (row.phase == 0) union_ids.emplace(row.layer, row.expert);
    uint64_t required_bytes = 0;
    for (const auto & [layer, expert] : union_ids) {
        (void) expert;
        if (layer < 0 || (size_t) layer >= trace.info.expert_bytes_per_layer.size()) return false;
        required_bytes += trace.info.expert_bytes_per_layer[(size_t) layer];
    }
    if (read_mib * (1024.0 * 1024.0) + 1.0 < (double) required_bytes) return false;
    for (int count : slots)
        if (count != 8) return false;
    // Repeated prompt tokens share some experts while also routing others; both cases must be
    // present so a flat or one-token-only ID gather cannot satisfy the test accidentally.
    bool duplicate = false;
    bool changed = false;
    for (int expert : later) {
        duplicate |= first.count(expert) != 0;
        changed |= first.count(expert) == 0;
    }
    return duplicate && changed && all.size() > 8;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    bool gpu_available = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        gpu_available |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    llama_backend_free();
    if (!gpu_available) return 77;

    bmoe::RunConfig resident;
    resident.model_path = argv[1];
    resident.prompt = std::string(32, 'a'); // BOS + 32 byte tokens: crosses CUDA's host-op threshold
    resident.n_predict = 1;
    resident.n_ctx = 256;
    resident.n_ubatch = 64;
    resident.n_threads = 2;
    resident.gpu.n_gpu_layers = -1;
    resident.gpu.kv_offload = false;
    resident.gpu.flash_attn = bmoe::RunConfig::GpuConfig::FlashAttention::On;
    resident.gpu.cache_type_k = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;
    resident.gpu.cache_type_v = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;

    const bmoe::RunResult reference = bmoe::run(resident);
    if (!reference || reference.summary.n_prompt != 33) {
        std::fprintf(stderr, "resident batched prefill failed: %s (tokens=%d)\n", reference.error.c_str(),
                     reference.summary.n_prompt);
        return 1;
    }

    bmoe::RunConfig streamed = resident;
    streamed.moe.enabled = true;
    streamed.moe.io_threads = 2;
    streamed.moe.cache_mb = 2;
    streamed.moe.force_cache = true;
    Trace trace;
    const bmoe::RunResult batched = bmoe::run(streamed, nullptr, nullptr, &trace);
    if (!batched || batched.summary.n_prompt != 33 || batched.summary.prefill_read_mib <= 0.0 ||
        batched.generated_text != reference.generated_text ||
        !check_prefill_routes(trace, 33, batched.summary.prefill_read_mib)) {
        std::fprintf(stderr, "streamed batched prefill failed: %s (tokens=%d, read=%.3f MiB)\n", batched.error.c_str(),
                     batched.summary.n_prompt, batched.summary.prefill_read_mib);
        return 1;
    }

    streamed.prompt.clear(); // BOS alone exercises a one-token graph with the same cache policy.
    const bmoe::RunResult one = bmoe::run(streamed);
    if (!one || one.summary.n_prompt != 1) {
        std::fprintf(stderr, "streamed one-token prefill failed: %s (tokens=%d)\n", one.error.c_str(),
                     one.summary.n_prompt);
        return 1;
    }
    streamed.prompt = resident.prompt;
    streamed.moe.overlap = true;
    const bmoe::RunResult overlap = bmoe::run(streamed);
    if (!overlap || overlap.generated_text != reference.generated_text) {
        std::fprintf(stderr, "overlapped batched prefill failed: %s\n", overlap.error.c_str());
        return 1;
    }
    streamed.gpu.expert_cache_mb = 2;
    streamed.gpu.expert_cache_promote = false;
    const bmoe::RunResult gpu_pool = bmoe::run(streamed);
    if (!gpu_pool || gpu_pool.generated_text != reference.generated_text) {
        std::fprintf(stderr, "GPU expert pool batched prefill failed: %s\n", gpu_pool.error.c_str());
        return 1;
    }
    streamed.gpu.expert_cache_mb = 0;
    streamed.moe.overlap = false;
    streamed.moe.release_mmap = true;
    streamed.moe.require_release_mmap = true;
    const bmoe::RunResult released = bmoe::run(streamed);
    if (!released || released.generated_text != reference.generated_text) {
        std::fprintf(stderr, "mapping-released batched prefill failed: %s\n", released.error.c_str());
        return 1;
    }
    std::string open_error;
    std::unique_ptr<bmoe::Session> session = bmoe::Session::open(bmoe::session_config_from(streamed), open_error);
    if (!session) {
        std::fprintf(stderr, "hybrid session open failed: %s\n", open_error.c_str());
        return 1;
    }
    bmoe::GenerateRequest request;
    request.prompt = streamed.prompt;
    request.n_predict = streamed.n_predict;
    request.clear_kv = true;
    const bmoe::RunResult session_run = session->generate(request);
    if (!session_run || session_run.summary.n_prompt != 33 || session_run.generated_text != reference.generated_text) {
        std::fprintf(stderr, "hybrid session batched prefill failed: %s\n", session_run.error.c_str());
        return 1;
    }
    std::puts("CUDA hybrid one-token and batched streamed prefill passed");
    return 0;
}
