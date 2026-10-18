#include "bmoe/runtime.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 6) return 2;
    const bool upload_gate = argc == 6 && std::string(argv[5]) == "upload";
    const bool hybrid_gate = argc == 6 && std::string(argv[5]) == "hybrid";
    const bool branch_overlap_gate = argc == 6 && std::string(argv[5]) == "overlap";
    if (argc == 6 && !upload_gate && !hybrid_gate && !branch_overlap_gate) return 2;

    llama_backend_init();
    bool gpu_available = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        gpu_available |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    llama_backend_free();
    if (!gpu_available) {
        std::puts("hybrid GPU gate skipped: no GPU backend available");
        return 77;
    }

    bmoe::RunConfig resident;
    resident.model_path = argv[1];
    resident.prompt = "Hello world, this is a streaming test.";
    resident.n_predict = 24;
    resident.n_threads = 2;
    resident.n_ctx = 256;
    resident.n_ubatch = 64;
    resident.gpu.n_gpu_layers = -1;
    resident.gpu.kv_offload = argc >= 4 && std::string(argv[3]) == "kv-gpu";

    bmoe::RunConfig streamed = resident;
    streamed.moe.enabled = true;
    streamed.moe.io_threads = 2;
    streamed.moe.dense_weights = bmoe::DenseWeightsMode::Anonymous;
    if (argc >= 3) streamed.moe.cache_mb = std::stoi(argv[2]);
    if (upload_gate || hybrid_gate || branch_overlap_gate) streamed.moe.force_cache = true;
    streamed.gpu.expert_overlap = branch_overlap_gate;
    if (argc >= 5) streamed.gpu.expert_cache_mb = std::stoi(argv[4]);
    // Keep the byte-identity gate all-miss; mixed CPU/CUDA arithmetic has a
    // separate numerical gate and real-model smoke test.
    if (streamed.gpu.expert_cache_mb > 0) streamed.gpu.expert_cache_promote = false;
    const int requested_cache_mb = streamed.moe.cache_mb;
    std::printf("hybrid GPU gate: serial RAM cache=%d MiB, GPU expert pool=%d MiB, KV=%s, promotion=%s\n",
                requested_cache_mb, streamed.gpu.expert_cache_mb, streamed.gpu.kv_offload ? "GPU" : "CPU",
                streamed.gpu.expert_cache_promote ? "on" : "off");
    std::fflush(stdout);

    const bmoe::RunResult reference = bmoe::run(resident);
    if (!reference) {
        std::fprintf(stderr, "hybrid resident run failed: %s\n", reference.error.c_str());
        return 1;
    }
    if (upload_gate || hybrid_gate) {
        bmoe::RunConfig upload = streamed;
        upload.moe.overlap = true;
        upload.gpu.expert_miss_mode = upload_gate ? bmoe::RunConfig::GpuConfig::ExpertMissMode::Upload
                                                  : bmoe::RunConfig::GpuConfig::ExpertMissMode::Hybrid;
        const bmoe::RunResult gpu_misses = bmoe::run(upload);
        if (!gpu_misses || (upload_gate && gpu_misses.generated_text != reference.generated_text) ||
            (hybrid_gate && gpu_misses.summary.moe_read_mib <= 0.0)) {
            std::fprintf(stderr, "GPU miss execution failed: %s; reference='%s'; actual='%s'\n",
                         gpu_misses.error.c_str(), reference.generated_text.c_str(), gpu_misses.generated_text.c_str());
            return 1;
        }
        std::puts(upload_gate ? "GPU upload decode matches resident GPU output"
                              : "hybrid CPU/GPU miss decode completed with exact expert weights");
        return 0;
    }
    const bmoe::RunResult serial = bmoe::run(streamed);
    if (!serial || serial.generated_text != reference.generated_text || serial.summary.moe_read_mib <= 0.0) {
        std::fprintf(stderr, "hybrid streamed run differed: %s; reference='%s'; actual='%s'; read=%.3f MiB\n",
                     serial.error.c_str(), reference.generated_text.c_str(), serial.generated_text.c_str(),
                     serial.summary.moe_read_mib);
        return 1;
    }

#ifdef BMOE_HAVE_EXPERT_READY_HOOK
    bmoe::RunConfig no_cache = streamed;
    no_cache.moe.cache_mb = 0;
    no_cache.moe.cache_auto = false;
    no_cache.moe.force_cache = false;
    no_cache.moe.overlap = true;
    const bmoe::RunResult rejected = bmoe::run(no_cache);
    if (rejected || rejected.error.find("requires an expert cache") == std::string::npos) {
        std::fprintf(stderr, "CUDA overlap without a cache was not rejected\n");
        return 1;
    }
    streamed.moe.overlap = true;
    if (streamed.moe.cache_mb == 0) {
        streamed.moe.cache_mb = 2;
        streamed.moe.force_cache = true;
    }
    if (argc >= 3 && streamed.moe.cache_mb != requested_cache_mb) {
        std::fprintf(stderr, "hybrid overlap cache differs from requested %d MiB\n", requested_cache_mb);
        return 1;
    }
    std::printf("hybrid GPU gate: overlap RAM cache=%d MiB, GPU expert pool=%d MiB, KV=%s, promotion=%s\n",
                streamed.moe.cache_mb, streamed.gpu.expert_cache_mb, streamed.gpu.kv_offload ? "GPU" : "CPU",
                streamed.gpu.expert_cache_promote ? "on" : "off");
    std::fflush(stdout);
    const bmoe::RunResult overlap = bmoe::run(streamed);
    if (!overlap || overlap.generated_text != reference.generated_text || overlap.summary.moe_read_mib <= 0.0) {
        std::fprintf(stderr, "hybrid overlap run differed: %s; reference='%s'; actual='%s'; read=%.3f MiB\n",
                     overlap.error.c_str(), reference.generated_text.c_str(), overlap.generated_text.c_str(),
                     overlap.summary.moe_read_mib);
        return 1;
    }
#endif
    bmoe::RunConfig partial = streamed;
    partial.gpu.n_gpu_layers = 1;
    partial.gpu.expert_cache_mb = 0;
    partial.gpu.expert_overlap = false;
    partial.moe.release_mmap = true;
    partial.moe.require_release_mmap = true;
    const bmoe::RunResult unsafe_release = bmoe::run(partial);
    if (unsafe_release || unsafe_release.error.find("partial CUDA placement") == std::string::npos) {
        std::fprintf(stderr, "partial CUDA mapping release was not rejected\n");
        return 1;
    }
    std::puts("hybrid GPU resident, serial stream, and overlap outputs match");
    return 0;
}
