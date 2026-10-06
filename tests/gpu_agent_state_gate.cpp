#include "bmoe/route_trace.h"
#include "bmoe/session.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct HitSink final : bmoe::IRouteTraceSink {
    struct Decision {
        int step;
        int layer;
        int slot;
        int expert;
        uint8_t gpu_hit;
        bool operator==(const Decision & rhs) const {
            return step == rhs.step && layer == rhs.layer && slot == rhs.slot && expert == rhs.expert &&
                   gpu_hit == rhs.gpu_hit;
        }
    };
    size_t hits = 0;
    std::vector<Decision> decisions[2];
    void on_static(const bmoe::RouteTraceStatic &) override {}
    void on_rows(const bmoe::RouteTraceRow * rows, size_t count) override {
        for (size_t i = 0; i < count; ++i)
            if (rows[i].phase == 1) {
                hits += rows[i].gpu_hit;
                if (rows[i].turn >= 0 && rows[i].turn < 2)
                    decisions[rows[i].turn].push_back(
                        {rows[i].step, rows[i].layer, rows[i].slot, rows[i].expert, rows[i].gpu_hit});
            }
    }
};

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    bool has_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        has_gpu |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (!has_gpu) {
        llama_backend_free();
        std::puts("GPU agent state gate skipped: no GPU backend");
        return 77;
    }

    const std::string checkpoint = std::string(argv[1]) + ".gpu-agent-state-gate";
    std::error_code ec;
    std::filesystem::remove(checkpoint, ec);
    bmoe::RunConfig cfg;
    cfg.model_path = argv[1];
    cfg.agent_state_cache_path = checkpoint;
    cfg.n_ctx = 1024;
    cfg.n_ubatch = 64;
    cfg.moe.enabled = true;
    cfg.moe.cache_mb = 8;
    cfg.gpu.n_gpu_layers = -1;
    cfg.gpu.expert_cache_mb = 16;
    cfg.gpu.safety_mb = 128;
    cfg.gpu.expert_cache_window = 4;

    bmoe::GenerateRequest request;
    request.prompt = std::string(700, 'a');
    request.n_predict = 48;
    request.render_text = false;
    std::string error;
    HitSink in_process;
    auto session = bmoe::Session::open(bmoe::session_config_from(cfg), error, &in_process);
    const bmoe::RunResult cold = session ? session->generate(request) : bmoe::RunResult{};
    const bmoe::RunResult restored = session ? session->generate(request) : bmoe::RunResult{};
    session.reset();
    HitSink restart;
    auto reopened = bmoe::Session::open(bmoe::session_config_from(cfg), error, &restart);
    const bmoe::RunResult persisted = reopened ? reopened->generate(request) : bmoe::RunResult{};
    reopened.reset();
    std::filesystem::remove(checkpoint, ec);
    llama_backend_free();

    if (!cold || !restored || !persisted || cold.generated_text != restored.generated_text ||
        cold.generated_text != persisted.generated_text || cold.summary.n_prompt <= restored.summary.n_prompt ||
        cold.summary.n_prompt <= persisted.summary.n_prompt || in_process.hits == 0 || restart.hits == 0 ||
        in_process.decisions[0] != in_process.decisions[1] || in_process.decisions[0] != restart.decisions[0]) {
        std::fprintf(stderr, "GPU agent state replay failed: %s; prefix=%d/%d/%d; hits=%zu/%zu; equal=%d/%d\n",
                     error.c_str(), cold.summary.n_prompt, restored.summary.n_prompt, persisted.summary.n_prompt,
                     in_process.hits, restart.hits, cold.generated_text == restored.generated_text,
                     cold.generated_text == persisted.generated_text);
        return 1;
    }
    std::printf("GPU agent state replay passed: prefix=%d/%d/%d, GPU hits=%zu/%zu\n", cold.summary.n_prompt,
                restored.summary.n_prompt, persisted.summary.n_prompt, in_process.hits, restart.hits);
    return 0;
}
