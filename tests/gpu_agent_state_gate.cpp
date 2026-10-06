#include "bmoe/route_trace.h"
#include "bmoe/session.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
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
    if (!cold || !restored || !persisted || cold.generated_text != restored.generated_text ||
        cold.generated_text != persisted.generated_text || cold.summary.n_prompt <= restored.summary.n_prompt ||
        cold.summary.n_prompt <= persisted.summary.n_prompt || in_process.hits == 0 || restart.hits == 0 ||
        in_process.decisions[0] != in_process.decisions[1] || in_process.decisions[0] != restart.decisions[0]) {
        std::fprintf(stderr, "GPU agent state replay failed: %s; prefix=%d/%d/%d; hits=%zu/%zu; equal=%d/%d\n",
                     error.c_str(), cold.summary.n_prompt, restored.summary.n_prompt, persisted.summary.n_prompt,
                     in_process.hits, restart.hits, cold.generated_text == restored.generated_text,
                     cold.generated_text == persisted.generated_text);
        std::filesystem::remove(checkpoint, ec);
        llama_backend_free();
        return 1;
    }
    std::printf("GPU agent state replay passed: prefix=%d/%d/%d, GPU hits=%zu/%zu\n", cold.summary.n_prompt,
                restored.summary.n_prompt, persisted.summary.n_prompt, in_process.hits, restart.hits);

    // A stable Claude prefix remains reusable after a different conversation replaces the rolling state.
    const std::string base_path = checkpoint + ".claude";
    std::filesystem::remove(checkpoint, ec);
    std::filesystem::remove(base_path, ec);
    cfg.claude_prefix_cache_path = base_path;
    const std::string stable(600, 'a');
    auto prompt_for = [&](char suffix) {
        bmoe::GenerateRequest r = request;
        r.prompt = stable + std::string(200, suffix);
        r.claude_prefix = stable;
        r.n_predict = 12;
        return r;
    };
    auto first = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bmoe::RunResult bootstrapped = first ? first->generate(prompt_for('b')) : bmoe::RunResult{};
    const auto read_base = [&]() {
        std::ifstream in(base_path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    const std::string saved_base = read_base();
    const bmoe::RunResult second = first ? first->generate(prompt_for('c')) : bmoe::RunResult{};
    first.reset();
    const bool base_unchanged = !saved_base.empty() && saved_base == read_base();
    auto after_restart = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    auto prewarm_request = prompt_for('e');
    prewarm_request.n_predict = 0;
    prewarm_request.claude_prewarm = true;
    const bmoe::RunResult prewarmed = after_restart ? after_restart->generate(prewarm_request) : bmoe::RunResult{};
    const bmoe::RunResult third = after_restart ? after_restart->generate(prompt_for('d')) : bmoe::RunResult{};
    after_restart.reset();
    cfg.agent_state_cache_path.clear();
    cfg.claude_prefix_cache_path.clear();
    auto reference = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bmoe::RunResult cold_second = reference ? reference->generate(prompt_for('c')) : bmoe::RunResult{};
    reference.reset();
    reference = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bmoe::RunResult cold_third = reference ? reference->generate(prompt_for('d')) : bmoe::RunResult{};
    reference.reset();
    std::filesystem::remove(checkpoint, ec);
    std::filesystem::remove(base_path, ec);
    llama_backend_free();
    if (!bootstrapped || !second || !prewarmed || prewarmed.summary.n_generated != 0 || !third || !cold_second ||
        !cold_third || !base_unchanged || second.generated_text != cold_second.generated_text ||
        third.generated_text != cold_third.generated_text || second.summary.n_prompt >= cold_second.summary.n_prompt ||
        third.summary.n_prompt >= cold_third.summary.n_prompt) {
        std::fprintf(stderr,
                     "Claude prefix replay failed: %s; prefilled=%d/%d, %d/%d; base unchanged=%d; equal=%d/%d; "
                     "ok=%d/%d/%d/%d/%d\n",
                     error.c_str(), second.summary.n_prompt, cold_second.summary.n_prompt, third.summary.n_prompt,
                     cold_third.summary.n_prompt, base_unchanged, second.generated_text == cold_second.generated_text,
                     third.generated_text == cold_third.generated_text, bool(bootstrapped), bool(second), bool(third),
                     bool(cold_second), bool(cold_third));
        return 1;
    }
    std::printf("Claude prefix replay passed: suffix=%d/%d vs cold=%d/%d; base unchanged across restart\n",
                second.summary.n_prompt, third.summary.n_prompt, cold_second.summary.n_prompt,
                cold_third.summary.n_prompt);
    return 0;
}
