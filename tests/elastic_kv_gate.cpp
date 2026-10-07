#include "bmoe/session.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    bool has_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        has_gpu |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (!has_gpu) {
        llama_backend_free();
        return 77;
    }

    bmoe::RunConfig cfg;
    cfg.model_path = argv[1];
    cfg.n_ctx = 1024;
    cfg.n_ubatch = 64;
    cfg.moe.enabled = true;
    cfg.moe.cache_mb = 8;
    cfg.gpu.n_gpu_layers = -1;
    cfg.gpu.expert_cache_mb = 16;
    cfg.gpu.safety_mb = 128;
    cfg.gpu.cache_type_k = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;
    cfg.gpu.cache_type_v = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;

    bmoe::GenerateRequest request;
    request.prompt = std::string(700, 'a');
    request.n_predict = 16;
    request.render_text = false;
    std::string error;
    auto baseline = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const size_t baseline_bytes = baseline ? baseline->physical_kv_bytes() : 0;
    const bmoe::RunResult reference = baseline ? baseline->generate(request) : bmoe::RunResult{};

    const std::string rolling_path = std::string(argv[1]) + ".elastic-rolling";
    const std::string claude_path = std::string(argv[1]) + ".elastic-claude";
    std::error_code ec;
    std::filesystem::remove(rolling_path, ec);
    std::filesystem::remove(claude_path, ec);
    cfg.gpu.elastic_kv = true;
    cfg.gpu.elastic_kv_chunk_tokens = 512;
    cfg.agent_state_cache_path = rolling_path;
    auto elastic = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const size_t initial_bytes = elastic ? elastic->physical_kv_bytes() : 0;
    const bool initial_small = elastic && elastic->n_ctx() == 1024 && elastic->physical_kv_tokens() == 512 &&
                               initial_bytes > 0 && initial_bytes < baseline_bytes;
    const bmoe::RunResult first = elastic ? elastic->generate(request) : bmoe::RunResult{};
    const bool grew = elastic && elastic->physical_kv_tokens() == 1024 &&
                      elastic->physical_kv_bytes() > initial_bytes && elastic->kv_growth_count() == 1;
    const bmoe::RunResult second = elastic ? elastic->generate(request) : bmoe::RunResult{};
    elastic.reset();
    auto restarted = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bmoe::RunResult third = restarted ? restarted->generate(request) : bmoe::RunResult{};
    restarted.reset();
    if (!initial_small || !grew || !reference || !first || !second || !third ||
        reference.generated_text != first.generated_text || first.generated_text != second.generated_text ||
        first.generated_text != third.generated_text || second.summary.n_prompt >= first.summary.n_prompt ||
        third.summary.n_prompt >= first.summary.n_prompt) {
        std::fprintf(stderr,
                     "elastic KV agent gate failed: %s; bytes=%zu/%zu small=%d grew=%d equal=%d/%d/%d "
                     "prompt=%d/%d/%d\n",
                     error.c_str(), initial_bytes, baseline_bytes, initial_small, grew,
                     reference.generated_text == first.generated_text, first.generated_text == second.generated_text,
                     first.generated_text == third.generated_text, first.summary.n_prompt, second.summary.n_prompt,
                     third.summary.n_prompt);
        std::filesystem::remove(rolling_path, ec);
        baseline.reset();
        llama_backend_free();
        return 1;
    }

    baseline.reset();
    cfg.agent_state_cache_path.clear();
    cfg.n_ctx = 262144;
    auto full_logical = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bool logical_preserved = full_logical && full_logical->n_ctx() == 262144 &&
                                   full_logical->physical_kv_tokens() == 512 &&
                                   full_logical->physical_kv_bytes() == initial_bytes;
    full_logical.reset();
    if (!logical_preserved) {
        std::fprintf(stderr, "elastic KV logical-context gate failed: %s\n", error.c_str());
        std::filesystem::remove(rolling_path, ec);
        llama_backend_free();
        return 1;
    }

    cfg.n_ctx = 1024;
    cfg.chatml = true;
    cfg.gpu.elastic_kv = false;
    auto continuing_baseline = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    cfg.gpu.elastic_kv = true;
    auto continuing_elastic = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    bmoe::GenerateRequest turn_one = request;
    turn_one.prompt = std::string(200, 'a');
    turn_one.n_predict = 4;
    bmoe::GenerateRequest turn_two = turn_one;
    turn_two.prompt = std::string(350, 'b');
    turn_two.clear_kv = false;
    const auto baseline_one = continuing_baseline ? continuing_baseline->generate(turn_one) : bmoe::RunResult{};
    const auto elastic_one = continuing_elastic ? continuing_elastic->generate(turn_one) : bmoe::RunResult{};
    const bool stayed_small = continuing_elastic && continuing_elastic->physical_kv_tokens() == 512;
    const auto baseline_two = continuing_baseline ? continuing_baseline->generate(turn_two) : bmoe::RunResult{};
    const auto elastic_two = continuing_elastic ? continuing_elastic->generate(turn_two) : bmoe::RunResult{};
    const bool migrated_live = continuing_elastic && continuing_elastic->physical_kv_tokens() == 1024 &&
                               continuing_elastic->kv_growth_count() == 1;
    continuing_baseline.reset();
    continuing_elastic.reset();
    if (!baseline_one || !elastic_one || !baseline_two || !elastic_two || !stayed_small || !migrated_live ||
        baseline_one.generated_text != elastic_one.generated_text ||
        baseline_two.generated_text != elastic_two.generated_text) {
        std::fprintf(stderr, "elastic KV live-state growth gate failed: %s; stayed=%d migrated=%d equal=%d/%d\n",
                     error.c_str(), stayed_small, migrated_live,
                     baseline_one.generated_text == elastic_one.generated_text,
                     baseline_two.generated_text == elastic_two.generated_text);
        std::filesystem::remove(rolling_path, ec);
        llama_backend_free();
        return 1;
    }

    cfg.chatml = false;
    cfg.n_ctx = 1024;
    cfg.gpu.elastic_kv = true;
    cfg.claude_prefix_cache_path = claude_path;
    const std::string stable(600, 'a');
    auto prompt_for = [&](char suffix) {
        bmoe::GenerateRequest r = request;
        r.prompt = stable + std::string(200, suffix);
        r.claude_prefix = stable;
        r.n_predict = 12;
        return r;
    };
    auto bootstrap = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bmoe::RunResult captured = bootstrap ? bootstrap->generate(prompt_for('b')) : bmoe::RunResult{};
    bootstrap.reset();
    auto replay = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    auto prewarm_request = prompt_for('e');
    prewarm_request.n_predict = 0;
    prewarm_request.claude_prewarm = true;
    const bmoe::RunResult prewarmed = replay ? replay->generate(prewarm_request) : bmoe::RunResult{};
    const bmoe::RunResult after_restart = replay ? replay->generate(prompt_for('c')) : bmoe::RunResult{};
    replay.reset();
    cfg.gpu.elastic_kv = false;
    cfg.claude_prefix_cache_path.clear();
    auto cold_reference = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bmoe::RunResult cold_third = cold_reference ? cold_reference->generate(prompt_for('c')) : bmoe::RunResult{};
    cold_reference.reset();
    std::filesystem::remove(rolling_path, ec);
    std::filesystem::remove(claude_path, ec);
    llama_backend_free();
    if (!captured || !prewarmed || !after_restart || !cold_third || prewarmed.summary.n_generated != 0 ||
        after_restart.summary.n_prompt >= cold_third.summary.n_prompt ||
        after_restart.generated_text != cold_third.generated_text) {
        std::fprintf(stderr, "elastic KV Claude prefix gate failed: %s; prefilled=%d/%d, equal=%d\n", error.c_str(),
                     after_restart.summary.n_prompt, cold_third.summary.n_prompt,
                     after_restart.generated_text == cold_third.generated_text);
        return 1;
    }
    std::printf(
        "elastic KV passed: logical=1024 physical=512->1024, KV=%.2f->%.2f MiB; agent and Claude replay exact\n",
        double(initial_bytes) / (1024.0 * 1024.0), double(baseline_bytes) / (1024.0 * 1024.0));
    return 0;
}
