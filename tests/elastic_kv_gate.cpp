#include "bmoe/session.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <string>

static int mtp_growth_gate(const char * model_path) {
    bmoe::RunConfig cfg;
    cfg.model_path = model_path;
    cfg.n_ctx = 262144;
    cfg.n_ubatch = 64;
    cfg.moe.enabled = true;
    cfg.moe.cache_mb = 4400;
    cfg.moe.release_mmap = true;
    cfg.moe.require_release_mmap = true;
    cfg.gpu.n_gpu_layers = -1;
    cfg.gpu.safety_mb = 128;
    cfg.gpu.elastic_kv = true;
    cfg.gpu.elastic_kv_chunk_tokens = 512;
    cfg.gpu.cache_type_k = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;
    cfg.gpu.cache_type_v = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;
    cfg.spec.source = bmoe::DraftSource::mtp;
    cfg.spec.draft_max = 4;
    cfg.agent_state_cache_path =
        (std::filesystem::current_path() /
         ("mtp-growth-state-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin"))
            .string();

    bmoe::GenerateRequest request;
    for (int i = 0; i < 75; ++i)
        request.prompt += "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n";
    request.n_predict = 8;
    request.render_text = false;
    std::string error;
    auto first_session = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    if (!first_session) {
        std::fprintf(stderr, "MTP growth gate open failed: %s\n", error.c_str());
        return 1;
    }
    const bmoe::RunResult first = first_session->generate(request);
    const int grows = first_session->kv_growth_count();
    first_session.reset();
    auto restarted = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const bmoe::RunResult second = restarted ? restarted->generate(request) : bmoe::RunResult{};
    restarted.reset();
    if (!first || !second || grows < 1 || first.generated_text != second.generated_text ||
        second.summary.n_prompt >= first.summary.n_prompt || first.summary.mtp_draft_read_mib != 0.0 ||
        second.summary.mtp_draft_read_mib != 0.0) {
        std::fprintf(stderr, "MTP growth gate failed: %s; grows=%d prompt=%d/%d mtp_reads=%.3f/%.3f MiB\n",
                     error.c_str(), grows, first.summary.n_prompt, second.summary.n_prompt,
                     first.summary.mtp_draft_read_mib, second.summary.mtp_draft_read_mib);
        return 1;
    }
    std::printf("MTP growth and restart gate passed: grows=%d prompt=%d/%d, exact continuation, no MTP reads\n", grows,
                first.summary.n_prompt, second.summary.n_prompt);
    return 0;
}

static int rebalance_gate(const char * model_path, bool mtp, bool pressure_mode = false) {
    bmoe::RunConfig cfg;
    cfg.model_path = model_path;
    cfg.n_ctx = mtp ? 262144 : 8192;
    cfg.n_ubatch = 64;
    cfg.moe.enabled = true;
    cfg.moe.cache_mb = mtp ? 4400 : 8;
    cfg.moe.release_mmap = mtp;
    cfg.moe.require_release_mmap = mtp;
    cfg.gpu.n_gpu_layers = -1;
    cfg.gpu.expert_cache_mb = mtp ? 1152 : 128;
    cfg.gpu.expert_performance_floor_mb = pressure_mode ? cfg.gpu.expert_cache_mb : (mtp ? 1024 : 32);
    cfg.gpu.expert_miss_mode =
        mtp ? bmoe::RunConfig::GpuConfig::ExpertMissMode::Cpu : bmoe::RunConfig::GpuConfig::ExpertMissMode::Upload;
    cfg.gpu.safety_mb = 128;
    cfg.gpu.elastic_kv = true;
    cfg.gpu.elastic_kv_chunk_tokens = 512;
    cfg.gpu.cache_type_k = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;
    cfg.gpu.cache_type_v = bmoe::RunConfig::GpuConfig::KvCacheType::Q4_0;
    if (mtp) {
        cfg.spec.source = bmoe::DraftSource::mtp;
        cfg.spec.draft_max = 4;
    }
    std::string error;
    auto probe = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    if (!probe) {
        std::fprintf(stderr, "rebalance probe open failed: %s\n", error.c_str());
        return 1;
    }
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = ggml_backend_dev_get(i);
            break;
        }
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
    probe.reset();
    if (!total_bytes || free_bytes < 169ull * 1024ull * 1024ull) return 77;
    cfg.gpu.safety_mb = int(free_bytes >> 20) - 40;
    auto session = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    if (!session) {
        std::fprintf(stderr, "rebalance open failed: %s\n", error.c_str());
        return 1;
    }
    const size_t initial_pool = session->gpu_expert_cache_bytes();
    ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
    const size_t safety_bytes = size_t(cfg.gpu.safety_mb) * 1024ull * 1024ull;
    if (free_bytes <= safety_bytes + 4ull * 1024ull * 1024ull) return 77;
    const size_t pressure_bytes = ((free_bytes - safety_bytes - 2ull * 1024ull * 1024ull) >> 21) << 21;
    auto * pressure = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(device), pressure_bytes);
    if (!pressure) {
        std::fprintf(stderr, "rebalance pressure allocation failed\n");
        return 1;
    }
    ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
    if (free_bytes < safety_bytes) {
        ggml_backend_buffer_free(pressure);
        std::fprintf(stderr, "rebalance pressure crossed the safety rail\n");
        return 1;
    }
    bmoe::GenerateRequest long_request;
    if (mtp) {
        for (int i = 0; i < 75; ++i)
            long_request.prompt += "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n";
    } else {
        long_request.prompt = std::string(6000, 'a');
    }
    long_request.n_predict = 4;
    long_request.render_text = false;
    const auto long_result = session->generate(long_request);
    const int long_kv = session->physical_kv_tokens();
    const size_t small_pool = session->gpu_expert_cache_bytes();
    ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
    const bool safe_after_growth = total_bytes && free_bytes >= safety_bytes;
    const size_t free_after_growth = free_bytes;
    ggml_backend_buffer_free(pressure);
    if (pressure_mode) {
        bmoe::GenerateRequest short_request = long_request;
        short_request.prompt = "short";
        short_request.clear_kv = false;
        const auto recovered = session->generate(short_request);
        cfg.gpu.safety_mb = 128;
        auto reference = bmoe::Session::open(bmoe::session_config_from(cfg), error);
        const auto expected = reference ? reference->generate(short_request) : bmoe::RunResult{};
        if (long_result || !long_result.context_pressure ||
            long_result.required_kv_tokens <= long_result.current_kv_tokens ||
            long_result.expert_floor_mb != cfg.gpu.expert_cache_mb || small_pool < initial_pool || !recovered ||
            !expected || recovered.generated_text != expected.generated_text ||
            session->physical_kv_tokens() != long_kv) {
            std::fprintf(stderr, "pressure gate failed: pressure=%d floor=%d cache=%zu/%zu recovered=%s reference=%s\n",
                         long_result.context_pressure, long_result.expert_floor_mb, small_pool, initial_pool,
                         recovered.error.c_str(), expected.error.c_str());
            return 1;
        }
        std::printf("pressure gate passed: required=%d current=%d floor=%d cache=%zu; next request recovered\n",
                    long_result.required_kv_tokens, long_result.current_kv_tokens, long_result.expert_floor_mb,
                    small_pool);
        return 0;
    }
    bmoe::GenerateRequest short_request = long_request;
    short_request.prompt = "short";
    const auto short_result = long_result ? session->generate(short_request) : bmoe::RunResult{};
    const int short_kv = session->physical_kv_tokens();
    const size_t restored_pool = session->gpu_expert_cache_bytes();
    ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
    const bool safe_after_shrink = total_bytes && free_bytes >= safety_bytes;
    session.reset();
    cfg.gpu.safety_mb = 128;
    auto reference = bmoe::Session::open(bmoe::session_config_from(cfg), error);
    const auto reference_long = reference ? reference->generate(long_request) : bmoe::RunResult{};
    const auto reference_short = reference_long ? reference->generate(short_request) : bmoe::RunResult{};
    if (!long_result || !short_result || !safe_after_growth || !safe_after_shrink || long_kv <= 512 ||
        small_pool >= initial_pool || short_kv >= long_kv || restored_pool <= small_pool ||
        restored_pool > initial_pool || !reference_long || !reference_short ||
        long_result.generated_text != reference_long.generated_text ||
        short_result.generated_text != reference_short.generated_text) {
        std::fprintf(stderr,
                     "rebalance gate failed: open=%s long=%s short=%s; KV=%d->%d cache=%zu->%zu->%zu exact=%d/%d "
                     "safe=%d/%d free=%zu/%zu rail=%zu\n",
                     error.c_str(), long_result.error.c_str(), short_result.error.c_str(), long_kv, short_kv,
                     initial_pool, small_pool, restored_pool,
                     long_result.generated_text == reference_long.generated_text,
                     short_result.generated_text == reference_short.generated_text, safe_after_growth,
                     safe_after_shrink, free_after_growth, free_bytes, safety_bytes);
        return 1;
    }
    std::printf("rebalance gate passed: KV=%d->%d cache=%zu->%zu->%zu\n", long_kv, short_kv, initial_pool, small_pool,
                restored_pool);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 2 && (argc != 3 || (std::string(argv[2]) != "mtp-real" && std::string(argv[2]) != "rebalance" &&
                                    std::string(argv[2]) != "mtp-rebalance" && std::string(argv[2]) != "pressure")))
        return 2;
    llama_backend_init();
    bool has_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        has_gpu |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (!has_gpu) {
        llama_backend_free();
        return 77;
    }
    if (argc == 3) {
        const std::string mode = argv[2];
        const int result = mode == "mtp-real" ? mtp_growth_gate(argv[1])
                           : rebalance_gate(argv[1], mode == "mtp-rebalance", mode == "pressure");
        llama_backend_free();
        return result;
    }

    bmoe::RunConfig cfg;
    cfg.model_path = argv[1];
    cfg.n_ctx = 1024;
    cfg.n_ubatch = 64;
    cfg.moe.enabled = true;
    cfg.moe.cache_mb = 8;
    cfg.gpu.n_gpu_layers = -1;
    cfg.gpu.expert_cache_mb = 16;
    cfg.gpu.expert_miss_mode = bmoe::RunConfig::GpuConfig::ExpertMissMode::Upload;
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
