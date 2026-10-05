#include "bmoe/runtime.h"
#include "bmoe/session.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

static bool check_case(llama_model * model, int n_tokens, int tile_tokens = 512) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 1024;
    params.n_batch = 1024;
    params.n_ubatch = 64;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.op_offload = false;

    std::unique_ptr<llama_context, decltype(&llama_free)> reference(llama_init_from_model(model, params), llama_free);
    std::unique_ptr<llama_context, decltype(&llama_free)> tiled(llama_init_from_model(model, params), llama_free);
    if (!reference || !tiled) return false;

    std::vector<llama_token> tokens(n_tokens);
    for (int i = 0; i < n_tokens; ++i)
        tokens[i] = 4 + (i % 220);
    if (llama_decode(reference.get(), llama_batch_get_one(tokens.data(), n_tokens)) != 0 ||
        llama_decode_layer_tiled(tiled.get(), llama_batch_get_one(tokens.data(), n_tokens), tile_tokens) != 0) {
        std::fprintf(stderr, "tiled decode failed at %d tokens with tile %d\n", n_tokens, tile_tokens);
        return false;
    }

    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const size_t logit_bytes = n_vocab * sizeof(float);
    if (std::memcmp(llama_get_logits(reference.get()), llama_get_logits(tiled.get()), logit_bytes) != 0) {
        const float * a = llama_get_logits(reference.get());
        const float * b = llama_get_logits(tiled.get());
        float max_delta = 0.0f;
        size_t diff_count = 0;
        for (size_t i = 0; i < n_vocab; ++i) {
            const float delta = std::fabs(a[i] - b[i]);
            if (delta != 0) ++diff_count;
            if (delta > max_delta) max_delta = delta;
        }
        std::fprintf(stderr, "tiled prompt logits differ at %d tokens: %zu/%zu max delta=%g first=%g/%g\n", n_tokens,
                     diff_count, n_vocab, max_delta, a[0], b[0]);
        if (max_delta > 1.0e-6f) return false;
    }
    const llama_pos reference_pos = llama_memory_seq_pos_max(llama_get_memory(reference.get()), 0);
    const llama_pos tiled_pos = llama_memory_seq_pos_max(llama_get_memory(tiled.get()), 0);
    if (reference_pos != tiled_pos || tiled_pos != n_tokens - 1) {
        std::fprintf(stderr, "tiled sequence position differs: reference=%d tiled=%d\n", reference_pos, tiled_pos);
        return false;
    }
    auto metadata_equal = [&](llama_state_seq_flags flags, size_t metadata_bytes) {
        const size_t size_a = llama_state_seq_get_size_ext(reference.get(), 0, flags);
        const size_t size_b = llama_state_seq_get_size_ext(tiled.get(), 0, flags);
        if (size_a != size_b || size_a < metadata_bytes) return false;
        std::vector<uint8_t> a(size_a), b(size_b);
        if (llama_state_seq_get_data_ext(reference.get(), a.data(), a.size(), 0, flags) != size_a ||
            llama_state_seq_get_data_ext(tiled.get(), b.data(), b.size(), 0, flags) != size_b)
            return false;
        return std::memcmp(a.data(), b.data(), metadata_bytes) == 0;
    };
    // The serialized single-stream KV header is magic, sequence ID, stream count and cell count.
    // Each IMRoPE cell contributes position, sequence count, 2D position plus token ID, and sequence ID.
    if (!metadata_equal(0, 16 + size_t(n_tokens) * 24)) {
        std::fprintf(stderr, "tiled KV cell metadata differs at %d tokens\n", n_tokens);
        return false;
    }
    // Partial-only export contains recurrent metadata and no KV tensor data.
    const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t recurrent_count = 1;
    if (!metadata_equal(partial, 12 + recurrent_count * 8)) {
        std::fprintf(stderr, "tiled recurrent cell metadata differs at %d tokens\n", n_tokens);
        return false;
    }

    // A hybrid checkpoint must restore recurrent state as well as attention KV. Mutate the
    // live context, restore the same snapshot twice, and compare its continuation each time.
    const size_t state_size = llama_state_seq_get_size(tiled.get(), 0);
    if (state_size == 0) return false;
    std::vector<uint8_t> checkpoint(state_size);
    if (llama_state_seq_get_data(tiled.get(), checkpoint.data(), checkpoint.size(), 0) != state_size) return false;
    const size_t recurrent_size = llama_state_seq_get_size_ext(tiled.get(), 0, partial);
    std::vector<uint8_t> recurrent_before(recurrent_size), recurrent_after(recurrent_size);
    if (recurrent_size == 0 || llama_state_seq_get_data_ext(tiled.get(), recurrent_before.data(), recurrent_size, 0,
                                                            partial) != recurrent_size)
        return false;
    llama_token mutation = 31;
    if (llama_decode(tiled.get(), llama_batch_get_one(&mutation, 1)) != 0) return false;
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (llama_state_seq_set_data(tiled.get(), checkpoint.data(), checkpoint.size(), 0) == 0 ||
            llama_state_seq_get_size_ext(tiled.get(), 0, partial) != recurrent_size ||
            llama_state_seq_get_data_ext(tiled.get(), recurrent_after.data(), recurrent_size, 0, partial) !=
                recurrent_size ||
            recurrent_before != recurrent_after) {
            std::fprintf(stderr, "hybrid recurrent state did not restore at %d tokens (attempt %d)\n", n_tokens,
                         attempt + 1);
            return false;
        }
        llama_token next = 15;
        if (llama_decode(tiled.get(), llama_batch_get_one(&next, 1)) != 0) return false;
        if (attempt == 0 && llama_decode(reference.get(), llama_batch_get_one(&next, 1)) != 0) return false;
        const float * a = llama_get_logits(reference.get());
        const float * b = llama_get_logits(tiled.get());
        float max_delta = 0.0f;
        for (size_t i = 0; i < n_vocab; ++i)
            max_delta = std::max(max_delta, std::fabs(a[i] - b[i]));
        if (max_delta > 1.0e-6f) {
            std::fprintf(stderr, "post-restore decode differs at %d tokens: max delta=%g\n", n_tokens, max_delta);
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    bool gpu_available = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        gpu_available |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], params),
                                                                    llama_model_free);
    if (!model) return 1;
    bool ok = check_case(model.get(), 1) && check_case(model.get(), 64) && check_case(model.get(), 65) &&
              check_case(model.get(), 130) && check_case(model.get(), 520) && check_case(model.get(), 520, 1024) &&
              check_case(model.get(), 900, 1024) && check_case(model.get(), 900, 8192);
    model.reset();
    llama_backend_free();
    if (ok && gpu_available) {
        bmoe::RunConfig cfg;
        cfg.model_path = argv[1];
        cfg.prompt = std::string(520, 'a');
        cfg.n_ctx = 1024;
        cfg.n_ubatch = 64;
        cfg.n_predict = 8;
        cfg.gpu.n_gpu_layers = -1;
        cfg.moe.enabled = true;
        cfg.moe.cache_mb = 2;
        cfg.moe.force_cache = true;
        const bmoe::RunResult reference = bmoe::run(cfg);
        cfg.layer_tile_tokens = 512;
        const bmoe::RunResult tiled = bmoe::run(cfg);
        cfg.layer_tile_tokens = -1;
        const bmoe::RunResult adaptive = bmoe::run(cfg);
        ok = reference && tiled && adaptive && reference.generated_text == tiled.generated_text &&
             reference.generated_text == adaptive.generated_text;
        if (!ok) {
            std::fprintf(
                stderr,
                "streamed generated output differs: reference='%s' tiled='%s' adaptive='%s'; errors='%s'/'%s'/'%s'\n",
                reference.generated_text.c_str(), tiled.generated_text.c_str(), adaptive.generated_text.c_str(),
                reference.error.c_str(), tiled.error.c_str(), adaptive.error.c_str());
        }
        auto two_turns = [&](int tile, bmoe::RunResult & first, bmoe::RunResult & second) {
            bmoe::SessionConfig sc = bmoe::session_config_from(cfg);
            sc.layer_tile_tokens = tile;
            sc.chatml = true;
            std::string error;
            std::unique_ptr<bmoe::Session> session = bmoe::Session::open(sc, error);
            if (!session) {
                std::fprintf(stderr, "chat session open failed: %s\n", error.c_str());
                return false;
            }
            bmoe::GenerateRequest req;
            req.prompt = std::string(520, 'a');
            req.n_predict = 8;
            req.think = false;
            first = session->generate(req);
            if (!first) return false;
            req.prompt = "What next?";
            req.clear_kv = false;
            second = session->generate(req);
            return second && first.summary.n_prompt > 512 && second.summary.n_prompt > 0;
        };
        bmoe::RunResult first_ref, second_ref, first_auto, second_auto;
        const bool continued = two_turns(512, first_ref, second_ref) && two_turns(-1, first_auto, second_auto);
        if (!continued || first_ref.generated_text != first_auto.generated_text ||
            second_ref.generated_text != second_auto.generated_text ||
            first_ref.summary.n_prompt != first_auto.summary.n_prompt ||
            second_ref.summary.n_prompt != second_auto.summary.n_prompt ||
            second_ref.summary.n_past != second_auto.summary.n_past) {
            std::fprintf(stderr, "chat continuation differs: ref='%s'/'%s', auto='%s'/'%s'; suffix=%d/%d\n",
                         first_ref.generated_text.c_str(), second_ref.generated_text.c_str(),
                         first_auto.generated_text.c_str(), second_auto.generated_text.c_str(),
                         second_ref.summary.n_prompt, second_auto.summary.n_prompt);
            ok = false;
        }
    }
    if (ok) {
        const std::string checkpoint = std::string(argv[1]) + ".agent-state-gate";
        std::error_code ec;
        std::filesystem::remove(checkpoint, ec);
        bmoe::RunConfig cfg;
        cfg.model_path = argv[1];
        cfg.prompt = std::string(700, 'a');
        cfg.n_ctx = 1024;
        cfg.n_predict = 8;
        cfg.agent_state_cache_path = checkpoint;
        std::string error;
        auto session = bmoe::Session::open(bmoe::session_config_from(cfg), error);
        bmoe::GenerateRequest req;
        req.prompt = cfg.prompt;
        req.n_predict = cfg.n_predict;
        const bmoe::RunResult cold = session ? session->generate(req) : bmoe::RunResult{};
        const bmoe::RunResult warm = session ? session->generate(req) : bmoe::RunResult{};
        session.reset();
        auto reopened = bmoe::Session::open(bmoe::session_config_from(cfg), error);
        const bmoe::RunResult persisted = reopened ? reopened->generate(req) : bmoe::RunResult{};
        reopened.reset();
        {
            std::fstream file(checkpoint, std::ios::binary | std::ios::in | std::ios::out);
            file.seekg(-1, std::ios::end);
            char byte = 0;
            file.read(&byte, 1);
            file.seekp(-1, std::ios::end);
            byte ^= 1;
            file.write(&byte, 1);
        }
        auto after_corruption = bmoe::Session::open(bmoe::session_config_from(cfg), error);
        const bmoe::RunResult rejected = after_corruption ? after_corruption->generate(req) : bmoe::RunResult{};
        req.prompt[0] = 'b';
        const bmoe::RunResult mismatch = after_corruption ? after_corruption->generate(req) : bmoe::RunResult{};
        after_corruption.reset();
        req.prompt = cfg.prompt;
        cfg.n_threads = 3; // different execution configuration invalidates the saved state
        auto changed_config = bmoe::Session::open(bmoe::session_config_from(cfg), error);
        const bmoe::RunResult incompatible = changed_config ? changed_config->generate(req) : bmoe::RunResult{};
        std::filesystem::remove(checkpoint, ec);
        if (!cold || !warm || !persisted || !rejected || !mismatch || !incompatible ||
            cold.generated_text != warm.generated_text || cold.generated_text != persisted.generated_text ||
            cold.summary.n_prompt <= warm.summary.n_prompt || cold.summary.n_prompt <= persisted.summary.n_prompt ||
            rejected.summary.n_prompt != cold.summary.n_prompt || mismatch.summary.n_prompt != cold.summary.n_prompt ||
            incompatible.summary.n_prompt != cold.summary.n_prompt) {
            std::fprintf(stderr,
                         "agent state cache mismatch: error=%s, prefix work cold/warm/disk/miss/config=%d/%d/%d/%d/%d, "
                         "output equality cold-warm/cold-disk=%d/%d\n",
                         error.c_str(), cold.summary.n_prompt, warm.summary.n_prompt, persisted.summary.n_prompt,
                         mismatch.summary.n_prompt, incompatible.summary.n_prompt,
                         cold.generated_text == warm.generated_text, cold.generated_text == persisted.generated_text);
            ok = false;
        }
    }
    if (ok) std::puts("layer-tiled metadata and available-backend generated output match; logits within 1e-6");
    return ok ? 0 : 1;
}
