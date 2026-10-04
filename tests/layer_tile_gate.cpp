#include "bmoe/runtime.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

static bool check_case(llama_model * model, int n_tokens) {
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
        llama_decode_layer_tiled(tiled.get(), llama_batch_get_one(tokens.data(), n_tokens), 512) != 0) {
        std::fprintf(stderr, "tiled decode failed at %d tokens\n", n_tokens);
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

    llama_token next = 15;
    if (llama_decode(reference.get(), llama_batch_get_one(&next, 1)) != 0 ||
        llama_decode(tiled.get(), llama_batch_get_one(&next, 1)) != 0) {
        std::fprintf(stderr, "post-tile decode failed at %d tokens\n", n_tokens);
        return false;
    }
    if (std::memcmp(llama_get_logits(reference.get()), llama_get_logits(tiled.get()), logit_bytes) != 0) {
        const float * a = llama_get_logits(reference.get());
        const float * b = llama_get_logits(tiled.get());
        float max_delta = 0.0f;
        for (size_t i = 0; i < n_vocab; ++i)
            max_delta = std::max(max_delta, std::fabs(a[i] - b[i]));
        std::fprintf(stderr, "post-tile decode differs at %d tokens: max delta=%g\n", n_tokens, max_delta);
        if (max_delta > 1.0e-6f) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], params),
                                                                    llama_model_free);
    if (!model) return 1;
    bool ok = check_case(model.get(), 1) && check_case(model.get(), 64) && check_case(model.get(), 65) &&
              check_case(model.get(), 130) && check_case(model.get(), 520);
    model.reset();
    llama_backend_free();
    if (ok) {
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
        ok = reference && tiled && reference.generated_text == tiled.generated_text;
        if (!ok) {
            std::fprintf(stderr, "streamed generated output differs: reference='%s' tiled='%s'; errors='%s'/'%s'\n",
                         reference.generated_text.c_str(), tiled.generated_text.c_str(), reference.error.c_str(),
                         tiled.error.c_str());
        }
    }
    if (ok) std::puts("layer-tiled metadata and generated output match; logits within 1e-6");
    return ok ? 0 : 1;
}
