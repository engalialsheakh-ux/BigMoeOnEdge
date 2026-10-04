# Byte-identity gates. The tiny synthetic models are generated at test time by
# scripts/make-tiny-moe.py (requires Python with gguf and numpy). One model per expert
# layout the streamer supports: qwen3moe (split gate/up/down), gemma4 (fused gate_up) and
# nemotron_h_moe (gate-less up/down).

add_executable(bmoe_moe_gates moe_gates.cpp)
target_link_libraries(bmoe_moe_gates PRIVATE bmoe_core)
add_executable(bmoe_hybrid_gpu_gate hybrid_gpu_gate.cpp)
target_link_libraries(bmoe_hybrid_gpu_gate PRIVATE bmoe_core)
# G16 probes the loopback rpc-server port before registering it.
if(WIN32)
    target_link_libraries(bmoe_moe_gates PRIVATE ws2_32)
endif()

# Explicit opt-in: this local integration gate compares resident and streamed execution on
# the actual Qwen3.6 hybrid attention/SSM model with the same CUDA and KV placement.
if(DEFINED ENV{BMOE_QWEN36_MODEL} AND EXISTS "$ENV{BMOE_QWEN36_MODEL}")
    add_test(NAME hybrid_gpu_qwen36
             COMMAND bmoe_hybrid_gpu_gate "$ENV{BMOE_QWEN36_MODEL}" 3000 kv-gpu)
    set_tests_properties(hybrid_gpu_qwen36 PROPERTIES LABELS "integration;gpu")
endif()
target_include_directories(bmoe_moe_gates PRIVATE ${CMAKE_SOURCE_DIR}/core/src/moe)

# Reasoning-parser wiring regression (issue #49). Needs no model — it drives the extracted
# bmoe::detail::build_parse_params directly — so it runs unconditionally, unlike the gates below.
add_executable(bmoe_chat_parse_test chat_parse_test.cpp)
target_link_libraries(bmoe_chat_parse_test PRIVATE bmoe_core)
target_include_directories(bmoe_chat_parse_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/engine)
target_compile_definitions(bmoe_chat_parse_test PRIVATE
    BMOE_QWEN3_TEMPLATE_PATH="${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates/Qwen3.5-4B.jinja")
add_test(NAME chat_parse COMMAND bmoe_chat_parse_test)

# Thinking-control tests (issue #82). Drives bmoe::detail::probe_think_control and the prefill
# against real vendored templates, so it needs no model and runs unconditionally.
add_executable(bmoe_think_control_test think_control_test.cpp)
target_link_libraries(bmoe_think_control_test PRIVATE bmoe_core)
target_include_directories(bmoe_think_control_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/engine)
set(_tmpl_dir "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates")
target_compile_definitions(bmoe_think_control_test PRIVATE
    BMOE_TMPL_QWEN3="${_tmpl_dir}/Qwen3.5-4B.jinja"
    BMOE_TMPL_LFM25="${_tmpl_dir}/LFM2.5-8B-A1B.jinja"
    BMOE_TMPL_GPTOSS="${_tmpl_dir}/openai-gpt-oss-120b.jinja"
    BMOE_TMPL_LFM2="${_tmpl_dir}/LFM2-8B-A1B.jinja"
    BMOE_TMPL_LFM25_INSTRUCT="${_tmpl_dir}/LFM2.5-Instruct.jinja"
    BMOE_TMPL_GEMMA4="${_tmpl_dir}/google-gemma-4-31B-it.jinja")
add_test(NAME think_control COMMAND bmoe_think_control_test)

# validate() unit tests (issue #51 + the pre-existing rules). Pure config, no model — runs
# unconditionally, and makes good on config.h's "unit-tested" claim.
add_executable(bmoe_config_test config_test.cpp)
target_link_libraries(bmoe_config_test PRIVATE bmoe_core)
add_test(NAME config_validate COMMAND bmoe_config_test)

# release_file_mappings: on Windows it must find and drop this process's own view + section of a
# file and leave the placeholders in place; on POSIX it must report unsupported and touch nothing.
add_executable(bmoe_mapping_release_test mapping_release_test.cpp)
target_link_libraries(bmoe_mapping_release_test PRIVATE bmoe_core)
target_include_directories(bmoe_mapping_release_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/io)
add_test(NAME mapping_release COMMAND bmoe_mapping_release_test)

# The n-gram draft matcher. Pure policy over token ids (no llama.cpp, no model), so the drafting
# decision is covered exhaustively here; what it costs is a bench question, not a test.
add_executable(bmoe_ngram_test ngram_test.cpp)
target_link_libraries(bmoe_ngram_test PRIVATE bmoe_core)
add_test(NAME ngram_draft COMMAND bmoe_ngram_test)

# The interval-union stall clock. Injected timestamps, no wall clock, so overlap/nesting/
# separation and the open-interval snapshot rule are verified exactly (issue #98).
add_executable(bmoe_stall_union_test stall_union_test.cpp)
target_link_libraries(bmoe_stall_union_test PRIVATE bmoe_core)
target_include_directories(bmoe_stall_union_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/moe)
add_test(NAME stall_union COMMAND bmoe_stall_union_test)

# The CSV `# summary` trailer serialization. Drives the sink directly with a filled RunSummary —
# no model, no llama.cpp — so the emitted key names and units are pinned by a test rather than by
# whatever script last read them. The prefill keys must match BMOE_DONE's, token for token.
add_executable(bmoe_csv_summary_test csv_summary_test.cpp)
target_link_libraries(bmoe_csv_summary_test PRIVATE bmoe_core)
add_test(NAME csv_summary COMMAND bmoe_csv_summary_test)

# decide()'s policy (the prompt split, choice checks and scoring, when the prefix state is
# restored, stored or left alone) over a scripted fake backend. No model, no llama.cpp, so it runs
# unconditionally; the same path on a real model is gate G18.
add_executable(bmoe_decide_test decide_test.cpp)
target_link_libraries(bmoe_decide_test PRIVATE bmoe_core)
target_include_directories(bmoe_decide_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/engine/decide)
add_test(NAME decide_policy COMMAND bmoe_decide_test)

foreach(_test_target IN ITEMS
        bmoe_moe_gates bmoe_hybrid_gpu_gate bmoe_chat_parse_test bmoe_think_control_test
        bmoe_config_test bmoe_mapping_release_test bmoe_ngram_test bmoe_stall_union_test
        bmoe_csv_summary_test bmoe_decide_test)
    bmoe_stage_runtime_dlls(${_test_target})
endforeach()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import gguf, numpy"
    RESULT_VARIABLE _python_deps_result ERROR_VARIABLE _python_deps_error)
if(NOT _python_deps_result EQUAL 0)
    message(FATAL_ERROR "MoE fixture tests require gguf and numpy in ${Python3_EXECUTABLE}. "
        "Install with: py -m pip install gguf numpy\n${_python_deps_error}")
endif()

# arch → (fixture id, gguf filename). Each gets a generate fixture + a gate run.
foreach(spec "qwen3moe" "gemma4" "nemotron_h_moe")
    set(_model "${CMAKE_CURRENT_BINARY_DIR}/tiny-moe-${spec}.gguf")
    set(_fixture "TINY_MOE_MODEL_${spec}")

    add_test(
        NAME moe_make_tiny_model_${spec}
        COMMAND "${Python3_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/scripts/make-tiny-moe.py --arch ${spec} --out ${_model}
    )
    set_tests_properties(moe_make_tiny_model_${spec} PROPERTIES FIXTURES_SETUP ${_fixture})

    add_test(NAME moe_gates_${spec} COMMAND bmoe_moe_gates ${_model})
    set_tests_properties(moe_gates_${spec} PROPERTIES FIXTURES_REQUIRED ${_fixture})
    if(spec STREQUAL "qwen3moe")
        add_test(NAME hybrid_gpu_gate COMMAND bmoe_hybrid_gpu_gate ${_model})
        set_tests_properties(hybrid_gpu_gate PROPERTIES FIXTURES_REQUIRED ${_fixture} SKIP_RETURN_CODE 77)
    endif()
endforeach()

# The same qwen3moe fixture, written as a 4-shard split gguf (metadata-only first shard —
# the layout large models actually ship in, since Hugging Face rejects files above 50 GB).
# 51 tensors at 20 per shard is deterministic: -00001-of-00004 .. -00004-of-00004. The gate
# binary gets the FIRST shard, exactly as a user would pass it; the streamer must resolve
# every expert tensor to its (shard, offset) and produce byte-identical output.
add_test(
    NAME moe_make_tiny_model_split
    COMMAND "${Python3_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/scripts/make-tiny-moe.py --arch qwen3moe
            --split-max-tensors 20 --out "${CMAKE_CURRENT_BINARY_DIR}/tiny-moe-split.gguf"
)
set_tests_properties(moe_make_tiny_model_split PROPERTIES FIXTURES_SETUP TINY_MOE_MODEL_split)

add_test(NAME moe_gates_split
         COMMAND bmoe_moe_gates "${CMAKE_CURRENT_BINARY_DIR}/tiny-moe-split-00001-of-00004.gguf")
set_tests_properties(moe_gates_split PROPERTIES FIXTURES_REQUIRED TINY_MOE_MODEL_split)
