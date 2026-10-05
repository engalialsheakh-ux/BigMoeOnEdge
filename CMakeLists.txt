# BigMoeOnEdge — top-level build.
#
# The engine (`bmoe_core`) depends only on its own public headers under include/bmoe
# and on llama.cpp's PUBLIC C API (llama.h / ggml.h). The serial streaming path never
# touches llama.cpp internals: MoE expert streaming is driven entirely through the public
# eval-callback and public gguf accessors, so it tracks upstream with nothing more than a
# submodule pointer bump — no in-tree patch. The released exception is the optional `--overlap`
# feature, which needs a per-expert wait point the public API does not expose: the submodule
# pins a fork branch (`bmoe/expert-ready-hook` on Helldez/llama.cpp) adding one ~25-line
# readiness hook, detected here as BMOE_HAVE_EXPERT_READY_HOOK. It is zero-cost when absent,
# the serial path still builds against stock upstream, and it sunsets the moment upstream
# ships an equivalent callback. The pinned fork also carries opt-in mixed CPU/GPU expert
# execution and quantized CUDA FlashAttention; see docs/seam.md for the exact seam.
#
# llama.cpp is an optional dependency at configure time: when the submodule is absent
# only the pure-policy code (config validation) compiles, so the scaffold and a subset
# of CI stay green before the native dependency is fetched.
cmake_minimum_required(VERSION 3.21)

# The version is declared here and nowhere else in the build: the engine reports it (bmoe-cli
# --version, and the run-parameter preamble of every metrics CSV), so a committed benchmark file
# says which engine produced it. Keep it in step with CHANGELOG.md; the Android app version
# advances separately when an app release includes the change.
project(bigmoeonedge VERSION 0.34.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BMOE_BUILD_TESTS "Build byte-identity gates and unit tests" ON)
option(BMOE_BUILD_CLI   "Build the bmoe-cli host tool"              ON)
# Off by default: diagnostics, not part of the product. They need no llama.cpp, so they can be
# configured and cross-compiled on their own without building the engine.
option(BMOE_BUILD_TOOLS "Build standalone diagnostic tools (bmoe-iobench)" OFF)

# --- Optional native dependency: llama.cpp ----------------------------------------
# The streaming seam uses only the public C API (llama.h). We additionally link llama.cpp's
# `common` layer for one thing: chat-template rendering + reasoning parsing (common_chat_*),
# so prompt formatting and thinking-extraction are template-driven per model instead of
# hardcoded. Unlike the streaming seam, `common` is NOT a stable API — it can change between
# upstream versions, so a submodule bump may need the chat glue in runtime.cpp updated (the
# gates catch it at build time). That trade-off is deliberate; see docs/seam.md.
set(BMOE_HAVE_LLAMA OFF)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    set(BMOE_HAVE_LLAMA ON)
    # Upstream examples/tests/server stay off; build the common utils we link for chat.
    set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON  ON  CACHE BOOL "" FORCE)
    set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)  # common's model-download path unused
    # The prefill-device gate (G16) needs a second device to move weights onto, and the one every
    # host has is a loopback rpc-server fronting the CPU. A test fixture, not a feature: host test
    # builds only (releases build without tests), no front-end accepts an RPC endpoint, and it is
    # not forced: -DGGML_RPC=OFF still wins, and G16 then reports itself skipped.
    if(BMOE_BUILD_TESTS AND NOT ANDROID)
        set(GGML_RPC ON CACHE BOOL "ggml: use RPC")
    endif()
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
endif()

# Keep each Windows executable self-contained for direct launches and CTest. The ggml
# backends can be loaded dynamically, so include their DLLs even when they are absent
# from the executable's transitive link dependencies.
function(bmoe_stage_runtime_dlls target)
    if(NOT WIN32)
        return()
    endif()

    set(_runtime_dlls "$<TARGET_RUNTIME_DLLS:${target}>")
    foreach(_backend IN LISTS GGML_AVAILABLE_BACKENDS)
        if(TARGET ${_backend})
            list(APPEND _runtime_dlls "$<TARGET_FILE:${_backend}>" "$<TARGET_RUNTIME_DLLS:${_backend}>")
        endif()
    endforeach()

    # FindCUDAToolkit may expose CUDA libraries as UNKNOWN imported targets, which
    # TARGET_RUNTIME_DLLS cannot resolve to their DLLs. CUDA 13 keeps them in bin/x64.
    if(TARGET ggml-cuda)
        set(_cuda_bin "${CUDAToolkit_BIN_DIR}")
        if(EXISTS "${_cuda_bin}/x64")
            string(APPEND _cuda_bin "/x64")
        endif()
        foreach(_pattern IN ITEMS "cudart64_*.dll" "cublas64_*.dll" "cublasLt64_*.dll")
            file(GLOB _matches "${_cuda_bin}/${_pattern}")
            if(NOT _matches)
                message(FATAL_ERROR "CUDA runtime ${_pattern} not found in ${_cuda_bin}")
            endif()
            list(APPEND _runtime_dlls ${_matches})
        endforeach()
    endif()

    list(REMOVE_DUPLICATES _runtime_dlls)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_runtime_dlls} "$<TARGET_FILE_DIR:${target}>"
        COMMAND_EXPAND_LISTS VERBATIM)
endfunction()

add_subdirectory(core)

if(BMOE_BUILD_CLI AND BMOE_HAVE_LLAMA)
    add_subdirectory(cli)
endif()

if(BMOE_BUILD_TESTS AND BMOE_HAVE_LLAMA)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BMOE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

message(STATUS "BigMoeOnEdge: llama.cpp present = ${BMOE_HAVE_LLAMA}")
