# Build moe_trace against an existing llama.cpp source tree.
#
#   cmake -B build -S src/capture -G Ninja \
#         -DLLAMA_CPP_DIR=/path/to/llama.cpp \
#         -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF
#   cmake --build build --config Release
#
# GGML_NATIVE=OFF is not optional (plan T0.2): a -march=native build is tied to the CPU that
# compiled it, and Kaggle does not guarantee the same host CPU between sessions. A binary that
# selects different SIMD paths across sessions breaks T3.6's cross-session byte-identity gate.
#
# CMAKE_CUDA_ARCHITECTURES=75 targets the T4. The local RTX 5060 Ti is sm_120 and its traces are a
# DIFFERENT EXPERIMENT (invariant I3) -- platform.gpu_arch is inside run_config_sha256 precisely so
# that mixing them is a hard error rather than a silent one.

cmake_minimum_required(VERSION 3.20)
project(moe_trace CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR
        "LLAMA_CPP_DIR is required: -DLLAMA_CPP_DIR=/path/to/llama.cpp\n"
        "Pin the commit and record it in BUILD_COMMIT.txt (plan T0.2). The commit is load-bearing "
        "for correctness, not only reproducibility: build_moe_ffn switched from ggml_top_k (a "
        "contiguous op) to ggml_argsort_top_k (a strided view), and the callback's read path "
        "depends on which one is in the graph.")
endif()

if(NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "no include/llama.h under LLAMA_CPP_DIR=${LLAMA_CPP_DIR}")
endif()

# Two upstream binaries are load-bearing for the plan, and at commit 7077abbe they live in
# different trees: llama-bench under tools/ (T0.5 throughput calibration) and llama-eval-callback
# under examples/ (T1.4 node-name verification). Both subdirectories are gated on
# LLAMA_BUILD_COMMON, which defaults to LLAMA_STANDALONE -- i.e. OFF for a subdirectory build like
# this one -- so enabling EXAMPLES/TOOLS alone silently yields no such targets.
#
# moe_trace links only llama + ggml, so this whole group is optional. It is worth turning OFF for a
# collection-only session (Kaggle, 12 h cap): common drags in vendor/cpp-httplib and the two tool
# trees are several minutes of compile for binaries a collection run never invokes.
option(MOE_TRACE_UPSTREAM_TOOLS "also build llama-eval-callback (T1.4) and llama-bench (T0.5)" ON)

# LLAMA_BUILD_TESTS stays OFF unconditionally: the test suite downloads a GGUF at configure time.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ${MOE_TRACE_UPSTREAM_TOOLS} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES ${MOE_TRACE_UPSTREAM_TOOLS} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ${MOE_TRACE_UPSTREAM_TOOLS} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)

add_executable(moe_trace moe_trace.cpp)
target_sources(moe_trace PRIVATE node_spec.hpp trace_writer.hpp trace_format.h)
target_include_directories(moe_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(moe_trace PRIVATE llama ggml)

# The generated header is the single source of truth for the on-disk layout. Fail the build if it is
# missing rather than letting a stale copy compile.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/trace_format.h")
    message(FATAL_ERROR
        "trace_format.h is missing. Generate it:\n"
        "  python -m src.traces.format --emit-header src/capture/trace_format.h")
endif()

install(TARGETS moe_trace RUNTIME DESTINATION bin)
