# ==================================================================
# C3/CMakeLists.txt — C3 JIT 编译器独立仓库顶层构建入口
#
# 目的：C3 物理独立（CGO2027 目标）的仓库顶层。
#   · 本文件把 include/C3（头文件 + .td）+ src/C3（源文件）定义为一个
#     可独立 configure / build 的 CMake 工程。
#   · 依赖说明（宿主侧，C3 保留直用）：
#       - C3 头文件直接 include Tensor.h / Ctools.h / AutoGrad/Node.h 等宿主头，
#         这是「物理独立 + 宿主类型直用」的设计（与 XLA/Triton 类似），
#         不做 C3Host 抽象（见 Phase 1/2 规划）。
#       - 因此 C3 仓库本身不携带宿主头；独立构建 / 被宿主集成时，宿主 include
#         路径通过 C3_HOST_INCLUDE_DIRS 传入（CTorch 通过 add_subdirectory(c3)
#         集成时传入其 include/ 与 src/ 目录）。
#   · MLIR/LLVM 通过 find_package(CONFIG) 获取，默认指向 Homebrew llvm。
#
# 生成：2026-08-17（由 CTorch 仓库 c3/CMakeLists.txt 物理拆分而来）
# ==================================================================

cmake_minimum_required(VERSION 3.26)
project(C3 VERSION 0.5.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 编译优化（与宿主工程保持一致：-O3 -ffast-math -march=native + ThinLTO）----
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
option(C3_ENABLE_LTO "Enable Link-Time Optimization (Thin LTO)" ON)
if(C3_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")
endif()

# ---- 宿主 include 目录（Tensor/Ctools/Node 等）----
# 独立构建时由用户/父项目传入；C3 仓库不携带宿主头。
set(C3_HOST_INCLUDE_DIRS
        ""
        CACHE STRING "Host header search paths (CTorch Tensor/Ctools/Node etc.)")

# ---- MLIR / LLVM (JIT 2.x / 3.0 后端) ----
option(CT_ENABLE_MLIR "Enable MLIR/LLVM JIT backend for C3" ON)
if(CT_ENABLE_MLIR)
    set(LLVM_DIR /opt/homebrew/opt/llvm/lib/cmake/llvm CACHE PATH "LLVM CMake config dir")
    set(MLIR_DIR /opt/homebrew/opt/llvm/lib/cmake/mlir CACHE PATH "MLIR CMake config dir")
    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)

    # C3 Dialect TableGen 自动生成（C3Ops.td / C3Combine.td 位于 include/C3/）
    # 注意：-I 必须指向 include/ 上层目录（.td 内以 C3/C3Ops.td 相对引用）
    set(C3_TD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(C3_TD_SRC "${C3_TD_DIR}/C3/C3Ops.td")
    set(LLVM_TARGET_DEFINITIONS ${C3_TD_SRC})
    list(APPEND MLIR_TABLEGEN_FLAGS "-I${MLIR_INCLUDE_DIRS}")
    list(APPEND MLIR_TABLEGEN_FLAGS "-I${LLVM_INCLUDE_DIRS}")

    mlir_tablegen(C3Ops.h.inc -gen-op-decls "-I${MLIR_INCLUDE_DIRS}" "-I${C3_TD_DIR}")
    mlir_tablegen(C3Ops.cpp.inc -gen-op-defs "-I${MLIR_INCLUDE_DIRS}" "-I${C3_TD_DIR}")
    mlir_tablegen(C3Dialect.h.inc -gen-dialect-decls -dialect=c3 "-I${MLIR_INCLUDE_DIRS}" "-I${C3_TD_DIR}")
    mlir_tablegen(C3Dialect.cpp.inc -gen-dialect-defs -dialect=c3 "-I${MLIR_INCLUDE_DIRS}" "-I${C3_TD_DIR}")

    set(LLVM_TARGET_DEFINITIONS "${C3_TD_DIR}/C3/C3Combine.td")
    mlir_tablegen(C3Combine.cpp.inc -gen-rewriters "-I${MLIR_INCLUDE_DIRS}" "-I${C3_TD_DIR}")

    add_public_tablegen_target(C3Gen)
endif()

# ---- 海光 DCU 后端（默认 OFF，需 DTK 24.04+ 节点）----
option(CT_ENABLE_DCU "Enable Hygon DCU (gfx906) backend for C3" OFF)
set(C3-DCU-Sources)
if(CT_ENABLE_DCU)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/dtk" CACHE STRING "Default ROCM installation directory")
    endif()
    set(GCVM_INCLUDE_DIR "${ROCM_PATH}/dcc/gcvm/include")
    set(GCVM_LIBRARY "${ROCM_PATH}/dcc/gcvm/lib/libgcvm.so")
    if(NOT EXISTS "${GCVM_INCLUDE_DIR}")
        message(FATAL_ERROR "GCVM include directory not found: ${GCVM_INCLUDE_DIR}. DTK 24.04+ required.")
    endif()
    set(C3-DCU-Sources
            ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/GCVMBridge.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/DCUCompiledKernel.cpp)
endif()

# ---- 源文件收集 ----
set(C3-Core-Sources
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/Graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/C3Engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/GraphMerger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/PGOManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/C3BackwardCapture.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/C3KernelRegistry.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/RollingHash.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/RegionFusionRegistry.cpp)

set(C3-MLIR-Sources
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/MLIRKernelGen.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/JITCache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/MLIRToLLVMIR.cpp)
if(CT_ENABLE_MLIR)
    list(APPEND C3-MLIR-Sources
            ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/C3Dialect.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/C3DialectLowering.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/LinalgElementwiseGen.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/LinalgFusedGen.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/C3/LinalgOneShotGen.cpp)
endif()

# ==================================================================
# C3Core — C3 核心库（OBJECT 库）
# 编译单元独立；链接阶段由消费方（CTorch 或独立单测）合并宿主符号。
# ==================================================================
add_library(C3Core OBJECT
        ${C3-Core-Sources}
        ${C3-MLIR-Sources}
        ${C3-DCU-Sources})

target_include_directories(C3Core
        PUBLIC
        ${C3_HOST_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CT_ENABLE_MLIR)
    target_include_directories(C3Core SYSTEM PUBLIC
            ${LLVM_INCLUDE_DIRS}
            ${MLIR_INCLUDE_DIRS}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(C3Core PUBLIC CT_ENABLE_MLIR=1)
    add_dependencies(C3Core C3Gen)
endif()

if(CT_ENABLE_DCU)
    target_include_directories(C3Core SYSTEM PUBLIC ${GCVM_INCLUDE_DIR})
    target_compile_definitions(C3Core PUBLIC CT_ENABLE_DCU=1 WITH_DCU=1)
endif()

# ==================================================================
# 独立构建自检：仅编译 C3Core（OBJECT 库无需链接宿主符号）
# 用法（需先设置 C3_HOST_INCLUDE_DIRS 指向宿主 include/src）：
#   cmake -S . -B build -DCT_ENABLE_MLIR=ON -DC3_HOST_INCLUDE_DIRS=<host include>;<host src>
#   cmake --build build --target C3Core -j
# ==================================================================
