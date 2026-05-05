cmake_minimum_required(VERSION 3.20)

# RTX 3050 Laptop (GA107) = sm_86. Override via -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(tiled_gemm
    LANGUAGES CXX CUDA
    DESCRIPTION "Tiled GEMM(FP32) multi optimization stepped kernels"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

add_library(commonLibs INTERFACE)
target_include_directories(commonLibs INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(commonLibs INTERFACE
    CUDA::cudart
    CUDA::cublas
)


add_executable(gemm_bench
    main.cu
    00_naive.cu
    01_tiled.cu
    02_tiled_coalesced.cu
    03_regblock.cu
    04_warp_rebalance.cu
    05_bank_pad_vec.cu
    06_k32_tpb4.cu
    07_sweep.cu
    08_splitk_z.cu
    09_multibuf.cu
    10_multibuf_sweep.cu
    11_tile128.cu
    12_wider_n.cu
    13_ldg_async.cu
    # Twelve trait-based kernels collapsed into one TU. Each is a
    # (Traits, launcher) pair over the shared tiled_kernel template.
    trait_based_kernels.cu
)



target_link_libraries(gemm_bench PRIVATE commonLibs)

