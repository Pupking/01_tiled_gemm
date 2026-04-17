add_executable(gemm_bench
    main.cu
    naive_gemm.cu
    tiled_gemm.cu
)

target_link_libraries(gemm_bench PRIVATE portfolio_common)

# Layer 0 only needs cuBLAS (already in portfolio_common).

