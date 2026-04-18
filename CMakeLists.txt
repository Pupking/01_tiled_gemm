add_executable(gemm_bench
    main.cu
    00_naive.cu
    01_tiled.cu
    02_tiled_coalesced.cu
    03_regblock.cu
    04_warp_rebalance.cu
)



target_link_libraries(gemm_bench PRIVATE portfolio_common)

# Layer 0 only needs cuBLAS (already in portfolio_common).

