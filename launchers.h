#pragma once

// Single source of truth for every variant's launch entry point.
// Each kernel TU defines its launcher and includes this header so signatures
// can't drift between definition and the registry in main.cu.

#include "gemm_common.h"

void naive_launch(const GemmParams& p);
void tiled_launch(const GemmParams& p);
void tiled_coalesced_launch(const GemmParams& p);
void regblock_launch(const GemmParams& p);
void warp_rebalance_launch(const GemmParams& p);
void bank_pad_vec_launch(const GemmParams& p);
void gmem_vec_launch(const GemmParams& p);
void tpb4_launch(const GemmParams& p);
void k32_tpb4_launch(const GemmParams& p);

void sweep_A_launch(const GemmParams& p);
void sweep_B_launch(const GemmParams& p);
void sweep_C_launch(const GemmParams& p);
void sweep_D_launch(const GemmParams& p);
void sweep_E_launch(const GemmParams& p);
void sweep_F_launch(const GemmParams& p);

void splitk1_launch(const GemmParams& p);
void splitk2_launch(const GemmParams& p);
void splitk3_launch(const GemmParams& p);
void splitk4_launch(const GemmParams& p);

void multibuf2_launch(const GemmParams& p);
void multibuf_b2_launch(const GemmParams& p);
void multibuf_b3_launch(const GemmParams& p);
void multibuf_b4_launch(const GemmParams& p);

void wider_m_launch(const GemmParams& p);
void tile128_launch(const GemmParams& p);
void wider_n_launch(const GemmParams& p);
void supertile_launch(const GemmParams& p);
void drop_oob_launch(const GemmParams& p);
void super_wider_launch(const GemmParams& p);

void ldg_vec_launch(const GemmParams& p);
void ldg_super_launch(const GemmParams& p);
void ldg_async_launch(const GemmParams& p);
void ldg_super_big_launch(const GemmParams& p);
void ldg_k128_launch(const GemmParams& p);
void ldg_wider2_launch(const GemmParams& p);
void ldg_max_launch(const GemmParams& p);
