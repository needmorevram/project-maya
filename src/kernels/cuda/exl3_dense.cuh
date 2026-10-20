// src/kernels/cuda/exl3_dense.cuh - the plain weights exl3::Mat also carries (kFmtFP8, kFmtBF16): exl3.cu's mv, view,
// supported, mv_cols and reconstruct_rows hand such a matrix to these (exl3_dense.cu).  k a multiple of 32, n of 32.
#pragma once

#include "strata/kernels/exl3.hpp"

namespace strata::kernels::exl3::dense {

/// The activations a job's weights are read once for (mv_cols).
constexpr int kMaxCols = 8;

bool supported(const Mat& m);
/// The rows [c0, c0 + nc) of W (outputs; c0, nc multiples of 32).
Mat view(const Mat& m, int c0, int nc);
/// mv's jobs of plain matrices (every one of them: the caller hands EXL3 jobs to its own kernels); skip: set_skip's
/// flag (null: none).
bool mv(const MvJob* jobs, int n, cudaStream_t s, const int* skip);
/// dst[r][c] = W[r][c] (n rows of k values), F16 or BF16 bits.
bool reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s);

}  // namespace strata::kernels::exl3::dense
