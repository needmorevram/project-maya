// src/kernels/cuda/exl3_dense.cu - the plain weights an exl3::Mat also carries (exl3_dense.cuh): DeepSeek's FP8 E4M3
// with a UE8M0 scale per 32 x 32 block, and BF16, stored as a torch Linear's [n][k].
//
//   * mv: a block owns 32 outputs (4 a warp), its activations' inputs 256 at a time in shared memory; lane l takes
//     inputs [8 l, 8 l + 8) of each 256 of its rows (one 8- or 16-byte load a row; the last 256 may be short, k a
//     multiple of 32), F32 FMAs, the warp's sum by shuffles - each output's arithmetic the same whatever the job's other
//     activations (a window's token gets the token path's bits);
//   * reconstruct_rows: the weight decoded to F16 / BF16 rows for the prompt path's GEMMs.
// The FP8 codes are decoded with integer arithmetic (no cuda_fp8.h: the same code on every GPU and on HIP).
#include "exl3_dense.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace strata::kernels::exl3::dense {
namespace {

constexpr int kThreads = 256, kWarps = 8, kRowsW = 4, kRowsB = kWarps * kRowsW, kTile = 256;
constexpr int kMaxUnits = 16;

__device__ __forceinline__ float e4m3(uint32_t b) {   // (no NaN in a weight)
    const uint32_t e = (b >> 3) & 0xfu, m = b & 7u;
    const float v = e != 0u ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float) m * 0.001953125f;
    return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}
__device__ __forceinline__ float e8m0(uint32_t b) { return b != 0u ? __uint_as_float(b << 23) : 5.877471754e-39f; }
__device__ __forceinline__ float bf16f(uint32_t h) { return __uint_as_float(h << 16); }

struct DJob {
    const uint8_t* w;
    const uint8_t* sc;
    const float* x;
    float* y;
    const float* bias;
    float alpha;
    int k, n, nc;
    size_t x_cs, y_cs;
};
struct DBatch {
    DJob j[kMaxUnits];
    int blk_end[kMaxUnits];
    int n;
    const int* skip;
};

// a row's 8 weights at inputs [i0, i0 + 8)
template <bool FP8>
__device__ __forceinline__ void row8(const DJob& J, int row, int i0, float w[8]) {
    if (FP8) {
        const uint2 q = *(const uint2*) (J.w + (size_t) row * J.k + i0);
        const float s = e8m0(J.sc[(size_t) (row >> 5) * (J.k >> 5) + (i0 >> 5)]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = e4m3((q.x >> (8 * i)) & 0xffu) * s;
            w[4 + i] = e4m3((q.y >> (8 * i)) & 0xffu) * s;
        }
    } else {
        const uint4 q = *(const uint4*) ((const uint16_t*) J.w + (size_t) row * J.k + i0);
        const uint32_t u[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[2 * i] = bf16f(u[i] & 0xffffu);
            w[2 * i + 1] = bf16f(u[i] >> 16);
        }
    }
}

template <int NC, bool FP8>
__global__ void __launch_bounds__(kThreads) mv_kernel(const DBatch b) {
    __shared__ __align__(16) float xs[NC][kTile];
    if (b.skip != nullptr && *b.skip) return;
    int ji = 0;
    while ((int) blockIdx.x >= b.blk_end[ji]) ++ji;
    const DJob& J = b.j[ji];
    const int blk = (int) blockIdx.x - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blk * kRowsB + warp * kRowsW;
    float acc[kRowsW][NC];
#pragma unroll
    for (int r = 0; r < kRowsW; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
    for (int k0 = 0; k0 < J.k; k0 += kTile) {
        const int kt = min(kTile, J.k - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < NC * kTile; i += kThreads) {
            const int c = i / kTile, q = i % kTile;
            if (q < kt) xs[c][q] = J.x[(size_t) c * J.x_cs + k0 + q];
        }
        __syncthreads();
        if (lane * 8 >= kt) continue;   // (a short last 256: the lanes past it add nothing)
        float xv[NC][8];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float4 a = *(const float4*) &xs[c][lane * 8], z = *(const float4*) &xs[c][lane * 8 + 4];
            xv[c][0] = a.x, xv[c][1] = a.y, xv[c][2] = a.z, xv[c][3] = a.w;
            xv[c][4] = z.x, xv[c][5] = z.y, xv[c][6] = z.z, xv[c][7] = z.w;
        }
#pragma unroll
        for (int r = 0; r < kRowsW; ++r) {
            float w[8];
            row8<FP8>(J, row0 + r, k0 + lane * 8, w);
#pragma unroll
            for (int c = 0; c < NC; ++c)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][c] = __fmaf_rn(w[i], xv[c][i], acc[r][c]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsW; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) {
                const int row = row0 + r;
                J.y[(size_t) c * J.y_cs + row] = __fmaf_rn(J.alpha, v, J.bias != nullptr ? J.bias[row] : 0.0f);
            }
        }
}

template <bool FP8>
__global__ void __launch_bounds__(kThreads) rows_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                                                        int k, size_t total8, uint16_t* __restrict__ dst, bool bf16) {
    const size_t at = (size_t) blockIdx.x * kThreads + threadIdx.x;   // 8 values: row at * 8 / k, input at * 8 % k
    if (at >= total8) return;
    const int row = (int) (at * 8 / (size_t) k), i0 = (int) (at * 8 % (size_t) k);
    DJob J{};
    J.w = w;
    J.sc = sc;
    J.k = k;
    float v[8];
    row8<FP8>(J, row, i0, v);
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t lo, hi;
        if (bf16) {   // (round to nearest even; the FP8 values and BF16 inputs are exact)
            const uint32_t a = __float_as_uint(v[2 * i]), c = __float_as_uint(v[2 * i + 1]);
            lo = (a + 0x7fffu + ((a >> 16) & 1u)) >> 16;
            hi = (c + 0x7fffu + ((c >> 16) & 1u)) >> 16;
        } else {
            lo = __half_as_ushort(__float2half_rn(v[2 * i]));
            hi = __half_as_ushort(__float2half_rn(v[2 * i + 1]));
        }
        o[i] = lo | (hi << 16);
    }
    *(uint4*) (dst + at * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

template <int NC>
void launch_mv(const DBatch& b, int blocks, bool fp8, cudaStream_t s) {
    if (fp8) mv_kernel<NC, true><<<blocks, kThreads, 0, s>>>(b);
    else mv_kernel<NC, false><<<blocks, kThreads, 0, s>>>(b);
}

}  // namespace

bool supported(const Mat& m) {
    return (m.fmt == kFmtFP8 || m.fmt == kFmtBF16) && m.w != nullptr && (m.fmt != kFmtFP8 || m.sc != nullptr) &&
           m.k > 0 && m.n > 0 && m.k % 32 == 0 && m.n % kRowsB == 0;
}

Mat view(const Mat& m, int c0, int nc) {
    Mat v = m;
    const size_t eb = m.fmt == kFmtFP8 ? 1 : 2;
    v.w = (const uint8_t*) m.w + (size_t) c0 * m.k * eb;
    if (m.sc != nullptr) v.sc = m.sc + (size_t) (c0 / 32) * (m.k / 32);
    v.n = nc;
    return v;
}

bool mv(const MvJob* jobs, int n, cudaStream_t s, const int* skip) {
    // launches of one format and one activation count, as many jobs as a batch takes
    bool done[kMaxMvJobs * kMaxMvCols] = {};
    struct Unit {
        int j, c0, nc;
    };
    Unit units[kMaxMvJobs * ((kMaxMvCols + kMaxCols - 1) / kMaxCols)];
    int nu = 0;
    for (int j = 0; j < n; ++j) {
        const Mat& m = *jobs[j].m;
        if (!dense::supported(m) || jobs[j].xf == nullptr || jobs[j].y == nullptr || jobs[j].ncol < 1 ||
            jobs[j].ncol > kMaxMvCols) {
            std::fprintf(stderr, "exl3 dense mv: job %d unsupported (fmt %d k %d n %d, %d activations%s)\n", j, m.fmt,
                         m.k, m.n, jobs[j].ncol, jobs[j].xf == nullptr ? ", q8_1 input" : "");
            return false;
        }
        for (int c0 = 0; c0 < jobs[j].ncol; c0 += kMaxCols)
            units[nu++] = {j, c0, std::min(kMaxCols, jobs[j].ncol - c0)};
    }
    for (int i = 0; i < nu; ++i) {
        if (done[i]) continue;
        const int fmt = jobs[units[i].j].m->fmt, nc = units[i].nc;
        DBatch b{};
        int blocks = 0;
        for (int u = i; u < nu && b.n < kMaxUnits; ++u) {
            const MvJob& job = jobs[units[u].j];
            if (done[u] || job.m->fmt != fmt || units[u].nc != nc) continue;
            done[u] = true;
            const Mat& m = *job.m;
            DJob& J = b.j[b.n];
            J.w = (const uint8_t*) m.w;
            J.sc = m.sc;
            J.x = job.xf + units[u].c0 * job.xf_cs;
            J.y = job.y + units[u].c0 * job.y_cs;
            J.bias = job.bias;
            J.alpha = job.alpha;
            J.k = m.k;
            J.n = m.n;
            J.nc = nc;
            J.x_cs = job.xf_cs;
            J.y_cs = job.y_cs;
            blocks += m.n / kRowsB;
            b.blk_end[b.n++] = blocks;
        }
        b.skip = skip;
        const bool fp8 = fmt == kFmtFP8;
        switch (nc) {
            case 1: launch_mv<1>(b, blocks, fp8, s); break;
            case 2: launch_mv<2>(b, blocks, fp8, s); break;
            case 3: launch_mv<3>(b, blocks, fp8, s); break;
            case 4: launch_mv<4>(b, blocks, fp8, s); break;
            case 5: launch_mv<5>(b, blocks, fp8, s); break;
            case 6: launch_mv<6>(b, blocks, fp8, s); break;
            case 7: launch_mv<7>(b, blocks, fp8, s); break;
            default: launch_mv<8>(b, blocks, fp8, s); break;
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

bool reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
    if (!dense::supported(m)) return false;
    const size_t total8 = (size_t) m.n * m.k / 8;
    const unsigned blocks = (unsigned) ((total8 + kThreads - 1) / kThreads);
    if (m.fmt == kFmtFP8) rows_kernel<true><<<blocks, kThreads, 0, s>>>((const uint8_t*) m.w, m.sc, m.k, total8, dst, bf16);
    else rows_kernel<false><<<blocks, kThreads, 0, s>>>((const uint8_t*) m.w, nullptr, m.k, total8, dst, bf16);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels::exl3::dense
