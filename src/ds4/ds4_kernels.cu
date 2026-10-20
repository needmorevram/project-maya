// src/ds4/ds4_kernels.cu - DeepSeek-V4.1's device kernels (see ds4_kernels.cuh).  No fast math here: the rotary
// angles, the softmaxes and the roundings follow the reference's F32 arithmetic.  A window's token is blockIdx.y.
#include "ds4_kernels.cuh"
#include "ds4_pool.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cctype>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <vector>
#include <chrono>
#include <atomic>
#include <immintrin.h>
#include <memory>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__device__ __forceinline__ uint16_t f2bf16(float f) {   // round to nearest even (the values here are exact anyway)
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block-wide sum for blockDim.x <= 1024 (a multiple of 32); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

// 2^ceil(log2(x)) for a positive normal x (the reference's fast_round_scale)
__device__ __forceinline__ float pow2_ceil(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}

__device__ __forceinline__ float fp8_rt(float v) {   // E4M3, nearest even (the input is already clamped)
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    const __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    return __half2float(__half(h));
}

__device__ __forceinline__ float e2m1_rt(float v) {   // E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}

// the packed FP4 rows (Finish::pdst): an E2M1 code (its sign in bit 3) <-> its value, and an E4M3 scale byte's value
__device__ __forceinline__ float fp4v(uint32_t c) {   // magnitudes 0 .5 1 1.5 2 3 4 6
    const uint32_t m = c & 7u;
    const uint32_t b = m >= 2u ? (((m >> 1) + 126u) << 23) | ((m & 1u) << 22) : m == 1u ? 126u << 23 : 0u;
    return __uint_as_float(b | ((c & 8u) << 28));
}
__device__ __forceinline__ uint32_t fp4c(float v) {   // (v on the E2M1 grid: e2m1_rt's value)
    const float a = fabsf(v);
    const uint32_t m = a < 0.25f ? 0u : a < 0.75f ? 1u : a < 1.25f ? 2u : a < 1.75f ? 3u : a < 2.5f ? 4u : a < 3.5f ? 5u
                     : a < 5.0f ? 6u : 7u;
    return m | (signbit(v) ? 8u : 0u);
}
__device__ __forceinline__ float e4m3v(uint32_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) b, __NV_E4M3);
    return __half2float(__half(h));
}

// ---------------------------------------------------------------- embed
__global__ void embed_kernel(const uint16_t* __restrict__ rows, float* __restrict__ h, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    const float v = bf16f(rows[(size_t) t * dim + e]);
    for (int s = 0; s < hc; ++s) h[((size_t) t * hc + s) * dim + e] = v;
}

// ---------------------------------------------------------------- finish
constexpr int kFinMax = 32;   // values a thread holds: len <= 32 * 256

__global__ void __launch_bounds__(kThreads) finish_kernel(float* x, int len, long long stride, Finish f) {
    float* v = x + (long long) blockIdx.x * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = rsqrtf(tot / (float) len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const int r0 = len - f.rd;
        const int pos = f.pos + (int) (blockIdx.x / f.vpp) * f.pos_step;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const float other = __shfl_xor_sync(0xffffffffu, r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = cosf(ang);
                const float sn = f.inverse ? -sinf(ang) : sinf(ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    float qsc[kFinMax];   // (quant 2 / 3: each value's group scale, for the packed row)
    if (f.quant != 0) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
                qsc[j] = s;
            } else {
                const float s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
                qsc[j] = s;
            }
        }
    }
    if (f.pdst != nullptr && (f.quant == 2 || f.quant == 3)) {   // packed: the codes two a byte, then the scales
        const int g = f.quant == 2 ? 16 : 32;
        uint8_t* pd = f.pdst + (long long) blockIdx.x * f.pdst_stride;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const uint32_t c = e < len ? fp4c(r[j] / qsc[j]) : 0u;   // (r / s: exactly the E2M1 value)
            const uint32_t hi = __shfl_down_sync(0xffffffffu, c, 1);
            if (e >= len) continue;
            if ((e & 1) == 0) pd[e >> 1] = (uint8_t) (c | (hi << 4));
            if (e % g == 0)
                pd[len / 2 + e / g] = f.quant == 2 ? (uint8_t) __nv_cvt_float_to_fp8(qsc[j], __NV_SATFINITE, __NV_E4M3)
                                                   : (uint8_t) (__float_as_uint(qsc[j]) >> 23);
        }
    }
    uint16_t* d = f.dst != nullptr ? f.dst + (long long) blockIdx.x * f.dst_stride : nullptr;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else if (f.pdst == nullptr) v[e] = r[j];
    }
}

// ---------------------------------------------------------------- hyper-connections
__global__ void __launch_bounds__(kThreads) hc_mix_kernel(const float* __restrict__ h, const float* __restrict__ fn,
                                                          int hcdim, float eps, float* __restrict__ mixes) {
    const int m = blockIdx.x, t = blockIdx.y;
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        const float4 wv = *(const float4*) (w + k);
        acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    if (threadIdx.x == 0) mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + eps);
}

// a warp a token: lane j * hc + k holds comb[j][k] (hc <= 4: lanes 0..15; wider streams fall back to lane 0 alone)
__global__ void hc_split_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                const float* __restrict__ base, int hc, int iters, float eps, float* pre, float* post,
                                float* comb) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* mx = mixes + (size_t) t * 32;
    pre += (size_t) t * 8;
    post += (size_t) t * 8;
    comb += (size_t) t * 64;
    if (lane < hc) {
        pre[lane] = 1.0f / (1.0f + expf(-(mx[lane] * scale[0] + base[lane]))) + eps;
        post[lane] = 2.0f / (1.0f + expf(-(mx[hc + lane] * scale[1] + base[hc + lane])));
    }
    if (hc == 4) {
        // rows are the 4-lane groups (j = lane / 4), columns the lanes 4 apart (k = lane % 4)
        const bool on = lane < 16;
        float c = on ? mx[8 + lane] * scale[2] + base[8 + lane] : -FLT_MAX;
        auto row_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            return v + __shfl_xor_sync(0xffffffffu, v, 2);
        };
        auto col_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            return v + __shfl_xor_sync(0xffffffffu, v, 8);
        };
        float m = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        c = on ? expf(c - m) : 0.0f;
        c = c / row_sum(c) + eps;                       // softmax over each row, + eps
        c = on ? c : 0.0f;
        c = c / (col_sum(c) + eps);                     // each destination column over its sources
        for (int it = 1; it < iters; ++it) {
            c = c / (row_sum(c) + eps);
            c = c / (col_sum(c) + eps);
        }
        if (on) comb[lane] = c;
        return;
    }
    if (lane != 0) return;
    float c[8][8];
    for (int j = 0; j < hc; ++j) {
        float m = -FLT_MAX;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
            m = fmaxf(m, c[j][k]);
        }
        float s = 0.0f;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = expf(c[j][k] - m);
            s += c[j][k];
        }
        for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + eps;
    }
    for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
        for (int j = 0; j < hc; ++j) s += c[j][k];
        for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.0f;
            for (int k = 0; k < hc; ++k) s += c[j][k];
            for (int k = 0; k < hc; ++k) c[j][k] /= s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.0f;
            for (int j = 0; j < hc; ++j) s += c[j][k];
            for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
        }
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
}

__global__ void __launch_bounds__(1024) hc_pre_norm_kernel(const float* __restrict__ h, const float* __restrict__ mix,
                                                           const float* __restrict__ w, int dim, int hc, float eps,
                                                           float* __restrict__ x) {
    const int t = blockIdx.x;
    h += (size_t) t * hc * dim;
    mix += (size_t) t * 8;
    x += (size_t) t * dim;
    float r[8];
    float ss = 0.0f;
    int n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) {
        float a = 0.0f;
        for (int s = 0; s < hc; ++s) a += mix[s] * h[(size_t) s * dim + e];
        r[n] = a;
        ss += a * a;
    }
    const float inv = rsqrtf(block_sum(ss) / (float) dim + eps);
    n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) x[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
}

__global__ void hc_post_kernel(const float* __restrict__ y, const float* __restrict__ res,
                               const float* __restrict__ post, const float* __restrict__ comb, int dim, int hc,
                               float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= dim * hc) return;
    const int k = i / dim, e = i - k * dim;
    const float* r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8 + k] * y[(size_t) t * dim + e];
    for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * r[(size_t) j * dim + e];
    out[(size_t) t * hc * dim + i] = a;
}

// ---------------------------------------------------------------- gemv over F16 weights, a warp a row
__global__ void __launch_bounds__(kThreads) gemv_f16_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                            int n, int k, float alpha, float* __restrict__ y) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const uint4*) (w + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            acc += f.x * xv[c + 2 * i] + f.y * xv[c + 2 * i + 1];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n + row] = alpha * acc;
}
// ... for T activations (rows ldx apart, 16-byte aligned; outputs ldy apart), kRowsT of them a block (grid y): each
// weight row read once for the block's activations, theirs in float4s - each activation's sum gemv_f16_kernel's (the
// same products in the same order), so its bits too.  (A block for all of them, 16 at most, was slower than a launch
// a token: its accumulators and guards kept the loop from unrolling - 66 against 29 us at 5 tokens on an RTX 5060 Ti,
// this 9)
constexpr int kRowsT = 2;
__global__ void __launch_bounds__(kThreads) gemv_f16_rows_kernel(const uint16_t* __restrict__ W,
                                                                 const float* __restrict__ x, size_t ldx, int n, int k,
                                                                 int T, float alpha, float* __restrict__ y, size_t ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int t0 = blockIdx.y * kRowsT, nt = min(kRowsT, T - t0);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    float acc[kRowsT];
#pragma unroll
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const uint4*) (w + c);
        const __half2* hp = (const __half2*) &q;
        float xv[kRowsT][8];
#pragma unroll
        for (int t = 0; t < kRowsT; ++t)
            if (t < nt) {
                const float* xp = x + (size_t) (t0 + t) * ldx + c;
                const float4 a = *(const float4*) xp, b = *(const float4*) (xp + 4);
                xv[t][0] = a.x, xv[t][1] = a.y, xv[t][2] = a.z, xv[t][3] = a.w;
                xv[t][4] = b.x, xv[t][5] = b.y, xv[t][6] = b.z, xv[t][7] = b.w;
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
#pragma unroll
            for (int t = 0; t < kRowsT; ++t)
                if (t < nt) acc[t] += f.x * xv[t][2 * i] + f.y * xv[t][2 * i + 1];
        }
    }
#pragma unroll
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

// ---------------------------------------------------------------- the router
__global__ void __launch_bounds__(512) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    int n, int k, float route_scale, int* idx, float* w,
                                                    const float* __restrict__ bias_vl, const VlBits vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl.w[t >> 5] >> (t & 31)) & 1u)) bias = bias_vl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        for (int r = 0; r < k; ++r) {
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float lim,
                              float* __restrict__ h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (lim > 0.0f) {
        gv = fminf(gv, lim);
        uv = fminf(fmaxf(uv, -lim), lim);
    }
    h[i] = gv / (1.0f + expf(-gv)) * uv;
}

// ---------------------------------------------------------------- the compressor (ratio > 1), a window in order
__global__ void __launch_bounds__(512) compress_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                                                       float* kv_state, float* score_state, int ratio, int p0, int T,
                                                       int d, const float* __restrict__ w, float eps,
                                                       float* __restrict__ latent) {
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        const bool pool = (p0 + t + 1) % ratio == 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * d + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * d + e];
        }
        __syncthreads();
        if (!pool) continue;
        // (kv * softmax(score, over the slots)).sum: each slot's probability first, then the sum, as torch does
        float val[2] = {0.0f, 0.0f};   // d <= 2 * blockDim
        float ss = 0.0f;
        int n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) {
            float m = -FLT_MAX;
            for (int s = 0; s < ratio; ++s) m = fmaxf(m, score_state[(size_t) s * d + e]);
            float den = 0.0f;
            for (int s = 0; s < ratio; ++s) den += expf(score_state[(size_t) s * d + e] - m);
            float acc = 0.0f;
            for (int s = 0; s < ratio; ++s)
                acc += kv_state[(size_t) s * d + e] * (expf(score_state[(size_t) s * d + e] - m) / den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = rsqrtf(block_sum(ss) / (float) d + eps);
        n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) latent[(size_t) g * d + e] = val[n] * inv * w[e];
        ++g;
        __syncthreads();
    }
}

// ---------------------------------------------------------------- per-query sizes
__device__ __forceinline__ void rows_at(const Rows& r, int t, int& n, int& k) {
    if (r.gen == 0) {
        n = r.n[t];
        k = r.k[t];
        return;
    }
    const int nn = (int) ((r.p0 + t + 1) / r.R);
    const int cb = min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
    case 1: n = nn, k = min(r.kcap, nn); break;
    case 2: n = nn, k = cb; break;
    case 3: n = cb * r.blk, k = min(r.kcap, nn); break;
    default: n = (nn + r.blk - 1) / r.blk, k = r.kcap; break;
    }
}

// ---------------------------------------------------------------- the indexer
// the same scores on the tensor cores (packed FP4 keys, hd 128, at most 32 heads), in integers: a key's 32 values are
// E2M1 codes times a power of two, and so are a query head's (quant 3) - twice each code an integer of at most 12, so a
// 32-value block's dot is an exact integer (mma.sync m16n8k32, int8 in, int32 out) times the two powers of two; the
// four blocks' sums in FP32 (index_score_kernel's FP32 sum of 128 products rounds more than this).  A tile of 128 key
// slots at a time, QT queries' 32 heads each (the shared rows: 1-6 queries a block; with candidates 1, each query its
// own rows), relu and the head weights' sum by lanes; a block loops over tiles (the queries converted once)
template <int QT>
__global__ void __launch_bounds__(256) index_score_tc_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                             const uint8_t* __restrict__ kp, int heads, int n, int T,
                                                             const int* __restrict__ cand, int cand_stride, int blk,
                                                             Rows lim, float* __restrict__ out, long long out_stride,
                                                             int ntiles) {
    constexpr int HD = 128, KT = 128, LD = HD + 16, NQ = QT * 32, NB = HD / 32, RB = HD / 2 + HD / 32;
    extern __shared__ __align__(16) int8_t tc_smem[];   // Ks [KT][LD], Qs [NQ][LD] (int8), ek [KT][NB], eq [NQ][NB]
    int8_t* Ks = tc_smem;
    int8_t* Qs = Ks + KT * LD;
    int* ek = (int*) (Qs + NQ * LD);
    int* eq = ek + KT * NB;
    __shared__ float ws[NQ];
    __shared__ int ln_s[QT], lk_s[QT];
    const int t0 = blockIdx.y * QT;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    // the queries to integers: a warp 4 heads of a query at a time, a lane 4 values of each (its loads all at once),
    // a block's 32 values on 8 lanes (their largest by shuffles: a = m 2^e, m in [0.5, 1) - the block's values times
    // 2^(4 - e) are integers of at most 12)
    for (int r0 = warp * 4; r0 < NQ; r0 += (blockDim.x >> 5) * 4) {
        float4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int r = r0 + u, t = t0 + r / 32, h = r % 32;
            v[u] = t < T && h < heads ? *(const float4*) (q + ((size_t) t * heads + h) * HD + lane * 4)
                                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            float a = fmaxf(fmaxf(fabsf(v[u].x), fabsf(v[u].y)), fmaxf(fabsf(v[u].z), fabsf(v[u].w)));
            for (int o = 4; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            // (a = m 2^e, m in [0.5, 1): e from its exponent bits; 0 for none - and its powers of two exactly)
            const int e = a > 0.0f ? (int) ((__float_as_uint(a) >> 23) & 0xffu) - 126 : 0;
            const float sc = __int_as_float((127 + 4 - e) << 23);
            const uint32_t b0 = (uint8_t) (int8_t) __float2int_rn(v[u].x * sc), b1 = (uint8_t) (int8_t) __float2int_rn(v[u].y * sc),
                           b2 = (uint8_t) (int8_t) __float2int_rn(v[u].z * sc), b3 = (uint8_t) (int8_t) __float2int_rn(v[u].w * sc);
            *(uint32_t*) (Qs + (r0 + u) * LD + lane * 4) = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            if ((lane & 7) == 0) eq[(r0 + u) * NB + lane / 8] = e - 4;
        }
    }
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) {
        const int t = t0 + i / 32, h = i % 32;
        ws[i] = t < T && h < heads ? w[(size_t) t * 64 + h] : 0.0f;
    }
    if ((int) threadIdx.x < QT) {
        int a = 0, b = 0;
        if (t0 + (int) threadIdx.x < T) rows_at(lim, t0 + (int) threadIdx.x, a, b);
        ln_s[threadIdx.x] = a, lk_s[threadIdx.x] = b;
    }
    __syncthreads();
    const int* cd = cand != nullptr ? cand + (size_t) t0 * cand_stride : nullptr;   // (QT 1 with candidates)
    // slot i's key position for query tq, or -1 (none: -inf)
    auto pos_of = [&](int i, int tq) -> int {
        if (i >= n) return -1;
        int p = i;
        if (cd != nullptr) {
            if (i >= lk_s[tq] * blk) return -1;
            p = cd[i / blk] * blk + i % blk;
        }
        return p < ln_s[tq] ? p : -1;
    };
    // a tile's rows: a thread 8 words of 8 codes (and their blocks' scale bytes) - fetched a tile ahead, into registers
    // while the tile before computes, then made integers in Ks (twice the E2M1 values: 0 1 2 3 4 6 8 12, signed)
    constexpr int NW = KT * (HD / 8) / 256;
    uint32_t rw[NW], sb[NW];
    auto fetch = [&](int tile) {
#pragma unroll
        for (int u = 0; u < NW; ++u) {
            const int x = threadIdx.x + u * 256, j = x / (HD / 8), c = (x % (HD / 8)) * 8, i = tile * KT + j;
            const int p = cd != nullptr ? pos_of(i, 0) : i < n ? i : -1;   // (shared rows: every slot below n)
            if (p >= 0) {
                const uint8_t* kr = kp + (size_t) p * RB;
                rw[u] = *(const uint32_t*) (kr + c / 2);
                sb[u] = (c & 31) == 0 ? kr[HD / 2 + c / 32] : 0u;
            } else
                rw[u] = 0u, sb[u] = 128u;   // (no row: zeros, its blocks' scale 2^0)
        }
    };
    auto stage = [&]() {
#pragma unroll
        for (int u = 0; u < NW; ++u) {
            const int x = threadIdx.x + u * 256, j = x / (HD / 8), c = (x % (HD / 8)) * 8;
            uint32_t lo = 0u, hi = 0u;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t cc = (rw[u] >> (4 * e)) & 15u;
                const uint32_t v = (uint32_t) (0x0C08060403020100ull >> (8 * (cc & 7u))) & 0xffu;
                const uint32_t byte = (cc & 8u ? 0u - v : v) & 0xffu;
                if (e < 4) lo |= byte << (8 * e);
                else hi |= byte << (8 * (e - 4));
            }
            *(uint2*) (Ks + j * LD + c) = make_uint2(lo, hi);
            if ((c & 31) == 0) ek[j * NB + c / 32] = (int) sb[u] - 128;
        }
    };
    if (blockIdx.x < ntiles) fetch(blockIdx.x);
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int i0 = tile * KT;
        stage();
        __syncthreads();
        if (tile + (int) gridDim.x < ntiles) fetch(tile + gridDim.x);   // (in flight beside the MMAs below)
        float acc[QT * 4][4];
#pragma unroll
        for (int nt = 0; nt < QT * 4; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
        const int ra = warp * 16 + g;   // (this thread's rows: ra and ra + 8)
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int8_t* kr0 = Ks + ra * LD + b * 32 + tg * 4;
            const uint32_t a0 = *(const uint32_t*) kr0, a1 = *(const uint32_t*) (kr0 + 8 * LD),
                           a2 = *(const uint32_t*) (kr0 + 16), a3 = *(const uint32_t*) (kr0 + 8 * LD + 16);
            const int ea = ek[ra * NB + b], eb = ek[(ra + 8) * NB + b];
#pragma unroll
            for (int nt = 0; nt < QT * 4; ++nt) {
                const int col = nt * 8 + tg * 2;
                const int8_t* qr = Qs + (nt * 8 + g) * LD + b * 32 + tg * 4;
                const uint32_t b0 = *(const uint32_t*) qr, b1 = *(const uint32_t*) (qr + 16);
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                             "{%0,%1,%2,%3};\n"
                             : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                const int q0 = eq[col * NB + b], q1 = eq[(col + 1) * NB + b];
                acc[nt][0] += (float) c0 * __int_as_float((127 + ea + q0) << 23);
                acc[nt][1] += (float) c1 * __int_as_float((127 + ea + q1) << 23);
                acc[nt][2] += (float) c2 * __int_as_float((127 + eb + q0) << 23);
                acc[nt][3] += (float) c3 * __int_as_float((127 + eb + q1) << 23);
            }
        }
        // rows ra (c0, c1) and ra + 8 (c2, c3); columns nt*8 + tg*2 + {0, 1}: query nt / 4, head (nt % 4)*8 + ..
#pragma unroll
        for (int tq = 0; tq < QT; ++tq) {
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float* wh = ws + tq * 32 + u * 8 + tg * 2;
                const float* av = acc[tq * 4 + u];
                s0 += fmaxf(av[0], 0.0f) * wh[0] + fmaxf(av[1], 0.0f) * wh[1];
                s1 += fmaxf(av[2], 0.0f) * wh[0] + fmaxf(av[3], 0.0f) * wh[1];
            }
            s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
            const int t = t0 + tq;
            if (tg == 0 && t < T) {
                float* o = out + (long long) t * out_stride;
                const int ia = i0 + ra, ib = ia + 8;
                if (ia < n) o[ia] = pos_of(ia, tq) >= 0 ? s0 : -INFINITY;
                if (ib < n) o[ib] = pos_of(ib, tq) >= 0 ? s1 : -INFINITY;
            }
        }
        __syncthreads();   // (the next tile's rows over these)
    }
}
template <int QT>
constexpr size_t index_tc_smem() { return (size_t) (128 + QT * 32) * (128 + 16) + (size_t) (128 + QT * 32) * 4 * 4; }
template <int QT>
void index_tc_launch(int gx, int gy, cudaStream_t s, const float* q, const float* w, const uint8_t* kp, int heads, int n,
                     int T, const int* cand, int cand_stride, int blk, const Rows& lim, float* out, long long out_stride,
                     int ntiles) {
    index_score_tc_kernel<QT><<<dim3(gx, gy), 256, index_tc_smem<QT>(), s>>>(q, w, kp, heads, n, T, cand, cand_stride, blk,
                                                                             lim, out, out_stride, ntiles);
}

// a thread a position: its key row (BF16) against every head's query (shared memory), relu, weighted sum
template <int HD>
__global__ void __launch_bounds__(kThreads) index_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                               const uint16_t* __restrict__ kc,
                                                               const uint8_t* __restrict__ kp, int heads, int n,
                                                               const int* __restrict__ cand, int cand_stride, int blk,
                                                               Rows lim, float* __restrict__ out,
                                                               long long out_stride) {
    extern __shared__ float qs[];   // heads * HD, then heads weights
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * HD; i += blockDim.x) qs[i] = q[(size_t) t * heads * HD + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * HD + i] = w[(size_t) t * 64 + i];
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + (long long) t * out_stride;
    int ln, lk;
    rows_at(lim, t, ln, lk);
    int p = i;
    if (cand != nullptr) {
        if (i >= lk * blk) {
            o[i] = -INFINITY;
            return;
        }
        p = cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
    }
    if (p >= ln) {
        o[i] = -INFINITY;
        return;
    }
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.0f;
    if (kp != nullptr) {   // packed FP4 (quant 3): 8 codes a 4-byte word, a power of two a 32 - BF16's values exactly
        const uint8_t* kr = kp + (size_t) p * (HD / 2 + HD / 32);
#pragma unroll 2
        for (int c = 0; c < HD; c += 8) {
            const uint32_t w8 = *(const uint32_t*) (kr + c / 2);
            const float sc = __uint_as_float((uint32_t) kr[HD / 2 + c / 32] << 23);
            float kv[8];
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = __fmul_rn(fp4v((w8 >> (4 * x)) & 15u), sc);
#pragma unroll
            for (int h = 0; h < 32; ++h) {
                if (h >= heads) break;
                const float* qh = qs + h * HD + c;
#pragma unroll
                for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
            }
        }
    } else {
        const uint16_t* kr = kc + (size_t) p * HD;
#pragma unroll 2
        for (int c = 0; c < HD; c += 8) {
            const uint4 raw = *(const uint4*) (kr + c);
            const uint16_t* b = (const uint16_t*) &raw;
            float kv[8];
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = bf16f(b[x]);
#pragma unroll
            for (int h = 0; h < 32; ++h) {
                if (h >= heads) break;
                const float* qh = qs + h * HD + c;
#pragma unroll
                for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
            }
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 32; ++h) {
        if (h >= heads) break;
        s += fmaxf(acc[h], 0.0f) * qs[heads * HD + h];
    }
    o[i] = s;
}

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// a block's selection over a sequence of n keys (key(i)): the k-th largest by four 8-bit radix passes, then everything
// above it and the first of its ties - placed by a block-wide scan in sequence order (not by atomics: the same rows, in
// the same order, run after run - the sparse attention sums them in that order).  emit(slot, i) takes pick i to its
// slot: LAYOUT 0 (the indexer's) the ones above the k-th in order, then its ties in order; LAYOUT 1 the picks as one run
// in sequence order (a row segment's candidates for topk_merge).  k < n; returns the k-th key
template <int LAYOUT, class Key, class Emit>
__device__ __forceinline__ uint32_t topk_block(int n, int k, Key key, Emit emit) {
    __shared__ unsigned hist[256];
    __shared__ uint32_t prefix_s, mask_s;
    __shared__ int remaining_s;
    __shared__ int warp_gt[32], warp_eq[32], warp_sel[32];
    __shared__ int base_gt, base_eq, base_sel;
    if (threadIdx.x == 0) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        base_gt = 0;
        base_eq = 0;
        base_sel = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t u = key(i);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = (int) hist[d];
                if (cum + h >= rem) {
                    prefix_s = prefix | ((uint32_t) d << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        __syncthreads();
    }
    const uint32_t thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = (int) blockDim.x >> 5;
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const uint32_t u = i < n ? key(i) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const unsigned mg = __ballot_sync(0xffffffffu, gt), me = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            warp_gt[wp] = __popc(mg);
            warp_eq[wp] = __popc(me);
        }
        __syncthreads();
        int bg = base_gt, be = base_eq, tg = 0, te = 0;
        for (int w = 0; w < nw; ++w) {
            if (w < wp) {
                bg += warp_gt[w];
                be += warp_eq[w];
            }
            tg += warp_gt[w];
            te += warp_eq[w];
        }
        const unsigned below = (1u << lane) - 1u;
        const int e = be + __popc(me & below);   // (this one's place among the ties)
        if constexpr (LAYOUT == 0) {
            int slot = -1;
            if (gt) slot = bg + __popc(mg & below);
            else if (eq && e < remaining) slot = n_gt + e;
            if (slot >= 0) emit(slot, i);
            __syncthreads();
        } else {
            const bool sel = gt || (eq && e < remaining);
            const unsigned ms = __ballot_sync(0xffffffffu, sel);
            if (lane == 0) warp_sel[wp] = __popc(ms);
            __syncthreads();
            int bs = base_sel, ts = 0;
            for (int w = 0; w < nw; ++w) {
                if (w < wp) bs += warp_sel[w];
                ts += warp_sel[w];
            }
            if (sel) emit(bs + __popc(ms & below), i);
            __syncthreads();
            if (threadIdx.x == 0) base_sel += ts;
        }
        if (threadIdx.x == 0) {
            base_gt += tg;
            base_eq += te;
        }
        __syncthreads();
    }
    return thr;
}

// a block a row: topk_block over the row
__global__ void __launch_bounds__(1024) topk_kernel(const float* __restrict__ vals, long long stride, Rows r,
                                                    const int* __restrict__ cand, int cand_stride, int blk,
                                                    int* __restrict__ out, int out_stride) {
    const int t = blockIdx.x;
    int n, k;
    rows_at(r, t, n, k);
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    const int* cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        return;
    }
    topk_block<0>(n, k, [&](int i) { return ord_key(vals[i]); },
                  [&](int slot, int i) { out[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i; });
}

// a few rows of many positions (a decode window's): each row in P segments, a block each - its own top k, in position
// order (every one of the row's picks in it is among them: fewer than k of the row's lie above the row's k-th value,
// and its ties there are the segment's first) - then a block a row over the segments' runs, in position order too: the
// picks and the order topk_kernel gives, a row's passes spread over P blocks instead of one
__global__ void __launch_bounds__(1024) topk_seg_kernel(const float* __restrict__ vals, long long stride, Rows r, int P,
                                                        int K, float* __restrict__ sv, int* __restrict__ si,
                                                        int* __restrict__ sn, uint32_t* __restrict__ sk) {
    const int p = blockIdx.x, t = blockIdx.y;
    int n, k;
    rows_at(r, t, n, k);
    if (n <= 0 || k >= n) return;   // (the merge takes the whole row)
    vals += (long long) t * stride;
    const int S = (n + P - 1) / P, lo = p * S, ns = max(0, min(n, lo + S) - lo);
    float* cv = sv + ((size_t) t * P + p) * K;
    int* ci = si + ((size_t) t * P + p) * K;
    if (ns <= k) {   // (all of it: no floor)
        for (int i = threadIdx.x; i < ns; i += blockDim.x) cv[i] = vals[lo + i], ci[i] = lo + i;
        if (threadIdx.x == 0) sn[t * P + p] = ns, sk[t * P + p] = 0u;
        return;
    }
    const uint32_t kth = topk_block<1>(ns, k, [&](int i) { return ord_key(vals[lo + i]); },
                                       [&](int slot, int i) { cv[slot] = vals[lo + i], ci[slot] = lo + i; });
    if (threadIdx.x == 0) sn[t * P + p] = k, sk[t * P + p] = kth;   // (the row's k-th is at least the segment's)
}
// the merge: only the segments' picks at or above the largest segment's k-th (the row's k-th is at least that) - into
// shared memory in position order, then topk_block over them (all of the candidates when more than kMergeCap pass: with
// 16K-position segments ~3% of a row pass, 8192 of them past 256K positions)
constexpr int kMergeCap = 8192;
__global__ void __launch_bounds__(1024) topk_merge_kernel(Rows r, int P, int K, const float* __restrict__ sv,
                                                          const int* __restrict__ si, const int* __restrict__ sn,
                                                          const uint32_t* __restrict__ sk, const int* __restrict__ cand,
                                                          int cand_stride, int blk, int* __restrict__ out,
                                                          int out_stride) {
    __shared__ int off[kTopkSegMax + 1];
    __shared__ uint32_t floor_s;
    extern __shared__ __align__(16) uint8_t merge_smem[];   // [kMergeCap] values, then [kMergeCap] positions
    float* mv = (float*) merge_smem;
    int* mi = (int*) (merge_smem + (size_t) kMergeCap * 4);
    __shared__ int warp_n[32], base_n;
    const int t = blockIdx.x;
    int n, k;
    rows_at(r, t, n, k);
    out += (size_t) t * out_stride;
    const int* cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        return;
    }
    if (threadIdx.x == 0) {
        off[0] = 0;
        uint32_t f = 0u;
        for (int p = 0; p < P; ++p) {
            off[p + 1] = off[p] + sn[t * P + p];
            f = max(f, sk[t * P + p]);
        }
        floor_s = f;
        base_n = 0;
    }
    __syncthreads();
    const float* cv = sv + (size_t) t * P * K;
    const int* ci = si + (size_t) t * P * K;
    const uint32_t fl = floor_s;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = (int) blockDim.x >> 5;
    bool fits = true;
    for (int p = 0; p < P && fits; ++p) {   // (each segment's run in order, the passing ones packed in order)
        const int c = off[p + 1] - off[p];
        for (int j0 = 0; j0 < c; j0 += blockDim.x) {
            const int j = j0 + threadIdx.x;
            const float x = j < c ? cv[(size_t) p * K + j] : 0.0f;
            const bool keep = j < c && ord_key(x) >= fl;
            const unsigned m = __ballot_sync(0xffffffffu, keep);
            if (lane == 0) warp_n[wp] = __popc(m);
            __syncthreads();
            int b = base_n, tot = 0;
            for (int w = 0; w < nw; ++w) {
                if (w < wp) b += warp_n[w];
                tot += warp_n[w];
            }
            const int at = b + __popc(m & ((1u << lane) - 1u));
            if (keep && at < kMergeCap) mv[at] = x, mi[at] = ci[(size_t) p * K + j];
            __syncthreads();
            if (threadIdx.x == 0) base_n += tot;
            __syncthreads();
            if (base_n > kMergeCap) {
                fits = false;
                break;
            }
        }
    }
    if (fits) {
        topk_block<0>(base_n, k, [&](int i) { return ord_key(mv[i]); }, [&](int slot, int i) {
            const int x = mi[i];
            out[slot] = cd != nullptr ? cd[x / blk] * blk + x % blk : x;
        });
        return;
    }
    auto at = [&](int i) {   // (all of them: candidate i in segment p's run, from off[p])
        int lo = 0, hi = P;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (off[mid] <= i) lo = mid;
            else hi = mid;
        }
        return (size_t) lo * K + (i - off[lo]);
    };
    topk_block<0>(off[P], k, [&](int i) { return ord_key(cv[at(i)]); }, [&](int slot, int i) {
        const int x = ci[at(i)];
        out[slot] = cd != nullptr ? cd[x / blk] * blk + x % blk : x;
    });
}

__global__ void block_max_kernel(const float* __restrict__ score, long long stride, Rows r, int blk,
                                 float* __restrict__ bm, long long bm_stride) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    int n, k_;
    rows_at(r, t, n, k_);
    const int nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const int p = b * blk + j;
        if (p < n) m = fmaxf(m, score[(long long) t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long long) t * bm_stride + b] = m;
}

// ---------------------------------------------------------------- sparse attention, a block a (head, query)
constexpr int kAttTile = 64;

__global__ void __launch_bounds__(kThreads) sparse_attn_kernel(const float* __restrict__ q,
                                                               const uint16_t* __restrict__ win,
                                                               const uint16_t* __restrict__ ckv,
                                                               const uint8_t* __restrict__ ckvp,
                                                               const int* __restrict__ sel, int sel_stride,
                                                               AttnRows ar, const float* __restrict__ sink, int heads,
                                                               int d, float scale, float* __restrict__ out) {
    // d == 512: a thread owns 2 output values, a lane 16 of a row's dot
    const int h = blockIdx.x, t = blockIdx.y;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        // the window buffer row of max(0, lo, pa - win + 1) (lo: the run's floor, a truncated prompt's tail)
        const long long first = max((long long) ar.win - ar.p0 + ar.lo, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    __shared__ float qs[512];
    __shared__ float sc[kAttTile];
    __shared__ const uint16_t* rows[kAttTile];
    __shared__ float m_run_s, tile_scale_s, den_s;
    for (int i = threadIdx.x; i < d; i += blockDim.x) qs[i] = q[((size_t) t * heads + h) * d + i];
    if (threadIdx.x == 0) {
        m_run_s = -1e30f;
        den_s = 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float o0 = 0.0f, o1 = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? wbase + (size_t) j * d
                              : ckvp != nullptr ? (const uint16_t*) (ckvp + (size_t) sl[j - n_win] * (512 / 2 + 512 / 16))
                                                : ckv + (size_t) sl[j - n_win] * d;
        }
        __syncthreads();
        for (int jj = wp; jj < nt; jj += kThreads / 32) {
            // (the lane's 16 values, either way the same floats - then one dot for both: the same bits)
            float kv[16];
            if (ckvp != nullptr && t0 + jj >= n_win) {   // a packed row (quant 2): a lane's 16 codes, its group's scale
                const uint8_t* rr = (const uint8_t*) rows[jj];
                const uint2 w = *(const uint2*) (rr + lane * 8);
                const float ks = e4m3v(rr[256 + lane]);
#pragma unroll
                for (int x = 0; x < 8; ++x) {
                    kv[x] = __fmul_rn(fp4v((w.x >> (4 * x)) & 15u), ks);
                    kv[8 + x] = __fmul_rn(fp4v((w.y >> (4 * x)) & 15u), ks);
                }
            } else {
                const uint16_t* rr = rows[jj];
                const uint4 a = *(const uint4*) (rr + lane * 16);
                const uint4 b = *(const uint4*) (rr + lane * 16 + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
#pragma unroll
                for (int x = 0; x < 8; ++x) {
                    kv[x] = bf16f(pa[x]);
                    kv[8 + x] = bf16f(pb[x]);
                }
            }
            float s = 0.0f;
#pragma unroll
            for (int x = 0; x < 16; ++x) s += qs[lane * 16 + x] * kv[x];
            s = warp_sum(s);
            if (lane == 0) sc[jj] = s * scale;
        }
        __syncthreads();
        if (wp == 0) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[j]);
            m = warp_max(m);
            const float m_old = m_run_s;
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = expf(sc[j] - m_new);
                sc[j] = p;
                s += p;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s = expf(m_old - m_new);
                den_s = den_s * tile_scale_s + s;
                m_run_s = m_new;
            }
        }
        __syncthreads();
        const float cs = tile_scale_s;
        o0 *= cs;
        o1 *= cs;
        const int c = threadIdx.x * 2;
        for (int j = 0; j < nt; ++j) {
            float v0, v1;
            if (ckvp != nullptr && t0 + j >= n_win) {   // (packed: this thread's two codes a byte, their group's scale)
                const uint8_t* rr = (const uint8_t*) rows[j];
                const uint32_t b = rr[c >> 1];
                const float ks = e4m3v(rr[256 + (c >> 4)]);
                v0 = __fmul_rn(fp4v(b & 15u), ks);
                v1 = __fmul_rn(fp4v(b >> 4), ks);
            } else {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                v0 = bf16f((uint16_t) (pr & 0xffffu));
                v1 = bf16f((uint16_t) (pr >> 16));
            }
            o0 += sc[j] * v0;
            o1 += sc[j] * v1;
        }
        __syncthreads();
    }
    const float den = den_s + expf(sink[h] - m_run_s);
    const int c = threadIdx.x * 2;
    float* ov = out + ((size_t) t * heads + h) * d;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

// ---------------------------------------------------------------- sparse attention, a block HG heads of a query
// sparse_attn_kernel's arithmetic a head a warp: each tile's rows read (a packed one decoded) once for the block's HG
// heads instead of once a head, into shared memory as BF16 - a packed row's products exact there (an E2M1 value times an
// E4M3 scale has at most 6 significant bits, BF16 has 8 and FP32's range), a BF16 row as it is - and each head's dots,
// softmax and sums in sparse_attn_kernel's order and tiles: the same bits.  A lane's dots for 32 rows at once, summed
// over the warp by recursive halving (lane l ends with row l's sum): warp_sum's tree for each row, in 31 shuffles
template <int HG>
__global__ void __launch_bounds__(HG * 32) sparse_attn_hg_kernel(const float* __restrict__ q,
                                                                 const uint16_t* __restrict__ win,
                                                                 const uint16_t* __restrict__ ckv,
                                                                 const uint8_t* __restrict__ ckvp,
                                                                 const int* __restrict__ sel, int sel_stride,
                                                                 AttnRows ar, const float* __restrict__ sink, int heads,
                                                                 float scale, float* __restrict__ out) {
    constexpr int d = 512, per = kAttTile * d / (HG * 32);   // (values of a row each thread decodes: 64 for HG 16)
    extern __shared__ __align__(16) uint16_t kt[];          // [kAttTile][512] BF16
    __shared__ float sc[HG][kAttTile];
    const int t = blockIdx.y, lane = threadIdx.x & 31, wp = threadIdx.x >> 5, h = blockIdx.x * HG + wp;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0 + ar.lo, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    float qv[16], o[16];
    {
        const float* qr = q + ((size_t) t * heads + h) * d + lane * 16;
#pragma unroll
        for (int x = 0; x < 16; ++x) {
            qv[x] = qr[x];
            o[x] = 0.0f;
        }
    }
    // a row's 16 values of this lane (dims lane * 16 ..): BF16 pairs a word, the low one first
    auto row16 = [&](int j, float (&v)[16]) {
        const uint4 a = *(const uint4*) (kt + (size_t) j * d + lane * 16);
        const uint4 b = *(const uint4*) (kt + (size_t) j * d + lane * 16 + 8);
        const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[2 * i] = __uint_as_float(w[i] << 16);
            v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
        }
    };
    float m_run = -1e30f, den = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        __syncthreads();   // (the last tile read by every warp)
        {   // thread i: row i / (512 / per), its values [c0, c0 + per)
            const int j = threadIdx.x / (d / per), c0 = threadIdx.x % (d / per) * per, jj = t0 + j;
            if (j < nt) {
                uint16_t* dst = kt + (size_t) j * d + c0;
                const uint16_t* src = jj < n_win ? wbase + (size_t) jj * d
                                    : ckvp == nullptr ? ckv + (size_t) sl[jj - n_win] * d : nullptr;
                if (src != nullptr) {
                    for (int i = 0; i < per; i += 8) *(uint4*) (dst + i) = *(const uint4*) (src + c0 + i);
                } else {
                    const uint8_t* rr = ckvp + (size_t) sl[jj - n_win] * (512 / 2 + 512 / 16);
                    for (int i = 0; i < per; i += 16) {   // 16 values: 8 code bytes, one scale
                        const uint2 w = *(const uint2*) (rr + (c0 + i) / 2);
                        const float ks = e4m3v(rr[256 + (c0 + i) / 16]);
                        uint32_t hv[8];
#pragma unroll
                        for (int x = 0; x < 4; ++x) {
                            hv[x] = (uint32_t) f2bf16(__fmul_rn(fp4v((w.x >> (8 * x)) & 15u), ks)) |
                                    ((uint32_t) f2bf16(__fmul_rn(fp4v((w.x >> (8 * x + 4)) & 15u), ks)) << 16);
                            hv[4 + x] = (uint32_t) f2bf16(__fmul_rn(fp4v((w.y >> (8 * x)) & 15u), ks)) |
                                        ((uint32_t) f2bf16(__fmul_rn(fp4v((w.y >> (8 * x + 4)) & 15u), ks)) << 16);
                        }
                        *(uint4*) (dst + i) = make_uint4(hv[0], hv[1], hv[2], hv[3]);
                        *(uint4*) (dst + i + 8) = make_uint4(hv[4], hv[5], hv[6], hv[7]);
                    }
                }
            }
        }
        __syncthreads();
        for (int k0 = 0; k0 < nt; k0 += 32) {   // this head's dots, 32 rows at once
            float p[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                float s = 0.0f;
                if (k0 + k < nt) {
                    float kv[16];
                    row16(k0 + k, kv);
#pragma unroll
                    for (int x = 0; x < 16; ++x) s += qv[x] * kv[x];
                }
                p[k] = s;
            }
#pragma unroll
            for (int off = 16, n = 32; off > 0; off >>= 1, n >>= 1) {   // keep the half of the rows this lane's bit names
                const bool up = (lane & off) != 0;
#pragma unroll
                for (int k = 0; k < n / 2; ++k) {
                    const float send = up ? p[k] : p[k + n / 2], keep = up ? p[k + n / 2] : p[k];
                    p[k] = keep + __shfl_xor_sync(0xffffffffu, send, off);
                }
            }
            if (k0 + lane < nt) sc[wp][k0 + lane] = p[0] * scale;
        }
        __syncwarp();
        float m = -INFINITY;
        for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[wp][j]);
        m = warp_max(m);
        const float m_old = m_run;
        const float m_new = fmaxf(m_old, m);
        float s = 0.0f;
        for (int j = lane; j < nt; j += 32) {
            const float pj = expf(sc[wp][j] - m_new);
            sc[wp][j] = pj;
            s += pj;
        }
        s = warp_sum(s);
        __syncwarp();
        const float cs = expf(m_old - m_new);
        den = den * cs + s;
        m_run = m_new;
#pragma unroll
        for (int x = 0; x < 16; ++x) o[x] *= cs;
        for (int j = 0; j < nt; ++j) {
            const float pj = sc[wp][j];
            float v[16];
            row16(j, v);
#pragma unroll
            for (int x = 0; x < 16; ++x) o[x] += pj * v[x];
        }
    }
    const float dn = den + expf(sink[h] - m_run);
    float* ov = out + ((size_t) t * heads + h) * d + lane * 16;
#pragma unroll
    for (int x = 0; x < 16; ++x) ov[x] = o[x] / dn;
}

__global__ void ring_to_ext_kernel(const uint16_t* __restrict__ ring, uint16_t* __restrict__ ext, long long p0, int n,
                                   int d) {
    const int r = blockIdx.x;
    const long long p = p0 - n + r;
    for (int e = threadIdx.x; e < d; e += blockDim.x)
        ext[(size_t) r * d + e] = p >= 0 ? ring[(size_t) (p % n) * d + e] : (uint16_t) 0;
}

__global__ void ext_to_ring_kernel(const uint16_t* __restrict__ ext, uint16_t* __restrict__ ring, long long p0, int n,
                                   int T, int d, uint16_t* __restrict__ backup) {
    const int t = blockIdx.x;
    if (T - t > n) return;   // overwritten by a later row of the window anyway
    uint16_t* row = ring + (size_t) ((p0 + t) % n) * d;
    for (int e = threadIdx.x; e < d; e += blockDim.x) {
        if (backup != nullptr) backup[(size_t) t * d + e] = row[e];
        row[e] = ext[(size_t) (n + t) * d + e];
    }
}

__global__ void ring_restore_kernel(uint16_t* __restrict__ ring, const uint16_t* __restrict__ backup, long long p0,
                                    int n, int keep, int d) {
    const int t = keep + blockIdx.x;
    uint16_t* row = ring + (size_t) ((p0 + t) % n) * d;
    for (int e = threadIdx.x; e < d; e += blockDim.x) row[e] = backup[(size_t) t * d + e];
}
__global__ void ring_restore_batch_kernel(const RingBatch b, long long p0, int n, int keep, int d) {
    const int t = keep + blockIdx.x, li = blockIdx.y;
    uint16_t* row = b.ring[li] + (size_t) ((p0 + t) % n) * d;
    const uint16_t* src = b.bak[li] + (size_t) t * d;
    for (int e = threadIdx.x; e < d; e += blockDim.x) row[e] = src[e];
}

__global__ void comp_restore_batch_kernel(const CompBatch b, long long p0, int keep, int d) {
    const int li = blockIdx.x, R = b.ratio[li];
    for (int i = threadIdx.x; i < R * d; i += blockDim.x) {
        const int slot = i / d, e = i % d;
        float kv = b.snap[li][i], sc = b.snap[li][(size_t) R * d + i];   // (the last kept row at the slot, else the snapshot)
        for (int t = 0; t < keep; ++t)
            if ((int) ((p0 + t) % R) == slot) {
                kv = b.kv[li][(size_t) t * d + e];
                sc = b.score[li][(size_t) t * d + e];
            }
        b.kv_state[li][i] = kv;
        b.score_state[li][i] = sc;
    }
}

__global__ void hc_mean_kernel(const float* __restrict__ h, int hc, int dim, float* __restrict__ out,
                               long long out_stride) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int s = 0; s < hc; ++s) a += h[((size_t) t * hc + s) * dim + e];
    out[(long long) t * out_stride + e] = a / (float) hc;
}

__global__ void iota_rows_kernel(int* out, int n, int stride) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[(size_t) t * stride + i] = i;
}

// a warp a vocabulary row: E[*tok] (shared memory) . H[v]
__global__ void __launch_bounds__(kThreads) markov_kernel(float* __restrict__ logits, const uint16_t* __restrict__ E,
                                                          const uint16_t* __restrict__ H, const int* __restrict__ tok,
                                                          int V, int R) {
    extern __shared__ float er[];
    const uint16_t* erow = E + (size_t) (*tok) * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint4 q = *(const uint4*) (hr + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            a += f.x * er[c + 2 * i] + f.y * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

// ---------------------------------------------------------------- the markov bias over a head group (markov_share)
// a member's rows [v0, v0 + vn) of the bias (markov_kernel's sums, the prev token's E row already on the device, BF16)
__global__ void __launch_bounds__(kThreads) markov_share_kernel(float* __restrict__ logits,
                                                                const uint16_t* __restrict__ H,
                                                                const uint16_t* __restrict__ erow, int vn, int R) {
    extern __shared__ float er[];
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= vn) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint4 q = *(const uint4*) (hr + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            a += f.x * er[c + 2 * i] + f.y * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}
// the share's argmax (the greedy sampler's rule: the larger value, on a tie the lower index) -> slot {value, v0 + i,
// seq}, the seq last
__global__ void __launch_bounds__(1024) argmax_share_kernel(const float* __restrict__ row, int vn, int v0,
                                                           unsigned* slot, unsigned seq) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = __int_as_float(0xff800000);
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < vn; i += blockDim.x) {
        const float x = row[i];
        if (x > bv) bv = x, bi = i;   // (ascending: the lowest index of a tie stays)
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = bv, si[threadIdx.x >> 5] = bi;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) bv = sv[w], bi = si[w];
        slot[0] = __float_as_uint(bv);
        slot[1] = (unsigned) (v0 + (bi == 0x7fffffff ? 0 : bi));
        __threadfence_system();
        *(volatile unsigned*) (slot + 2) = seq;
    }
}
// every member's share argmax of this step (slots nm apart) -> the global one into *tok, and its E row (host-mapped
// BF16 [V][R]) into erow for the next step's bias
__global__ void markov_pick_kernel(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E,
                                   uint16_t* erow, int R) {
    __shared__ int pick;
    __shared__ float sv[256];
    __shared__ int si[256];
    if ((int) threadIdx.x < nm) {   // (each member's slot read by a thread of its own: host memory, a PCIe round trip each)
        while ((int) (*(volatile const unsigned*) (slots + (size_t) threadIdx.x * 4 + 2) - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(32);
#endif
        }
        sv[threadIdx.x] = __uint_as_float(*(volatile const unsigned*) (slots + (size_t) threadIdx.x * 4));
        si[threadIdx.x] = (int) *(volatile const unsigned*) (slots + (size_t) threadIdx.x * 4 + 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = __int_as_float(0xff800000);
        int bi = 0x7fffffff;
        for (int m = 0; m < nm; ++m) {   // (members in vocabulary order: a tie keeps the lower index)
            const float v = sv[m];
            const int i = si[m];
            if (v > bv || (v == bv && i < bi)) bv = v, bi = i;
        }
        pick = bi;
        *tok = bi;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < R; i += blockDim.x) erow[i] = __ldcv(E + (size_t) pick * R + i);
}

// ---------------------------------------------------------------- sampled drafts over a head group (spec_pick)
// a share's top k (k <= 64) of a logits row -> a host-mapped list {seq, n, then n x (ordered value, global index)}:
// the sampler's order (the larger value, on a tie the lower index)
__device__ __forceinline__ unsigned spec_ord(float x) {   // (sampler.cu's ordered(): NaN lowest, -0 as +0)
    unsigned b = __float_as_uint(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) return 0u;
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__global__ void __launch_bounds__(1024) topk_share_kernel(const float* __restrict__ row, int vn, int v0, int k,
                                                          unsigned* list, unsigned seq) {
    constexpr int kBins = 2048, kCap = 1024;   // (the ordered value's top 11 bits; the candidates the fast path ranks)
    __shared__ unsigned pool[kBins + 1024];    // the coarse histogram, then the candidates (values, then indices)
    __shared__ unsigned hist[256], cu[64], scan[1024];
    __shared__ int ci[64];
    __shared__ unsigned sh_prefix, sh_rem, sh_n, sh_nc, sh_base;
    __shared__ int sh_bin;
    if (k > vn) k = vn;
    // the fast path (sampler.cu's): the coarse bin holding the k-th largest, every value in it or above ranked in shared
    // memory when they fit (a share's top values seldom crowd one bin past kCap)
    for (int b = threadIdx.x; b < kBins; b += blockDim.x) pool[b] = 0u;
    if (threadIdx.x == 0) sh_nc = 0u, sh_base = 0u;
    __syncthreads();
    for (int v = threadIdx.x; v < vn; v += blockDim.x) atomicAdd(&pool[spec_ord(row[v]) >> 21], 1u);
    __syncthreads();
    {   // suffix counts over the bins, from the top (2 bins a thread)
        const int nt = (int) blockDim.x, per = (kBins + nt - 1) / nt, b_hi = kBins - 1 - (int) threadIdx.x * per;
        unsigned mine = 0u;
        for (int j = 0; j < per && b_hi - j >= 0; ++j) mine += pool[b_hi - j];
        scan[threadIdx.x] = mine;
        __syncthreads();
        for (int off = 1; off < nt; off <<= 1) {
            const unsigned add = (int) threadIdx.x >= off ? scan[threadIdx.x - off] : 0u;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }
        unsigned above = scan[threadIdx.x] - mine;
        if (above < (unsigned) k && above + mine >= (unsigned) k)
            for (int j = 0; j < per && b_hi - j >= 0; ++j) {
                above += pool[b_hi - j];
                if (above >= (unsigned) k) {
                    sh_bin = b_hi - j;
                    sh_base = above;   // (the values in that bin and above)
                    break;
                }
            }
        __syncthreads();
    }
    if (sh_base <= (unsigned) kCap) {
        const int bin = sh_bin;
        unsigned* cv = pool;
        int* cix = (int*) pool + kCap;
        __syncthreads();
        for (int v = threadIdx.x; v < vn; v += blockDim.x) {
            const unsigned u = spec_ord(row[v]);
            if ((int) (u >> 21) >= bin) {
                const unsigned at = atomicAdd(&sh_nc, 1u);
                if (at < (unsigned) kCap) cv[at] = u, cix[at] = v;
            }
        }
        __syncthreads();
        const int nc = (int) min(sh_nc, (unsigned) kCap);
        for (int i = threadIdx.x; i < nc; i += blockDim.x) {
            const unsigned ui = cv[i];
            const int ii = cix[i];
            int r = 0;
            for (int j = 0; j < nc && r < k; ++j) r += cv[j] > ui || (cv[j] == ui && cix[j] < ii);
            if (r < k) {
                list[2 + 2 * r] = ui;
                list[3 + 2 * r] = (unsigned) (v0 + ii);
            }
        }
    } else {
        // the exact radix select of the k-th largest (4 byte passes), everything above it and the lowest-indexed of its
        // ties, then a rank sort
        if (threadIdx.x == 0) {
            sh_prefix = 0u;
            sh_rem = (unsigned) k;
            sh_n = 0u;
        }
        unsigned mask = 0u;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0u;
            __syncthreads();
            const unsigned prefix = sh_prefix;
            for (int v = threadIdx.x; v < vn; v += blockDim.x) {
                const unsigned u = spec_ord(row[v]);
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned cum = 0u, rem = sh_rem;
                int d = 255;
                for (; d > 0; --d) {
                    if (cum + hist[d] >= rem) break;
                    cum += hist[d];
                }
                sh_rem = rem - cum;
                sh_prefix = prefix | ((unsigned) d << shift);
            }
            mask |= 255u << shift;
            __syncthreads();
        }
        const unsigned T = sh_prefix, need = sh_rem;
        for (int v = threadIdx.x; v < vn; v += blockDim.x) {
            const unsigned u = spec_ord(row[v]);
            if (u > T) {
                const unsigned at = atomicAdd(&sh_n, 1u);
                if (at < 64u) cu[at] = u, ci[at] = v;
            }
        }
        __syncthreads();
        {   // its ties in index order: each thread a contiguous range, an inclusive scan orders the ranges
            const unsigned base = sh_n;
            const int nt = (int) blockDim.x, per = (vn + nt - 1) / nt, lo = (int) threadIdx.x * per, hi = min(vn, lo + per);
            unsigned mine = 0u;
            for (int v = lo; v < hi; ++v) mine += spec_ord(row[v]) == T;
            scan[threadIdx.x] = mine;
            __syncthreads();
            for (int off = 1; off < nt; off <<= 1) {
                const unsigned add = (int) threadIdx.x >= off ? scan[threadIdx.x - off] : 0u;
                __syncthreads();
                scan[threadIdx.x] += add;
                __syncthreads();
            }
            unsigned rank = scan[threadIdx.x] - mine;
            for (int v = lo; v < hi && rank < need; ++v)
                if (spec_ord(row[v]) == T) {
                    if (base + rank < 64u) cu[base + rank] = T, ci[base + rank] = v;
                    ++rank;
                }
        }
        __syncthreads();
        if ((int) threadIdx.x < k) {   // (the order: rank by the composite key)
            const unsigned ui = cu[threadIdx.x];
            const int ii = ci[threadIdx.x];
            int r = 0;
            for (int j = 0; j < k; ++j) r += cu[j] > ui || (cu[j] == ui && ci[j] < ii);
            list[2 + 2 * r] = ui;
            list[3 + 2 * r] = (unsigned) (v0 + ii);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        list[1] = (unsigned) k;
        __threadfence_system();
        *(volatile unsigned*) list = seq;
    }
}
// Philox 4x32-10 and the draft stream's key: sampler.cu's (src/kernels/cuda/sampler.cu), so a draft drawn here is the
// one sample_draft draws from the same row
__device__ __forceinline__ float spec_philox(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32), c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = __umulhi(0x9E3779B9u, c0), hi1 = __umulhi(0xBB67AE85u, c2);
        const uint32_t lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ (uint32_t) i, n2 = hi0 ^ c3;
        c0 = n0, c1 = lo1, c2 = n2, c3 = lo0;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}
constexpr uint64_t kSpecDraftKey = 0x5bd1e9955bd1e995ull;
// every member's list of the step (lists `stride` apart) -> the row's top k, then sample_draft's chain over it (top_p,
// min_p, temperature in double, one Philox draw): the draft into *tok and its E row (host-mapped BF16 [V][R]) into erow,
// and (q non-null) the kept list
__global__ void __launch_bounds__(1024) spec_pick_kernel(const unsigned* lists, int nm, size_t stride, unsigned seq,
                                                         SpecChain c, int* tok, const uint16_t* E, uint16_t* erow, int R,
                                                         strata::kernels::SpecList* q) {
    __shared__ unsigned raw[kSpecMembers * kSpecList];   // (the step's lists, read from host memory in one go)
    __shared__ int sel_ids[64];
    __shared__ float sel_logit[64];
    __shared__ int pick, n_all, at[kSpecMembers + 1];
    if ((int) threadIdx.x < nm)
        while ((int) (*(volatile const unsigned*) (lists + (size_t) threadIdx.x * stride) - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(32);
#endif
        }
    __syncthreads();
    // every list at once, 16 bytes a read (each read of host memory a PCIe request: a word at a time, and a list's
    // count before its entries, they were most of a step)
    {
        const uint4* src = (const uint4*) lists;
        uint4* dst = (uint4*) raw;
        const int per = (int) (kSpecList / 4), sper = (int) (stride / 4);
        for (int i = threadIdx.x; i < nm * per; i += blockDim.x) dst[i] = __ldcv(src + (size_t) (i / per) * sper + i % per);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int m = 0; m < nm; ++m) {
            at[m] = n;
            n += (int) raw[m * kSpecList + 1];
        }
        at[nm] = n;
        n_all = n;
    }
    __syncthreads();
    const int k = min(c.k, n_all);
    // the global top k: each entry's rank its place in its own list (descending) plus, in every other list, how many
    // come before it (a binary search: the lists are in the sampler's order, the indices distinct)
    for (int i = threadIdx.x; i < n_all; i += blockDim.x) {
        int m = 0;
        while (m + 1 < nm && at[m + 1] <= i) ++m;
        const int j = i - at[m];
        const unsigned ui = raw[m * kSpecList + 2 + 2 * j];
        const int ii = (int) raw[m * kSpecList + 3 + 2 * j];
        int r = j;
        for (int o = 0; o < nm && r < k; ++o) {
            if (o == m) continue;
            const unsigned* l = raw + o * kSpecList + 2;
            int lo = 0, hi = at[o + 1] - at[o];   // (the first of list o not before entry i)
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                const unsigned um = l[2 * mid];
                const int im = (int) l[2 * mid + 1];
                if (um > ui || (um == ui && im < ii)) lo = mid + 1;
                else hi = mid;
            }
            r += lo;
        }
        if (r < k) {
            sel_ids[r] = ii;
            const unsigned bb = (ui & 0x80000000u) ? (ui & 0x7fffffffu) : ~ui;   // (spec_ord's inverse)
            sel_logit[r] = __uint_as_float(bb);
        }
    }
    __syncthreads();
    // sampler.cu's chain, MODE 1 (sample_draft) - its serial sums in its order (the same bits), the exponentials and
    // divisions a thread an entry (double precision runs at 1/64 rate on these cards: ~130 of them in one thread were
    // most of a draft's step)
    __shared__ double ex[64];
    __shared__ int n_keep_s;
    __shared__ double sum_s;
    const float inv_t = c.temperature > 0.0f ? 1.0f / c.temperature : 0.0f;
    const float mx = sel_logit[0];   // (descending: the first is the largest)
    if (threadIdx.x == 0) n_keep_s = k;
    if (c.top_p < 1.0f) {
        if ((int) threadIdx.x < k) ex[threadIdx.x] = exp((double) sel_logit[threadIdx.x] - (double) mx);
        __syncthreads();
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (int i = 0; i < k; ++i) sum += ex[i];
            double cum = 0.0;
            int cut = k;
            for (int i = 0; i < k; ++i) {
                cum += ex[i] / sum;
                if (cum >= (double) c.top_p) {
                    cut = i + 1;
                    break;
                }
            }
            n_keep_s = cut < 1 ? 1 : cut;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && c.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(c.min_p);
        for (int i = 0; i < n_keep_s; ++i)
            if (sel_logit[i] < thresh) {
                n_keep_s = i;
                break;
            }
    }
    __syncthreads();
    const int n_keep = n_keep_s;
    const float smx = sel_logit[0] * inv_t;
    if ((int) threadIdx.x < n_keep) ex[threadIdx.x] = exp((double) (sel_logit[threadIdx.x] * inv_t) - (double) smx);
    __syncthreads();
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
        sum_s = sum;
    }
    __syncthreads();
    if ((int) threadIdx.x < n_keep) {
        ex[threadIdx.x] /= sum_s;
        if (q != nullptr) {
            q->ids[threadIdx.x] = sel_ids[threadIdx.x];
            q->p[threadIdx.x] = ex[threadIdx.x];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float u = spec_philox(c.seed ^ kSpecDraftKey, c.counter);
        double cum = 0.0;
        int pk = sel_ids[n_keep - 1];
        for (int i = 0; i < n_keep; ++i) {
            cum += ex[i];
            if ((double) u < cum) {
                pk = sel_ids[i];
                break;
            }
        }
        if (q != nullptr) q->n = n_keep;
        pick = pk;
        *tok = pk;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < R; i += blockDim.x) erow[i] = __ldcv(E + (size_t) pick * R + i);
}

__global__ void __launch_bounds__(1024) confidence_kernel(const float* __restrict__ x, const uint16_t* __restrict__ E,
                                                          const int* __restrict__ tok, const float* __restrict__ w,
                                                          int dim, int R, float* __restrict__ out) {
    const int t = blockIdx.x;
    const float* xt = x + (size_t) t * dim;
    const uint16_t* er = E + (size_t) tok[t] * R;
    float a = 0.0f;
    for (int i = threadIdx.x; i < dim + R; i += blockDim.x) a += w[i] * (i < dim ? xt[i] : bf16f(er[i - dim]));
    a = block_sum(a);
    if (threadIdx.x == 0) out[t] = 1.0f / (1.0f + expf(-a));
}

// ---------------------------------------------------------------- engram
__global__ void engram_dequant_kernel(const uint8_t* __restrict__ rows, const uint8_t* __restrict__ scales, int hd,
                                      float* __restrict__ out) {
    const int r = blockIdx.x;
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const __half_raw hv = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) rows[(size_t) r * hd + e], __NV_E4M3);
        const uint8_t sb = scales[(size_t) r * (hd / 32) + e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = __half2float(__half(hv)) * s;
    }
}

__global__ void __launch_bounds__(1024) engram_gate_kernel(float* __restrict__ h, const float* __restrict__ kv,
                                                           const float* __restrict__ w, int hc, int dim, float eps) {
    const int k = blockIdx.x, t = blockIdx.y;
    float* hk = h + ((size_t) t * hc + k) * dim;
    const float* kvt = kv + (size_t) t * (hc + 1) * dim;
    const float* key = kvt + (size_t) k * dim;
    const float* val = kvt + (size_t) hc * dim;
    const float* wk = w + (size_t) k * dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        const float hv = hk[e], kv_ = key[e];
        sh += hv * hv;
        sk += kv_ * kv_;
        sd += hv * wk[e] * kv_;
    }
    sh = block_sum(sh);
    sk = block_sum(sk);
    sd = block_sum(sd);
    const float rstd = rsqrtf(sh / (float) dim + eps) * rsqrtf(sk / (float) dim + eps);
    const float dot = sd * rstd * rsqrtf((float) dim);
    const float sq = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + expf(-sq));
    for (int e = threadIdx.x; e < dim; e += blockDim.x) hk[e] = hk[e] + gate * val[e];
}

__global__ void gather_kernel(const float* __restrict__ x, Gather g, int cols, float* __restrict__ out) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) out[(size_t) r * cols + c] = x[(size_t) g.src[r] * cols + c];
}

__global__ void combine_kernel(const float* __restrict__ ed, const float* __restrict__ cpu,
                               const float* __restrict__ sh, Combine cm, int dim, float* __restrict__ out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int j = 0; j < cm.n[t]; ++j) a += cm.w[t][j] * ed[(size_t) cm.row[t][j] * dim + e];
    if (cpu != nullptr && cm.cpu[t]) a += cpu[(size_t) t * dim + e];
    if (sh != nullptr) a += sh[(size_t) t * dim + e];
    out[(size_t) t * dim + e] = a;
}

__global__ void __launch_bounds__(512) route_publish_kernel(const float* __restrict__ logits,
                                                            const float* __restrict__ bias, int n, int k,
                                                            float route_scale, const unsigned long long* __restrict__ vram,
                                                            const float* __restrict__ x, int dim, int* idx, float* w,
                                                            unsigned long long* plan_ptr, Req* req,
                                                            const float* __restrict__ bias_vl, unsigned vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    __shared__ int on_cpu;
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl >> t) & 1u)) bias = bias_vl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        int cpu = 0;
        for (int r = 0; r < k; ++r) {
            const float wv = sc[pick[r]] / (s + 1e-20f) * route_scale;
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = wv;
            plan_ptr[(size_t) t * kPicks + r] = vram[pick[r]];
            cpu |= vram[pick[r]] == 0ull;
            if (req != nullptr) {
                req->ids[t][r] = pick[r];
                req->w[t][r] = wv;
            }
        }
        on_cpu = cpu;
    }
    // the token's FFN input to the host only where the CPU lane reads it (a pick of its on the CPU lane): each row
    // 20 KB over PCIe, where most windows have none
    __syncthreads();
    if (req != nullptr && on_cpu)
        for (int i = threadIdx.x; i < dim; i += blockDim.x) req->x[t][i] = x[(size_t) t * dim + i];
}

__global__ void publish_kernel(Req* req, int T, unsigned seq) {
    __threadfence_system();
    req->T = T;
    __threadfence_system();
    *(volatile unsigned*) &req->seq = seq;
}

__global__ void wait_seq_kernel(const unsigned* seq_ptr, unsigned seq) {
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) seq_ptr != seq) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}

__global__ void init_pre_kernel(float* pre, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * 8) pre[i] = (i % 8) == 0 ? 1.0f : 0.0f;
}


// a segment's copy unit: 16 bytes where its ends and its size allow, else 4, else 1 (the packed FP4 index keys' rows
// are 68 bytes: a window's rows start and end off the 16)
__host__ __device__ __forceinline__ size_t xfer_unit(const Xfer& x, int i) {
    const size_t a = (size_t) x.src[i] | (size_t) x.dst[i] | x.bytes[i];
    return (a & 15) == 0 ? 16 : (a & 3) == 0 ? 4 : 1;
}
__host__ __device__ __forceinline__ size_t xfer_units(const Xfer& x) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / xfer_unit(x, i);
    return total;
}

// the hand-offs through host-mapped memory: a grid-stride copy of the segments (a unit a thread), then (put, flag
// given) the last block to finish sets the flag after a system fence; a get first waits for the flag
__device__ __forceinline__ void xfer_copy(const Xfer& x, bool from_host) {
    const size_t total = xfer_units(x);
    for (size_t q = (size_t) blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (size_t) gridDim.x * blockDim.x) {
        size_t r = q;
        int i = 0;
        while (r >= x.bytes[i] / xfer_unit(x, i)) r -= x.bytes[i] / xfer_unit(x, i), ++i;
        const size_t u = xfer_unit(x, i);
        if (u == 16) {
            const uint4* src = (const uint4*) x.src[i] + r;
            ((uint4*) x.dst[i])[r] = from_host ? __ldcv(src) : *src;
        } else if (u == 4) {
            const unsigned* src = (const unsigned*) x.src[i] + r;
            ((unsigned*) x.dst[i])[r] = from_host ? __ldcv(src) : *src;
        } else {
            const unsigned char* src = (const unsigned char*) x.src[i] + r;
            ((unsigned char*) x.dst[i])[r] = from_host ? __ldcv(src) : *src;
        }
    }
}

__global__ void __launch_bounds__(kThreads) xfer_put_kernel(const Xfer x, unsigned* flag, unsigned seq, unsigned* cnt) {
    xfer_copy(x, false);
    if (flag == nullptr) return;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(cnt, 1u) == gridDim.x - 1) {
        *cnt = 0u;
        __threadfence_system();
        *(volatile unsigned*) flag = seq;
    }
}

__device__ __forceinline__ void wait_at_least(const unsigned* flag, unsigned seq) {
    while ((int) (*(volatile const unsigned*) flag - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
}

__global__ void __launch_bounds__(kThreads) xfer_get_kernel(const Xfer x, const unsigned* flag, unsigned seq) {
    if (threadIdx.x == 0) wait_at_least(flag, seq);
    __syncthreads();
    xfer_copy(x, true);
}

__global__ void wait_flags_kernel(const unsigned* flags, int n, int stride, unsigned seq) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) wait_at_least(flags + (size_t) i * stride, seq);
    __threadfence_system();
}
// a batch of table entries: *pairs[2i] = pairs[2i + 1] (the expert cache's recent slots, between windows)
__global__ void tab_set_kernel(const unsigned long long* __restrict__ pairs, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) *(unsigned long long*) pairs[2 * i] = pairs[2 * i + 1];
}
__global__ void set_flag_kernel(unsigned* flag, unsigned v) {
    __threadfence_system();
    *(volatile unsigned*) flag = v;
}

// three members or more: a reduce-scatter and an all-gather in one launch, a block a contiguous range of the row cut
// in nm chunks.  The block puts its range of this member's partial out and flags it; once every member's flag of the
// range is up it sums chunk mi of the range in member order (ar_sum_kernel's sums, bit for bit), puts that out and
// flags it; once every member's sum flag is up it reads the other chunks' sums.  Each member reads about two rows over
// PCIe whatever nm is, where every member reading every other row grows with nm
__global__ void __launch_bounds__(kThreads) ar_rs_kernel(float* x, int n4, float* rows, float* sums, size_t stride,
                                                        unsigned* bflags, int nm, int mi, int slot, int slots,
                                                        unsigned seq) {
    float4* x4 = (float4*) x;
    const int per = (n4 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int lo = blockIdx.x * per, hi = min(n4, lo + per), cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) { return bflags + (((size_t) m * slots + slot) * kArBlocks + blockIdx.x) * 8; };
    auto sflag = [&](int m) {
        return bflags + ((((size_t) nm + m) * slots + slot) * kArBlocks + blockIdx.x) * 8;
    };
    auto row = [&](int m) { return (float4*) (rows + ((size_t) m * slots + slot) * stride); };
    float4* sum = (float4*) (sums + (size_t) slot * stride);
    if (lo < hi) {
        float4* mine = row(mi);
        for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) mine[q] = x4[q];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) rflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(rflag(threadIdx.x), seq);
    __syncthreads();
    const int c0 = min(hi, lo + mi * cs), c1 = min(hi, c0 + cs);
    for (int q = c0 + threadIdx.x; q < c1; q += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? x4[q] : __ldcv(row(m) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        sum[q] = v;
        x4[q] = v;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) sflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(sflag(threadIdx.x), seq);
    __syncthreads();
    for (int q = lo + threadIdx.x; q < hi; q += blockDim.x)
        if (q < c0 || q >= c1) x4[q] = __ldcv(sum + q);
}

// ar_rs_kernel with the rows and the sums in FP16 (DS4_AR_F16): half the bytes over PCIe.  Every member's partial is
// rounded to FP16 (the owner's too), summed in F32 in member order, the sum rounded to FP16 - and every member, the
// owner too, takes that rounded sum, so the members' copies of the stream stay the same
__device__ __forceinline__ uint2 to_h4(float4 v) {
    const __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
    return make_uint2(*(const unsigned*) &a, *(const unsigned*) &b);
}
__device__ __forceinline__ float4 from_h4(uint2 u) {
    const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 sat4(float4 v) {   // (FP16's range: a partial past it saturates instead of inf)
    const float m = 65504.0f;
    return make_float4(fminf(fmaxf(v.x, -m), m), fminf(fmaxf(v.y, -m), m), fminf(fmaxf(v.z, -m), m), fminf(fmaxf(v.w, -m), m));
}
__global__ void __launch_bounds__(kThreads) ar_rs16_kernel(float* x, int n4, float* rows, float* sums, size_t stride,
                                                          unsigned* bflags, int nm, int mi, int slot, int slots,
                                                          unsigned seq) {
    float4* x4 = (float4*) x;
    const int per = (n4 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int lo = blockIdx.x * per, hi = min(n4, lo + per), cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) { return bflags + (((size_t) m * slots + slot) * kArBlocks + blockIdx.x) * 8; };
    auto sflag = [&](int m) {
        return bflags + ((((size_t) nm + m) * slots + slot) * kArBlocks + blockIdx.x) * 8;
    };
    auto row = [&](int m) { return (uint2*) (rows + ((size_t) m * slots + slot) * stride); };   // (FP16: half a row)
    uint2* sum = (uint2*) (sums + (size_t) slot * stride);
    if (lo < hi) {
        uint2* mine = row(mi);
        for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) mine[q] = to_h4(sat4(x4[q]));
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) rflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(rflag(threadIdx.x), seq);
    __syncthreads();
    const int c0 = min(hi, lo + mi * cs), c1 = min(hi, c0 + cs);
    for (int q = c0 + threadIdx.x; q < c1; q += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const uint2 u = m == mi ? to_h4(sat4(x4[q])) : __ldcv(row(m) + q);
            const float4 p = from_h4(u);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        const uint2 s16 = to_h4(sat4(v));
        sum[q] = s16;
        x4[q] = from_h4(s16);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) sflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(sflag(threadIdx.x), seq);
    __syncthreads();
    for (int q = lo + threadIdx.x; q < hi; q += blockDim.x)
        if (q < c0 || q >= c1) x4[q] = from_h4(__ldcv(sum + q));
}

// allreduce_dma's pieces: this member's partial as FP16 (ar_rs16's rounding), its slice's sum from the members' FP16
// partials in member order, and the other slices' sums back to F32
__global__ void ar_to16_kernel(const float4* __restrict__ x, int n4, uint2* __restrict__ h) {
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) h[q] = to_h4(sat4(x[q]));
}
__global__ void ar_sum16_kernel(float4* x, uint2* own, const uint2* __restrict__ peers, size_t pstride, int nm, int mi,
                                int q0, int q1) {
    for (int q = q0 + blockIdx.x * blockDim.x + threadIdx.x; q < q1; q += gridDim.x * blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = from_h4(m == mi ? own[q] : peers[(size_t) m * pstride + (q - q0)]);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        const uint2 s16 = to_h4(sat4(v));
        own[q] = s16;
        x[q] = from_h4(s16);
    }
}
// (the H2D side by the SMs' loads from host memory, the copy engine left to the experts' streaming: a set's sums do
// not queue behind its batches) - the others' partials of slice [q0, q1) read in place, every member's at once
__global__ void ar_sum16_zc_kernel(float4* x, uint2* own, const uint2* rows, size_t rstride, int nm, int mi, int q0,
                                   int q1) {
    for (int q = q0 + blockIdx.x * blockDim.x + threadIdx.x; q < q1; q += gridDim.x * blockDim.x) {
        uint2 u[8];
#pragma unroll
        for (int m = 0; m < 8; ++m)
            if (m < nm) u[m] = m == mi ? own[q] : __ldcv(rows + (size_t) m * rstride + q);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m >= nm) break;
            const float4 p = from_h4(u[m]);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        const uint2 s16 = to_h4(sat4(v));
        own[q] = s16;
        x[q] = from_h4(s16);
    }
}
__global__ void ar_from16_zc_kernel(float4* x, const uint2* hsum, int n4, int q0, int q1, int r0 = 0) {
    for (int q = r0 + blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x)
        if (q < q0 || q >= q1) x[q] = from_h4(__ldcv(hsum + q));
}
__global__ void ar_from16_kernel(float4* x, const uint2* __restrict__ h, int n4, int q0, int q1) {
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x)
        if (q < q0 || q >= q1) x[q] = from_h4(h[q]);
}

// x[i] = the members' partials summed in member order (this member's from x, the others' from their host rows)
__global__ void __launch_bounds__(kThreads) ar_sum_kernel(float* x, int n4, const float* peers, size_t stride,
                                                         const unsigned* flags, int nm, int mi, int slot, int slots,
                                                         unsigned seq) {
    if (threadIdx.x == 0)
        for (int m = 0; m < nm; ++m)
            if (m != mi) wait_at_least(flags + ((size_t) m * slots + slot) * 32, seq);
    __syncthreads();
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? ((const float4*) x)[q]
                                     : __ldcv((const float4*) (peers + ((size_t) m * slots + slot) * stride) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        ((float4*) x)[q] = v;
    }
}


// a thread a (token, value): every stream's value read before any is written
__global__ void hc_post_inplace_kernel(const float* __restrict__ y, float* h, const float* __restrict__ post,
                                       const float* __restrict__ comb, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float* r = h + (size_t) t * hc * dim;
    float rv[8];
    for (int j = 0; j < hc; ++j) rv[j] = r[(size_t) j * dim + e];
    const float yv = y[(size_t) t * dim + e];
    for (int k = 0; k < hc; ++k) {
        float a = post[(size_t) t * 8 + k] * yv;
        for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * rv[j];
        r[(size_t) k * dim + e] = a;
    }
}

__global__ void __launch_bounds__(kThreads) hc_mix_scale_kernel(const float* __restrict__ h, int hcdim, int n_mix,
                                                                float eps, float* mixes) {
    const int t = blockIdx.x;
    const float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) hcdim + eps);
    if (threadIdx.x < n_mix) mixes[(size_t) t * 32 + threadIdx.x] *= inv;
}

__global__ void engram_dequant_tok_kernel(const uint8_t* __restrict__ src, size_t per, int cols, int hd,
                                          float* __restrict__ out) {
    const int r = blockIdx.x, t = r / cols, c = r - t * cols;
    const uint8_t* rows = src + (size_t) t * per + (size_t) c * hd;
    const uint8_t* scales = src + (size_t) t * per + (size_t) cols * hd + (size_t) c * (hd / 32);
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const __half_raw hv = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) rows[e], __NV_E4M3);
        const uint8_t sb = scales[e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = __half2float(__half(hv)) * s;
    }
}

// a block a virtual slot: its picks counted (sort pass 1), or (fill) placed in token order by a block-wide scan
template <bool FILL>
__global__ void __launch_bounds__(kThreads) moe_sort_kernel(const int* __restrict__ idx, int T, int k,
                                                            const int* __restrict__ vs, int* bounds, int* row_tok,
                                                            int* pos) {
    const int slot = blockIdx.x;
    __shared__ int warp_n[kThreads / 32];
    __shared__ int base_s;
    if (threadIdx.x == 0) base_s = FILL ? bounds[slot] : 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int n = T * k;
    for (int q0 = 0; q0 < n; q0 += kThreads) {
        const int q = q0 + threadIdx.x;
        const int t = q / k, i = q - t * k;
        const bool hit = q < n && vs[idx[(size_t) t * kPicks + i]] == slot;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) warp_n[wp] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < kThreads / 32; ++w) {
            if (w < wp) before += warp_n[w];
            total += warp_n[w];
        }
        if (FILL && hit) {
            const int r = base_s + before + __popc(m & ((1u << lane) - 1u));
            row_tok[r] = t;
            pos[(size_t) t * kPicks + i] = r;
        }
        __syncthreads();
        if (threadIdx.x == 0) base_s += total;
        __syncthreads();
    }
    if (!FILL && threadIdx.x == 0) bounds[slot + 1] = base_s;   // (the counts; the scan makes them bounds)
}

__global__ void bounds_scan_kernel(int* bounds, int n_vs) {   // one thread: bounds[0] = 0, bounds[s + 1] += bounds[s]
    if (threadIdx.x != 0) return;
    bounds[0] = 0;
    for (int s = 0; s < n_vs; ++s) bounds[s + 1] += bounds[s];
}

__global__ void moe_combine_kernel(float* acc, const float* __restrict__ y, const int* __restrict__ pos,
                                   const float* __restrict__ w, int k, int r0, int r1, int dim) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    bool any = false;   // (a token without rows here leaves acc unread: a chunk's set covers a few of its tokens)
    for (int i = 0; i < k; ++i) {
        const int r = pos[(size_t) t * kPicks + i];
        any = any || (r >= r0 && r < r1);
    }
    if (!any) return;
    float a = acc[(size_t) t * dim + e];
    for (int i = 0; i < k; ++i) {
        const int r = pos[(size_t) t * kPicks + i];
        if (r < r0 || r >= r1) continue;
        a = __fmaf_rn(w[(size_t) t * kPicks + i], y[(size_t) (r - r0) * dim + e], a);
    }
    acc[(size_t) t * dim + e] = a;
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

void route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale,
                   const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                   unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl,
                   unsigned vl) {
    route_publish_kernel<<<T, 512, 0, s>>>(logits, bias, n, k, route_scale, vram, x, dim, idx, w, plan_ptr, req,
                                           vl != 0 ? bias_vl : nullptr, vl);
    if (req != nullptr) publish_kernel<<<1, 1, 0, s>>>(req, T, seq);
}
void wait_seq(const unsigned* seq_ptr, unsigned seq, cudaStream_t s) { wait_seq_kernel<<<1, 32, 0, s>>>(seq_ptr, seq); }
void init_pre(float* pre, int T, cudaStream_t s) { init_pre_kernel<<<cdiv(T * 8, 128), 128, 0, s>>>(pre, T); }

void embed(const uint16_t* rows, int T, float* h, int dim, int hc, cudaStream_t s) {
    embed_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(rows, h, dim, hc);
}

void finish(float* x, int n_vec, int len, long long stride, const Finish& f, cudaStream_t s) {
    if (len > kFinMax * kThreads) {
        std::fprintf(stderr, "ds4 finish: a vector of %d values is more than %d\n", len, kFinMax * kThreads);
        return;
    }
    if (n_vec <= 0) return;
    finish_kernel<<<n_vec, kThreads, 0, s>>>(x, len, stride, f);
}

void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
    hc_mix_kernel<<<dim3(n_mix, T), kThreads, 0, s>>>(h, fn, hcdim, eps, mixes);
}
void hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb, cudaStream_t s) {
    hc_split_kernel<<<T, 32, 0, s>>>(mixes, scale, base, hc, iters, hc_eps, pre, post, comb);
}
void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 cudaStream_t s) {
    hc_pre_norm_kernel<<<T, 1024, 0, s>>>(h, mix, w, dim, hc, eps, x);
}
void hc_post(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T,
             float* out, cudaStream_t s) {
    hc_post_kernel<<<dim3(cdiv(dim * hc, kThreads), T), kThreads, 0, s>>>(y, res, post, comb, dim, hc, out);
}

void hc_post_inplace(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T,
                     cudaStream_t s) {
    hc_post_inplace_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(y, h, post, comb, dim, hc);
}
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
    hc_mix_scale_kernel<<<T, kThreads, 0, s>>>(h, hcdim, n_mix, eps, mixes);
}
void engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, cudaStream_t s) {
    if (T > 0) engram_dequant_tok_kernel<<<T * cols, kThreads, 0, s>>>(src, per, cols, hd, out);
}
void moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos,
              cudaStream_t s) {
    moe_sort_kernel<false><<<n_vs, kThreads, 0, s>>>(idx, T, k, vs, bounds, row_tok, pos);
    bounds_scan_kernel<<<1, 32, 0, s>>>(bounds, n_vs);
    moe_sort_kernel<true><<<n_vs, kThreads, 0, s>>>(idx, T, k, vs, bounds, row_tok, pos);
}
void moe_combine(float* acc, const float* y, const int* pos, const float* w, int T, int k, int r0, int r1, int dim,
                 cudaStream_t s) {
    if (r1 > r0 && T > 0) moe_combine_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(acc, y, pos, w, k, r0, r1, dim);
}

void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, cudaStream_t s) {
    gemv_f16_kernel<<<dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s>>>(W, x, n, k, alpha, y);
}
void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y, size_t ldy,
                   cudaStream_t s) {
    if (T > 0)
        gemv_f16_rows_kernel<<<dim3(cdiv(n, kThreads / 32), cdiv(T, kRowsT)), kThreads, 0, s>>>(W, x, ldx, n, k, T, alpha,
                                                                                               y, ldy);
}

void route(const float* logits, const float* bias, int n, int k, int T, float route_scale, int* idx, float* w,
           cudaStream_t s, const float* bias_vl, const VlBits* vl) {
    static const VlBits none = {};
    route_kernel<<<T, 512, 0, s>>>(logits, bias, n, k, route_scale, idx, w, vl != nullptr ? bias_vl : nullptr,
                                   vl != nullptr ? *vl : none);
}

void swiglu(const float* g, const float* u, int n, float lim, float* h, cudaStream_t s) {
    if (n > 0) swiglu_kernel<<<cdiv(n, kThreads), kThreads, 0, s>>>(g, u, n, lim, h);
}

void compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent, cudaStream_t s) {
    compress_kernel<<<1, 512, 0, s>>>(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent);
}

void index_score(const float* q, const float* w, const uint16_t* k_bf16, const uint8_t* k_packed, int heads, int hd,
                 int T, int n, const int* cand, int cand_stride, int blk, const Rows& lim, float* out,
                 long long out_stride, cudaStream_t s, bool tc) {
    if (hd != 128 || heads > 32) {
        std::fprintf(stderr, "ds4 index_score: %d heads of %d not covered\n", heads, hd);
        return;
    }
    if (n <= 0) return;
    // (opt-in, DS4_INDEX_TC=1: its sums differ from the CUDA cores' in the last bits - perplexity the same, 7.5366 on
    // 4 x 1024 tokens, but a long greedy run's tokens part after a while)
    static const bool tc_ok = std::getenv("DS4_INDEX_TC") != nullptr && std::atoi(std::getenv("DS4_INDEX_TC")) != 0;
    if (tc && tc_ok && k_packed != nullptr) {   // (the tensor cores: a block a tile, looping, ~2 blocks an SM)
        static std::atomic<uint64_t> opted{0};
        static int sms[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (!(opted.load(std::memory_order_relaxed) >> dev & 1u)) {
            cudaFuncSetAttribute(index_score_tc_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) index_tc_smem<1>());
            cudaFuncSetAttribute(index_score_tc_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) index_tc_smem<2>());
            cudaFuncSetAttribute(index_score_tc_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) index_tc_smem<3>());
            cudaFuncSetAttribute(index_score_tc_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) index_tc_smem<4>());
            cudaFuncSetAttribute(index_score_tc_kernel<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) index_tc_smem<5>());
            cudaFuncSetAttribute(index_score_tc_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) index_tc_smem<6>());
            cudaDeviceGetAttribute(&sms[dev & 63], cudaDevAttrMultiProcessorCount, dev);
            opted.fetch_or(1ull << dev);
        }
        // queries a block: 1 with candidates (each its own rows); else a decode window's all (up to 6), a slab's 4
        const int qt = cand != nullptr ? 1 : T <= 6 ? T : 4, qg = (T + qt - 1) / qt, ntiles = (n + 127) / 128;
        const int gx = std::max(1, std::min(ntiles, 2 * sms[dev & 63] / qg));
        switch (qt) {
        case 1: index_tc_launch<1>(gx, qg, s, q, w, k_packed, heads, n, T, cand, cand_stride, blk, lim, out, out_stride, ntiles); break;
        case 2: index_tc_launch<2>(gx, qg, s, q, w, k_packed, heads, n, T, cand, cand_stride, blk, lim, out, out_stride, ntiles); break;
        case 3: index_tc_launch<3>(gx, qg, s, q, w, k_packed, heads, n, T, cand, cand_stride, blk, lim, out, out_stride, ntiles); break;
        case 5: index_tc_launch<5>(gx, qg, s, q, w, k_packed, heads, n, T, cand, cand_stride, blk, lim, out, out_stride, ntiles); break;
        case 6: index_tc_launch<6>(gx, qg, s, q, w, k_packed, heads, n, T, cand, cand_stride, blk, lim, out, out_stride, ntiles); break;
        default: index_tc_launch<4>(gx, qg, s, q, w, k_packed, heads, n, T, cand, cand_stride, blk, lim, out, out_stride, ntiles); break;
        }
        return;
    }
    const size_t smem = (size_t) (heads * hd + heads) * sizeof(float);
    index_score_kernel<128><<<dim3(cdiv(n, kThreads), T), kThreads, smem, s>>>(q, w, k_bf16, k_packed, heads, n, cand,
                                                                                cand_stride, blk, lim, out,
                                                                                out_stride);
}

void topk(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
          int* out, int out_stride, cudaStream_t s, const TopkScratch* sc) {
    if (sc != nullptr && T <= sc->T) {   // (a few long rows: spread over segments)
        int nmax = 0, kmax = 0;
        for (int t = 0; t < T; ++t) {
            int n = 0, k = 0;
            if (r.gen == 0) n = r.n[t], k = r.k[t];
            else {
                const int nn = (int) ((r.p0 + t + 1) / r.R), cb = std::min(r.ccap, (nn + r.blk - 1) / r.blk);
                switch (r.gen) {
                case 1: n = nn, k = std::min(r.kcap, nn); break;
                case 2: n = nn, k = cb; break;
                case 3: n = cb * r.blk, k = std::min(r.kcap, nn); break;
                default: n = (nn + r.blk - 1) / r.blk, k = r.kcap; break;
                }
            }
            nmax = std::max(nmax, n), kmax = std::max(kmax, k);
        }
        if (nmax >= kTopkSegMin && kmax <= sc->K) {
            const int P = std::min(sc->P, std::max(2, (nmax + kTopkSeg - 1) / kTopkSeg));
            topk_seg_kernel<<<dim3(P, T), 1024, 0, s>>>(vals, stride, r, P, sc->K, sc->val, sc->idx, sc->cnt, sc->kth);
            static std::atomic<uint64_t> big_smem{0};   // (the devices the merge's shared memory was opted in on)
            int dev = 0;
            cudaGetDevice(&dev);
            if (!(big_smem.load(std::memory_order_relaxed) >> dev & 1u)) {
                cudaFuncSetAttribute(topk_merge_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMergeCap * 8);
                big_smem.fetch_or(1ull << dev);
            }
            topk_merge_kernel<<<T, 1024, (size_t) kMergeCap * 8, s>>>(r, P, sc->K, sc->val, sc->idx, sc->cnt, sc->kth, cand,
                                                                      cand_stride, blk, out, out_stride);
            return;
        }
    }
    topk_kernel<<<T, 1024, 0, s>>>(vals, stride, r, cand, cand_stride, blk, out, out_stride);
}

void cand_blocks(const float* score, long long stride, int T, const Rows& r, int blk, float* bm, long long bm_stride,
                 int* out, int out_stride, cudaStream_t s) {
    int nb_max = 1;
    Rows rb = r;
    if (r.gen != 0) {   // (a chunk: the widest query's blocks, made on the device per query)
        rb.gen = 4;
        rb.blk = blk;
        nb_max = cdiv((int) ((r.p0 + T) / r.R), blk);
    } else
        for (int t = 0; t < T; ++t) {
            rb.n[t] = cdiv(r.n[t], blk);
            rb.k[t] = r.k[t];
            nb_max = rb.n[t] > nb_max ? rb.n[t] : nb_max;
        }
    block_max_kernel<<<dim3(cdiv(nb_max, kThreads), T), kThreads, 0, s>>>(score, stride, r, blk, bm, bm_stride);
    topk(bm, bm_stride, T, rb, nullptr, 0, 1, out, out_stride, s);
}

void sparse_attn(const float* q, const uint16_t* win, const uint16_t* ckv, const uint8_t* ckv_packed, const int* sel,
                 int sel_stride, const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                 cudaStream_t s) {
    if (d != 512) {
        std::fprintf(stderr, "ds4 sparse_attn: head dim %d not covered\n", d);
        return;
    }
    // HG heads a block where the queries fill the device with them (a prompt's chunk), else a block a head (decode's
    // few queries); DS4_ATTN_HG=1 / 8 / 16 picks one (the same bits whichever)
    const char* hg_s = std::getenv("DS4_ATTN_HG");   // (read each call: the parity test switches it)
    const int hg_env = hg_s != nullptr ? std::atoi(hg_s) : 0;
    const int hg = hg_env > 0 ? hg_env : (heads % 16 == 0 && T >= 64) ? 16 : (heads % 8 == 0 && T >= 128) ? 8 : 1;
    const size_t smem = (size_t) kAttTile * 512 * 2;
    if (hg == 16 && heads % 16 == 0) {
        static bool set[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (!set[dev & 63]) {
            cudaFuncSetAttribute(sparse_attn_hg_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
            set[dev & 63] = true;
        }
        sparse_attn_hg_kernel<16><<<dim3(heads / 16, T), 16 * 32, smem, s>>>(q, win, ckv, ckv_packed, sel, sel_stride, a,
                                                                              sink, heads, scale, out);
    } else if (hg == 8 && heads % 8 == 0) {
        static bool set[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (!set[dev & 63]) {
            cudaFuncSetAttribute(sparse_attn_hg_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
            set[dev & 63] = true;
        }
        sparse_attn_hg_kernel<8><<<dim3(heads / 8, T), 8 * 32, smem, s>>>(q, win, ckv, ckv_packed, sel, sel_stride, a,
                                                                          sink, heads, scale, out);
    } else
        sparse_attn_kernel<<<dim3(heads, T), kThreads, 0, s>>>(q, win, ckv, ckv_packed, sel, sel_stride, a, sink, heads,
                                                               d, scale, out);
}

void ring_to_ext(const uint16_t* ring, uint16_t* ext, int64_t p0, int n, int d, cudaStream_t s) {
    ring_to_ext_kernel<<<n, kThreads, 0, s>>>(ring, ext, (long long) p0, n, d);
}
void ext_to_ring(const uint16_t* ext, uint16_t* ring, int64_t p0, int n, int T, int d, cudaStream_t s,
                 uint16_t* backup) {
    ext_to_ring_kernel<<<T, kThreads, 0, s>>>(ext, ring, (long long) p0, n, T, d, backup);
}
void ring_restore(uint16_t* ring, const uint16_t* backup, int64_t p0, int n, int keep, int T, int d, cudaStream_t s) {
    if (T > keep) ring_restore_kernel<<<T - keep, kThreads, 0, s>>>(ring, backup, (long long) p0, n, keep, d);
}
void ring_restore_batch(const RingBatch& b, int64_t p0, int n, int keep, int T, int d, cudaStream_t s) {
    if (T > keep && b.n > 0)
        ring_restore_batch_kernel<<<dim3(T - keep, b.n), kThreads, 0, s>>>(b, (long long) p0, n, keep, d);
}
void comp_restore_batch(const CompBatch& b, int64_t p0, int keep, int d, cudaStream_t s) {
    if (b.n > 0) comp_restore_batch_kernel<<<b.n, kThreads, 0, s>>>(b, (long long) p0, keep, d);
}
void hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride, cudaStream_t s) {
    hc_mean_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(h, hc, dim, out, out_stride);
}
void iota_rows(int* out, int T, int n, int stride, cudaStream_t s) { iota_rows_kernel<<<T, 64, 0, s>>>(out, n, stride); }
void markov_add(float* logits, const uint16_t* E, const uint16_t* H, const int* tok, int V, int R, cudaStream_t s) {
    markov_kernel<<<cdiv(V, kThreads / 32), kThreads, (size_t) R * sizeof(float), s>>>(logits, E, H, tok, V, R);
}
void markov_share(float* logits, const uint16_t* H, const uint16_t* erow, int vn, int R, cudaStream_t s) {
    markov_share_kernel<<<cdiv(vn, kThreads / 32), kThreads, (size_t) R * sizeof(float), s>>>(logits, H, erow, vn, R);
}
void argmax_share(const float* row, int vn, int v0, unsigned* slot, unsigned seq, cudaStream_t s) {
    argmax_share_kernel<<<1, 1024, 0, s>>>(row, vn, v0, slot, seq);
}
void markov_pick(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E, uint16_t* erow, int R,
                 cudaStream_t s) {
    markov_pick_kernel<<<1, 256, 0, s>>>(slots, nm, seq, tok, E, erow, R);
}
void topk_share(const float* row, int vn, int v0, int k, unsigned* list, unsigned seq, cudaStream_t s) {
    topk_share_kernel<<<1, 1024, 0, s>>>(row, vn, v0, std::min(64, std::max(1, k)), list, seq);
}
void spec_pick(const unsigned* lists, int nm, size_t stride, unsigned seq, const SpecChain& c, int* tok, const uint16_t* E,
               uint16_t* erow, int R, strata::kernels::SpecList* q, cudaStream_t s) {
    spec_pick_kernel<<<1, 1024, 0, s>>>(lists, nm, stride, seq, c, tok, E, erow, R, q);
}
void confidence(const float* x, const uint16_t* E, const int* tok, const float* w, int T, int dim, int R, float* out,
                cudaStream_t s) {
    confidence_kernel<<<T, 1024, 0, s>>>(x, E, tok, w, dim, R, out);
}
void engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, cudaStream_t s) {
    engram_dequant_kernel<<<n, kThreads, 0, s>>>(rows, scales, hd, out);
}

void engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps, cudaStream_t s) {
    engram_gate_kernel<<<dim3(hc, T), 1024, 0, s>>>(h, kv, w, hc, dim, eps);
}

void gather_rows(const float* x, const Gather& g, int n, int cols, float* out, cudaStream_t s) {
    if (n > 0) gather_kernel<<<dim3(cdiv(cols, kThreads), n), kThreads, 0, s>>>(x, g, cols, out);
}

void combine(const float* ed, const float* cpu, const float* sh, const Combine& c, int T, int dim, float* out,
             cudaStream_t s) {
    combine_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(ed, cpu, sh, c, dim, out);
}

void xfer_put(const Xfer& x, unsigned* flag, unsigned seq, unsigned* cnt, cudaStream_t s) {
    const size_t total = xfer_units(x);
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    xfer_put_kernel<<<blocks, kThreads, 0, s>>>(x, flag, seq, cnt);
}
void xfer_get(const Xfer& x, const unsigned* flag, unsigned seq, cudaStream_t s) {
    const size_t total = xfer_units(x);
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    xfer_get_kernel<<<blocks, kThreads, 0, s>>>(x, flag, seq);
}
void wait_flags(const unsigned* flags, int n, int stride, unsigned seq, cudaStream_t s) {
    if (n > 0) wait_flags_kernel<<<1, 32, 0, s>>>(flags, n, stride, seq);
}
void set_flag(unsigned* flag, unsigned v, cudaStream_t s) { set_flag_kernel<<<1, 1, 0, s>>>(flag, v); }
void tab_set(const unsigned long long* pairs, int n, cudaStream_t s) {
    if (n > 0) tab_set_kernel<<<cdiv(n, 128), 128, 0, s>>>(pairs, n);
}
void allreduce(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, cudaStream_t s) {
    const int slot = (int) (seq % (unsigned) a.slots);
    if (a.nm > 2 && a.sums != nullptr && a.bflags != nullptr) {   // (the reduce-scatter: ar_rs_kernel)
        static const bool f16 = [] {
            const char* e = std::getenv("DS4_AR_F16");
            return e == nullptr || std::atoi(e) != 0;   // (on unless DS4_AR_F16=0)
        }();
        const int n4 = n / 4;
        const int blocks = std::max(1, std::min(kArBlocks, cdiv(n4, kThreads)));
        if (f16)
            ar_rs16_kernel<<<blocks, kThreads, 0, s>>>(x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        else
            ar_rs_kernel<<<blocks, kThreads, 0, s>>>(x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        return;
    }
    Xfer xp;
    xp.n = 1;
    xp.src[0] = x;
    xp.dst[0] = a.data + ((size_t) mi * a.slots + slot) * a.stride;
    xp.bytes[0] = (size_t) n * 4;
    xfer_put(xp, a.flags + ((size_t) mi * a.slots + slot) * 32, seq, cnt, s);
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));
    ar_sum_kernel<<<blocks, kThreads, 0, s>>>(x, n4, a.data, a.stride, a.flags, a.nm, mi, slot, a.slots, seq);
}

void allreduce_dma(float* x, int n, const ArGroup& a, int mi, unsigned seq, void* scratch, cudaStream_t s,
                   const ArAux* aux) {
    const int slot = (int) (seq % (unsigned) a.slots), nm = a.nm;
    const int n4 = n / 4, cq = (n4 + nm - 1) / nm;
    const int c0 = std::min(n4, mi * cq), c1 = std::min(n4, (mi + 1) * cq);
    uint2* own = (uint2*) scratch;   // this member's FP16 partial (a uint2 a float4), then the sums
    uint2* peers = own + n4;         // [nm][cq]: the members' partials of slice mi
    auto hrow = [&](int m) { return (uint2*) (a.data + ((size_t) m * a.slots + slot) * a.stride); };
    uint2* hsum = (uint2*) (a.sums + (size_t) slot * a.stride);
    // (ar_rs16_kernel's first block's flags: member m's partial out at rf + m * fs, its sums at sf + m * fs)
    unsigned* rf = a.bflags + (size_t) slot * kArBlocks * 8;
    unsigned* sf = a.bflags + ((size_t) nm * a.slots + slot) * kArBlocks * 8;
    const int fs = a.slots * kArBlocks * 8;
    const int blocks = std::max(1, std::min(1024, cdiv(n4, kThreads)));
    ar_to16_kernel<<<blocks, kThreads, 0, s>>>((const float4*) x, n4, own);
    // the H2D side by the SMs up to 4 members (16K-token prompt on 9 GPUs, --tp 4: 743 -> 785 tok/s), the copy
    // engine past that (--tp 8: 719 by the copy engine, 676 by the SMs); DS4_AR_ZC=0 / 1 / 2.. sets the most members
    static const int zc_max = [] {
        const char* e = std::getenv("DS4_AR_ZC");
        return e != nullptr ? std::atoi(e) : 4;
    }();
    if (nm <= std::min(8, zc_max) && aux != nullptr && aux->s2 != nullptr) {
        // in kArPieces pieces, each cut in nm slices: the copies out on s2 (the D2H engine) beside the sums' and the
        // gather's reads on s (the SMs, H2D) - piece p + 1 goes out while piece p is summed; ar_rs16's flags, a
        // block's slot a piece.  (The first flags wait for every member's partial of the piece; s's last gather waits
        // for this member's own copies too, so c.y is free again past it.)
        const int P4 = cdiv(n4, kArPieces);
        auto piece = [&](int p, int& p0, int& p1, int& q0, int& q1) {
            p0 = std::min(n4, p * P4), p1 = std::min(n4, p0 + P4);
            const int cqp = cdiv(std::max(1, p1 - p0), nm);
            q0 = std::min(p1, p0 + mi * cqp), q1 = std::min(p1, q0 + cqp);
        };
        auto rfp = [&](int p) { return a.bflags + ((size_t) slot * kArBlocks + p) * 8; };
        auto sfp = [&](int p) { return a.bflags + (((size_t) nm * a.slots + slot) * kArBlocks + p) * 8; };
        cudaEventRecord(aux->ev[0], s);
        cudaStreamWaitEvent(aux->s2, aux->ev[0], 0);
        for (int p = 0; p < kArPieces; ++p) {
            int p0, p1, q0, q1;
            piece(p, p0, p1, q0, q1);
            if (q0 > p0) cudaMemcpyAsync(hrow(mi) + p0, own + p0, (size_t) (q0 - p0) * 8, cudaMemcpyDeviceToHost, aux->s2);
            if (p1 > q1) cudaMemcpyAsync(hrow(mi) + q1, own + q1, (size_t) (p1 - q1) * 8, cudaMemcpyDeviceToHost, aux->s2);
            set_flag_kernel<<<1, 1, 0, aux->s2>>>(rfp(p) + (size_t) mi * fs, seq);
        }
        for (int p = 0; p < kArPieces; ++p) {
            int p0, p1, q0, q1;
            piece(p, p0, p1, q0, q1);
            wait_flags_kernel<<<1, 32, 0, s>>>(rfp(p), nm, fs, seq);
            if (q1 > q0)
                ar_sum16_zc_kernel<<<std::max(1, std::min(1024, cdiv(q1 - q0, kThreads))), kThreads, 0, s>>>(
                    (float4*) x, own, hrow(0), a.slots * a.stride / 2, nm, mi, q0, q1);
            cudaEventRecord(aux->ev[1 + p], s);
        }
        for (int p = 0; p < kArPieces; ++p) {
            int p0, p1, q0, q1;
            piece(p, p0, p1, q0, q1);
            cudaStreamWaitEvent(aux->s2, aux->ev[1 + p], 0);
            if (q1 > q0) cudaMemcpyAsync(hsum + q0, own + q0, (size_t) (q1 - q0) * 8, cudaMemcpyDeviceToHost, aux->s2);
            set_flag_kernel<<<1, 1, 0, aux->s2>>>(sfp(p) + (size_t) mi * fs, seq);
        }
        for (int p = 0; p < kArPieces; ++p) {
            int p0, p1, q0, q1;
            piece(p, p0, p1, q0, q1);
            wait_flags_kernel<<<1, 32, 0, s>>>(sfp(p), nm, fs, seq);
            if (p1 > p0)
                ar_from16_zc_kernel<<<std::max(1, std::min(1024, cdiv(p1 - p0, kThreads))), kThreads, 0, s>>>(
                    (float4*) x, hsum, p1, q0, q1, p0);
        }
        return;
    }
    if (c0 > 0) cudaMemcpyAsync(hrow(mi), own, (size_t) c0 * 8, cudaMemcpyDeviceToHost, s);   // (the others' slices)
    if (c1 < n4) cudaMemcpyAsync(hrow(mi) + c1, own + c1, (size_t) (n4 - c1) * 8, cudaMemcpyDeviceToHost, s);
    set_flag_kernel<<<1, 1, 0, s>>>(rf + (size_t) mi * fs, seq);
    wait_flags_kernel<<<1, 32, 0, s>>>(rf, nm, fs, seq);
    if (nm <= std::min(8, zc_max)) {
        const int nb = std::max(1, std::min(1024, cdiv(c1 - c0, kThreads)));
        if (c1 > c0) {
            ar_sum16_zc_kernel<<<nb, kThreads, 0, s>>>((float4*) x, own, hrow(0), a.slots * a.stride / 2, nm, mi, c0, c1);
            cudaMemcpyAsync(hsum + c0, own + c0, (size_t) (c1 - c0) * 8, cudaMemcpyDeviceToHost, s);
        }
        set_flag_kernel<<<1, 1, 0, s>>>(sf + (size_t) mi * fs, seq);
        wait_flags_kernel<<<1, 32, 0, s>>>(sf, nm, fs, seq);
        ar_from16_zc_kernel<<<blocks, kThreads, 0, s>>>((float4*) x, hsum, n4, c0, c1);
        return;
    }
    if (c1 > c0) {
        for (int m = 0; m < nm; ++m)
            if (m != mi)
                cudaMemcpyAsync(peers + (size_t) m * cq, hrow(m) + c0, (size_t) (c1 - c0) * 8, cudaMemcpyHostToDevice, s);
        ar_sum16_kernel<<<std::max(1, std::min(1024, cdiv(c1 - c0, kThreads))), kThreads, 0, s>>>(
            (float4*) x, own, peers, (size_t) cq, nm, mi, c0, c1);
        cudaMemcpyAsync(hsum + c0, own + c0, (size_t) (c1 - c0) * 8, cudaMemcpyDeviceToHost, s);
    }
    set_flag_kernel<<<1, 1, 0, s>>>(sf + (size_t) mi * fs, seq);
    wait_flags_kernel<<<1, 32, 0, s>>>(sf, nm, fs, seq);
    if (c0 > 0) cudaMemcpyAsync(own, hsum, (size_t) c0 * 8, cudaMemcpyHostToDevice, s);
    if (c1 < n4) cudaMemcpyAsync(own + c1, hsum + c1, (size_t) (n4 - c1) * 8, cudaMemcpyHostToDevice, s);
    ar_from16_kernel<<<blocks, kThreads, 0, s>>>((float4*) x, own, n4, c0, c1);
}

// ---------------------------------------------------------------- the host-summed all-reduce
int ar_host_pieces() {
    static const int n = [] {
        const char* e = std::getenv("DS4_AR_HOST_PIECES");
        return std::max(1, std::min(kArBlocks, e != nullptr ? std::atoi(e) : 4));
    }();
    return n;
}
namespace {
int cuda_numa_node(int dev) {   // (-1: unknown)
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) return -1;
    std::string id(bus);
    for (char& c : id) c = (char) std::tolower((unsigned char) c);
    std::FILE* f = std::fopen(("/sys/bus/pci/devices/" + id + "/numa_node").c_str(), "r");
    if (f == nullptr) return -1;
    int node = -1;
    if (std::fscanf(f, "%d", &node) != 1) node = -1;
    std::fclose(f);
    return node;
}
// pinned host memory on NUMA node `node` (-1: anywhere), mapped for every device
void* numa_pinned(size_t bytes, int node) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    if (node >= 0) {
        unsigned long mask[4] = {0, 0, 0, 0};
        mask[node / 64] = 1ul << (node % 64);
        syscall(SYS_mbind, p, bytes, 2 /* MPOL_BIND */, mask, 256, 0);
    }
    std::memset(p, 0, bytes);
    if (cudaHostRegister(p, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        munmap(p, bytes);
        return nullptr;
    }
    return p;
}
}  // namespace

struct HostReducer::Impl {
    ArGroup a;
    size_t max_n = 0, bytes = 0;
    std::vector<uint16_t*> rows;   // [member]: slots x max_n
    uint16_t* sums = nullptr;      // slots x max_n
    std::unique_ptr<Pool> pool;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<unsigned, int>> q;
    bool quit = false;
    std::thread th;
    double t_wait = 0, t_sum = 0;   // (DS4_AR_HOST_STATS: printed at the end)
    long n_sums = 0;
    void loop() {
        for (;;) {
            std::pair<unsigned, int> job;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !q.empty(); });
                if (q.empty()) return;
                job = q.front();
                q.pop_front();
            }
            const unsigned seq = job.first;
            const int slot = (int) (seq % (unsigned) a.slots), nm = a.nm, n4 = job.second / 4;
            const int NP = ar_host_pieces(), P4 = (n4 + NP - 1) / NP, fs = a.slots * kArBlocks * 8;
            const uint16_t* rp[16];
            for (int m = 0; m < nm && m < 16; ++m) rp[m] = rows[(size_t) m] + (size_t) slot * max_n;
            uint16_t* out = sums + (size_t) slot * max_n;
            for (int p = 0; p < NP; ++p) {
                const size_t h0 = (size_t) std::min(n4, p * P4) * 4, h1 = (size_t) std::min(n4, (p + 1) * P4) * 4;
                const unsigned* rf = a.bflags + ((size_t) slot * kArBlocks + p) * 8;
                const auto tw = std::chrono::steady_clock::now();
                for (int m = 0; m < nm; ++m) {   // (every member's piece out: its flag, after its copy)
                    int spins = 0;
                    while ((int) (*(volatile const unsigned*) (rf + (size_t) m * fs) - seq) < 0) {
                        if (++spins > 2000) std::this_thread::sleep_for(std::chrono::microseconds(20));
                        else _mm_pause();
                    }
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                const auto ts = std::chrono::steady_clock::now();
                t_wait += std::chrono::duration<double, std::milli>(ts - tw).count();
                const int T = pool->size();
                pool->run(T, [&](int t) {
                    const size_t a0 = h0 + (h1 - h0) * (size_t) t / (size_t) T / 8 * 8,
                                 a1 = t + 1 == T ? h1 : h0 + (h1 - h0) * (size_t) (t + 1) / (size_t) T / 8 * 8;
                    if (a1 > a0) ar_host_sum(rp, nm, out, a0, a1);
                });
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *(volatile unsigned*) (a.bflags + (((size_t) nm * a.slots + slot) * kArBlocks + p) * 8) = seq;
                t_sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
                ++n_sums;
            }
        }
    }
};
HostReducer::HostReducer(const ArGroup& a, const std::vector<int>& devs, size_t max_n, int threads) : p_(new Impl) {
    p_->a = a;
    p_->max_n = (max_n + 63) / 64 * 64;
    p_->bytes = (size_t) a.slots * p_->max_n * 2;
    std::vector<int> nodes;
    for (int dv : devs) nodes.push_back(cuda_numa_node(dv));
    int sum_node = -1, best = 0;
    for (int nd : nodes) {
        const int c = (int) std::count(nodes.begin(), nodes.end(), nd);
        if (nd >= 0 && c > best) best = c, sum_node = nd;
    }
    for (size_t m = 0; m < devs.size(); ++m) p_->rows.push_back((uint16_t*) numa_pinned(p_->bytes, nodes[m]));
    p_->sums = (uint16_t*) numa_pinned(p_->bytes, sum_node);
    std::vector<int> cpus;   // (DS4_AR_HOST_PIN=1: pinned to the physical cores, both nodes interleaved)
    if (const char* pe = std::getenv("DS4_AR_HOST_PIN"); pe != nullptr && std::atoi(pe) != 0)
        for (int c = 0; c < (int) std::thread::hardware_concurrency() / 2; ++c) cpus.push_back(c);
    p_->pool = std::make_unique<Pool>(std::max(0, threads - 1), cpus, 2000);
    p_->th = std::thread([this] { p_->loop(); });
}
bool HostReducer::ok() const {
    if (p_->sums == nullptr) return false;
    for (uint16_t* r : p_->rows)
        if (r == nullptr) return false;
    return true;
}
uint16_t* HostReducer::row(int m, int slot) const { return p_->rows[(size_t) m] + (size_t) slot * p_->max_n; }
uint16_t* HostReducer::sum(int slot) const { return p_->sums + (size_t) slot * p_->max_n; }
HostReducer::~HostReducer() {
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->quit = true;
    }
    p_->cv.notify_all();
    if (p_->th.joinable()) p_->th.join();
    if (std::getenv("DS4_AR_HOST_STATS") != nullptr)
        std::fprintf(stderr, "host all-reduce: %ld pieces summed, waiting %.1f ms, summing %.1f ms\n", p_->n_sums,
                     p_->t_wait, p_->t_sum);
    for (uint16_t* r : p_->rows)
        if (r != nullptr) {
            cudaHostUnregister(r);
            munmap(r, p_->bytes);
        }
    if (p_->sums != nullptr) {
        cudaHostUnregister(p_->sums);
        munmap(p_->sums, p_->bytes);
    }
    delete p_;
}
void HostReducer::submit(unsigned seq, int n) {
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->q.push_back({seq, n});
    }
    p_->cv.notify_one();
}

void allreduce_host(float* x, int n, const ArGroup& a, const HostReducer& hr, int mi, unsigned seq, void* scratch,
                    cudaStream_t s, const ArAux& aux) {
    const int slot = (int) (seq % (unsigned) a.slots), nm = a.nm, n4 = n / 4, NP = ar_host_pieces(), P4 = cdiv(n4, NP);
    const int fs = a.slots * kArBlocks * 8;
    uint2* own = (uint2*) scratch;
    uint2* hrow = (uint2*) hr.row(mi, slot);
    const uint2* hsum = (const uint2*) hr.sum(slot);
    ar_to16_kernel<<<std::max(1, std::min(1024, cdiv(n4, kThreads))), kThreads, 0, s>>>((const float4*) x, n4, own);
    cudaEventRecord(aux.ev[0], s);
    cudaStreamWaitEvent(aux.s2, aux.ev[0], 0);
    for (int p = 0; p < NP; ++p) {
        const int p0 = std::min(n4, p * P4), p1 = std::min(n4, p0 + P4);
        if (p1 > p0) cudaMemcpyAsync(hrow + p0, own + p0, (size_t) (p1 - p0) * 8, cudaMemcpyDeviceToHost, aux.s2);
        set_flag_kernel<<<1, 1, 0, aux.s2>>>(a.bflags + ((size_t) slot * kArBlocks + p) * 8 + (size_t) mi * fs, seq);
    }
    for (int p = 0; p < NP; ++p) {
        const int p0 = std::min(n4, p * P4), p1 = std::min(n4, p0 + P4);
        wait_flags_kernel<<<1, 32, 0, s>>>(a.bflags + (((size_t) nm * a.slots + slot) * kArBlocks + p) * 8, 1, 0, seq);
        if (p1 > p0)
            ar_from16_zc_kernel<<<std::max(1, std::min(1024, cdiv(p1 - p0, kThreads))), kThreads, 0, s>>>(
                (float4*) x, hsum, p1, p1, p1, p0);
    }
}

}  // namespace strata::ds4

