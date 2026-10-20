// tests/ds4_set_gu_mixed.cu - the prompt path's gate / up of a set whose gate and up differ in K (a SAGE pack's
// experts, 5% of them): set_gate_up's FP16 swiglu rows against gemm_rows of each matrix alone (an EXL3 Mat a view of
// the blob) and the clamped swiglu on the host - within FP16's rounding of the largest value (gemm_rows takes FP16
// inputs: rounded once more than the set's); and a set of one K against the same check.
//   nvcc -O3 -std=c++17 -arch=native -I include -I src/kernels/cuda tests/ds4_set_gu_mixed.cu src/kernels/cuda/exl3.cu \
//        src/kernels/cuda/exl3_dense.cu src/kernels/cuda/exl3_legacy.cu -o build/ds4_set_gu_mixed
#include "strata/kernels/exl3.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace x3 = strata::kernels::exl3;

namespace {
size_t a256(size_t b) { return (b + 255) / 256 * 256; }
template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
}  // namespace

int main() {
    const int D = 1024, FF = 512, NE = 2, ROWS = 37;   // two experts, 37 rows each
    std::mt19937 rng(5);
    cudaStream_t s;
    cudaStreamCreate(&s);
    int fails = 0;
    for (std::array<int, 3> K : {std::array<int, 3>{1, 2, 2}, std::array<int, 3>{3, 1, 1}, std::array<int, 3>{2, 4, 3},
                                 std::array<int, 3>{2, 2, 2}}) {
        x3::ExpertLayout lay;
        size_t off = x3::kBlobHdr;   // (as a mixed layer's: the header's room first)
        for (int r = 0; r < 3; ++r) {
            const int k = r < 2 ? D : FF, n = r < 2 ? FF : D;
            lay.K[r] = K[r];
            lay.trellis[r] = off;
            off += a256(x3::trellis_bytes(k, n, K[r]));
            lay.suh[r] = off;
            off += a256((size_t) k * 2);
            lay.svh[r] = off;
            off += a256((size_t) n * 2);
        }
        lay.cb = 2;
        const size_t stride = off;
        std::vector<uint8_t> blobs(stride * NE);
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_real_distribution<float> u(0.5f, 1.5f), sg(0.0f, 1.0f);
        for (int e = 0; e < NE; ++e)
            for (int r = 0; r < 2; ++r) {
                uint8_t* b = blobs.data() + e * stride;
                for (size_t i = 0; i < x3::trellis_bytes(D, FF, K[r]); ++i) b[lay.trellis[r] + i] = (uint8_t) byte(rng);
                for (int i = 0; i < D; ++i) {
                    const __half h = __float2half((sg(rng) < 0.5f ? -1.0f : 1.0f) * u(rng) * 0.02f);
                    std::memcpy(b + lay.suh[r] + (size_t) i * 2, &h, 2);
                }
                for (int i = 0; i < FF; ++i) {
                    const __half h = __float2half((sg(rng) < 0.5f ? -1.0f : 1.0f) * u(rng));
                    std::memcpy(b + lay.svh[r] + (size_t) i * 2, &h, 2);
                }
            }
        uint8_t* db = dev(blobs);
        std::vector<float> x((size_t) NE * ROWS * D);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (auto& v : x) v = g(rng);
        float* dx = dev(x);
        std::vector<int> idx(NE * ROWS), bounds = {0, ROWS, 2 * ROWS};
        for (int i = 0; i < NE * ROWS; ++i) idx[(size_t) i] = i;
        int *didx = dev(idx), *dbd = dev(bounds);
        uint16_t* dh = nullptr;
        cudaMalloc(&dh, (size_t) NE * ROWS * FF * 2);
        x3::SetArgs sa;
        sa.wbase = db;
        sa.stride = stride;
        sa.lay = lay;
        sa.n_exp = NE;
        sa.bounds = dbd;
        sa.row0 = 0;
        sa.max_rows = ROWS;
        sa.n_embd = D;
        sa.n_ff = FF;
        sa.limit = 10.0f;
        x3::set_gate_up(sa, dx, D, didx, dh, FF, s);
        // each matrix alone: its rows' FP16 inputs, gemm_rows over an EXL3 view of the blob
        std::vector<uint16_t> x16((size_t) NE * ROWS * D);
        for (size_t i = 0; i < x.size(); ++i) {
            const __half h = __float2half_rn(x[i]);
            std::memcpy(&x16[i], &h, 2);
        }
        uint16_t* dx16 = dev(x16);
        float *dg = nullptr, *du = nullptr;
        cudaMalloc(&dg, (size_t) NE * ROWS * FF * 4);
        cudaMalloc(&du, (size_t) NE * ROWS * FF * 4);
        for (int e = 0; e < NE; ++e)
            for (int r = 0; r < 2; ++r) {
                x3::Mat m;
                m.trellis = (const uint32_t*) (db + e * stride + lay.trellis[r]);
                m.suh = (const uint16_t*) (db + e * stride + lay.suh[r]);
                m.svh = (const uint16_t*) (db + e * stride + lay.svh[r]);
                m.k = D, m.n = FF, m.K = K[r], m.cb = 2;
                x3::gemm_rows(m, dx16 + (size_t) e * ROWS * D, D, ROWS, (r == 0 ? dg : du) + (size_t) e * ROWS * FF, FF,
                              0.0f, s);
            }
        cudaStreamSynchronize(s);
        std::vector<uint16_t> h((size_t) NE * ROWS * FF);
        std::vector<float> gv(h.size()), uv(h.size());
        cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(gv.data(), dg, gv.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(uv.data(), du, uv.size() * 4, cudaMemcpyDeviceToHost);
        double worst = 0, mag = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            float a = std::fmin(gv[i], 10.0f), b = std::fmin(std::fmax(uv[i], -10.0f), 10.0f);
            const float ref = (a / (1.0f + std::exp(-a))) * b;
            __half hh;
            std::memcpy(&hh, &h[i], 2);
            const float got = __half2float(hh);
            worst = std::max(worst, (double) std::fabs(got - ref));
            mag = std::max(mag, (double) std::fabs(ref));
        }
        worst /= mag;   // (of the largest value: the reference's inputs are rounded to FP16 first, the kernel's not)
        const bool ok = worst < 2e-3;
        std::printf("gate K %d, up K %d: worst %.2e of the swiglu (values up to %.3g) - %s\n", K[0], K[1], worst, mag,
                    ok ? "ok" : "WRONG");
        fails += !ok;
    }
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
