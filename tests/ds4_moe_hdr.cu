// tests/ds4_moe_hdr.cu - the token path's routed experts with each blob's own layout (ExpertLayout::hdr, a SAGE pack's
// experts: each matrix its own K): (1) a plan of one layout, headed blobs -> the same bits as the layer-wide layout's
// kernels; (2) a plan of experts of other K (gate and up apart too) -> each expert's down rows as that expert alone gives
// them, added up in plan order as the kernel does - the same bits.  Random trellis words (any decode a valid weight).
//   nvcc -O3 -std=c++17 -arch=native -I include -I src/kernels/cuda tests/ds4_moe_hdr.cu src/kernels/cuda/exl3.cu \
//        src/kernels/cuda/exl3_dense.cu src/kernels/cuda/exl3_legacy.cu -o build/ds4_moe_hdr
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
// one expert's blob for (K gate, K up, K down), offsets from hdr0 on; its layout in lay
std::vector<uint8_t> blob(int D, int FF, std::array<int, 3> K, bool hdr, x3::ExpertLayout& lay, std::mt19937& rng) {
    size_t off = hdr ? x3::kBlobHdr : 0;
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
    lay.hdr = hdr ? 1 : 0;
    std::vector<uint8_t> b(off, 0);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.5f, 1.5f), sg(0.0f, 1.0f);
    for (int r = 0; r < 3; ++r) {
        const int k = r < 2 ? D : FF, n = r < 2 ? FF : D;
        for (size_t i = 0; i < x3::trellis_bytes(k, n, K[r]); ++i) b[lay.trellis[r] + i] = (uint8_t) byte(rng);
        for (int i = 0; i < k; ++i) {
            const __half h = __float2half((sg(rng) < 0.5f ? -1.0f : 1.0f) * sc(rng) * 0.02f);
            std::memcpy(&b[lay.suh[r] + (size_t) i * 2], &h, 2);
        }
        for (int i = 0; i < n; ++i) {
            const __half h = __float2half((sg(rng) < 0.5f ? -1.0f : 1.0f) * sc(rng));
            std::memcpy(&b[lay.svh[r] + (size_t) i * 2], &h, 2);
        }
    }
    if (hdr) {
        x3::BlobHdr h{};
        for (int r = 0; r < 3; ++r) h.suh[r] = lay.suh[r], h.svh[r] = lay.svh[r], h.tr[r] = lay.trellis[r], h.K[r] = lay.K[r];
        h.cb = 2;
        std::memcpy(b.data(), &h, sizeof h);
    }
    return b;
}
template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
// the token path's output for the plan (blobs on the device) under lay
std::vector<float> run(const std::vector<uint8_t*>& blobs, const std::vector<float>& w, const x3::ExpertLayout& lay,
                       const float* dx, const float* dsh, int D, int FF, cudaStream_t s) {
    std::vector<unsigned long long> pp;
    for (uint8_t* b : blobs) pp.push_back((unsigned long long) b);
    unsigned long long* dpp = dev(pp);
    float *dw = dev(w), *dout = nullptr, *dh = nullptr;
    cudaMalloc(&dout, (size_t) D * 4);
    cudaMalloc(&dh, (size_t) 8 * FF * 4);
    x3::MoeArgs a;
    a.plan_ptr = dpp;
    a.plan_w = dw;
    a.lay = lay;
    a.k = (int) blobs.size();
    a.n_embd = D;
    a.n_ff = FF;
    a.limit = 10.0f;
    x3::moe_gate_up(a, dx, dh, s);
    x3::moe_down(a, dh, dsh, dout, s);
    std::vector<float> out(D);
    cudaStreamSynchronize(s);
    cudaMemcpy(out.data(), dout, (size_t) D * 4, cudaMemcpyDeviceToHost);
    cudaFree(dpp), cudaFree(dw), cudaFree(dout), cudaFree(dh);
    return out;
}
}  // namespace

int main() {
    const int D = 1024, FF = 512;
    std::mt19937 rng(11);
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::vector<float> x(D), sh(D);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
    for (auto& v : sh) v = g(rng);
    float *dx = dev(x), *dsh = dev(sh);
    int fails = 0;
    // (1) one layout: headed blobs against plain ones, the same words
    for (std::array<int, 3> K : {std::array<int, 3>{1, 1, 1}, std::array<int, 3>{2, 2, 3}, std::array<int, 3>{4, 4, 4}}) {
        std::vector<uint8_t*> plain, headed;
        x3::ExpertLayout lp, lh;
        for (int e = 0; e < 6; ++e) {
            std::mt19937 r1(100 + e), r2(100 + e);
            std::vector<uint8_t> bp = blob(D, FF, K, false, lp, r1), bh = blob(D, FF, K, true, lh, r2);
            plain.push_back((uint8_t*) dev(bp));
            headed.push_back((uint8_t*) dev(bh));
        }
        const std::vector<float> w = {0.3f, 0.1f, 0.25f, 0.05f, 0.2f, 0.1f};
        const std::vector<float> a = run(plain, w, lp, dx, dsh, D, FF, s), b = run(headed, w, lh, dx, dsh, D, FF, s);
        const bool same = std::memcmp(a.data(), b.data(), (size_t) D * 4) == 0;
        std::printf("K %d/%d/%d, one layout: headed blobs %s\n", K[0], K[1], K[2], same ? "the same bits" : "DIFFER");
        fails += !same;
    }
    // (2) a plan of mixed layouts (gate and up apart in two of them)
    {
        const std::array<int, 3> Ks[6] = {{1, 1, 1}, {1, 1, 2}, {2, 2, 3}, {2, 3, 2}, {3, 3, 3}, {1, 2, 4}};
        std::vector<uint8_t*> headed;
        std::vector<x3::ExpertLayout> lays(6);
        for (int e = 0; e < 6; ++e) {
            std::mt19937 r(200 + e);
            headed.push_back((uint8_t*) dev(blob(D, FF, Ks[e], true, lays[(size_t) e], r)));
        }
        const std::vector<float> w = {0.3f, 0.1f, 0.25f, 0.05f, 0.2f, 0.1f};
        const std::vector<float> mixed = run(headed, w, lays[0], dx, dsh, D, FF, s);
        // each expert alone (weight 1, no shared expert): its down rows; added up in plan order, then the shared row
        std::vector<std::vector<float>> p(6);
        std::vector<float> zero(D, 0.0f);
        float* dz = dev(zero);
        for (int e = 0; e < 6; ++e) p[(size_t) e] = run({headed[(size_t) e]}, {1.0f}, lays[(size_t) e], dx, dz, D, FF, s);
        bool same = true;
        for (int r = 0; r < D; ++r) {
            float m = 0.0f;
            for (int i = 0; i < 6; ++i) m = std::fmaf(w[(size_t) i], p[(size_t) i][(size_t) r], m);
            m = m + sh[(size_t) r];
            if (std::memcmp(&m, &mixed[(size_t) r], 4) != 0) same = false;
        }
        double mag = 0;
        for (float v : mixed) mag = std::max(mag, (double) std::fabs(v));
        std::printf("a plan of six layouts: %s (outputs up to %.3g)\n", same ? "each expert's own bits, added in plan order" : "DIFFER", mag);
        fails += !same;
    }
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
