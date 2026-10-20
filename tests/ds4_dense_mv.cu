// tests/ds4_dense_mv.cu - exl3::Mat's plain weights (kFmtFP8: DeepSeek's E4M3 with a UE8M0 scale per 32 x 32 block,
// kFmtBF16) through exl3::mv against a double-precision reference on the host: 1 to 16 activations (each one's outputs
// the same bits whatever the others), alpha and a bias, several jobs of both formats and an EXL3-free call in one go,
// a view of the rows, and reconstruct_rows's F16 / BF16 rows.
//   nvcc -O3 -std=c++17 -arch=native -I include -I src/kernels/cuda tests/ds4_dense_mv.cu src/kernels/cuda/exl3.cu \
//        src/kernels/cuda/exl3_dense.cu src/kernels/cuda/exl3_legacy.cu -o build/ds4_dense_mv
#include "strata/kernels/exl3.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace x3 = strata::kernels::exl3;

namespace {
double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return (b & 0x80) ? -v : v;
}
float bf16f(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

struct W {
    int fmt, k, n;
    std::vector<uint8_t> w8, sc;
    std::vector<uint16_t> wb;
    double at(int r, int c) const {
        if (fmt == x3::kFmtFP8) return e4m3(w8[(size_t) r * k + c]) * std::ldexp(1.0, (int) sc[(size_t) (r / 32) * (k / 32) + c / 32] - 127);
        return bf16f(wb[(size_t) r * k + c]);
    }
    x3::Mat mat() const {
        x3::Mat m;
        m.fmt = fmt;
        m.k = k;
        m.n = n;
        if (fmt == x3::kFmtFP8) {
            m.w = dev(w8);
            m.sc = dev(sc);
        } else {
            m.w = dev(wb);
        }
        return m;
    }
};
W make(int fmt, int k, int n, std::mt19937& rng) {
    W w{fmt, k, n, {}, {}, {}};
    std::uniform_int_distribution<int> b(0, 255), e(120, 130);
    if (fmt == x3::kFmtFP8) {
        w.w8.resize((size_t) n * k);
        for (auto& v : w.w8) {
            v = (uint8_t) b(rng);
            if ((v & 0x7f) == 0x7f) v ^= 1;   // (no NaN)
        }
        w.sc.resize((size_t) (n / 32) * (k / 32));
        for (auto& v : w.sc) v = (uint8_t) e(rng);
    } else {
        std::normal_distribution<float> g(0.0f, 0.05f);
        w.wb.resize((size_t) n * k);
        for (auto& v : w.wb) v = f2bf(g(rng));
    }
    return w;
}

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    std::mt19937 rng(7);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][3] = {{x3::kFmtFP8, 5120, 512}, {x3::kFmtFP8, 1280, 32768}, {x3::kFmtBF16, 5120, 512},
                             {x3::kFmtBF16, 512, 128}, {x3::kFmtFP8, 15360, 5120}, {x3::kFmtFP8, 384, 5120},
                             {x3::kFmtBF16, 160, 256}};
    for (const auto& sh : shapes) {
        const W w = make(sh[0], sh[1], sh[2], rng);
        const x3::Mat m = w.mat();
        check(x3::supported(m), "supported");
        const int k = w.k, n = w.n, NC = 16;
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> x((size_t) NC * k), bias(n);
        for (auto& v : x) v = g(rng);
        for (auto& v : bias) v = g(rng);
        float *dx = dev(x), *db = dev(bias), *dy = nullptr, *dy1 = nullptr;
        cudaMalloc(&dy, (size_t) NC * n * 4);
        cudaMalloc(&dy1, (size_t) NC * n * 4);
        x3::MvJob j;
        j.m = &m;
        j.xf = dx;
        j.y = dy;
        j.bias = db;
        j.alpha = 0.5f;
        j.ncol = NC;
        j.xf_cs = k;
        j.y_cs = n;
        check(x3::mv(&j, 1, s), "mv");
        for (int c = 0; c < NC; ++c) {   // each activation alone: the same bits
            x3::MvJob o = j;
            o.xf = dx + (size_t) c * k;
            o.y = dy1 + (size_t) c * n;
            o.ncol = 1;
            check(x3::mv(&o, 1, s), "mv one");
        }
        std::vector<float> y((size_t) NC * n), y1(y.size());
        cudaStreamSynchronize(s);
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost);
        double worst = 0;
        for (int c = 0; c < NC; c += 5)
            for (int r = 0; r < n; r += (n > 4096 ? 7 : 1)) {
                double ref = 0, mag = 0;
                for (int i = 0; i < k; ++i) {
                    const double p = w.at(r, i) * x[(size_t) c * k + i];
                    ref += p;
                    mag += std::fabs(p);
                }
                ref = 0.5 * ref + bias[r];
                worst = std::max(worst, std::fabs(ref - y[(size_t) c * n + r]) / (mag * 0.5 + 1e-30));
            }
        check(worst < 1e-5, "values");
        check(std::memcmp(y.data(), y1.data(), y.size() * 4) == 0, "one activation at a time: the same bits");
        // a view of rows [v0, v0 + 128) and reconstruct_rows
        const int v0 = n >= 256 ? 64 : 0;
        const x3::Mat v = x3::view(m, v0, 128);
        std::vector<uint16_t> r16((size_t) 128 * k), rb((size_t) 128 * k);
        uint16_t *d16 = nullptr, *dbf = nullptr;
        cudaMalloc(&d16, r16.size() * 2);
        cudaMalloc(&dbf, rb.size() * 2);
        x3::reconstruct_rows(v, d16, false, s);
        x3::reconstruct_rows(v, dbf, true, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(r16.data(), d16, r16.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(rb.data(), dbf, rb.size() * 2, cudaMemcpyDeviceToHost);
        double wr = 0;
        bool bf_exact = true;
        for (int r = 0; r < 128; ++r)
            for (int i = 0; i < k; ++i) {
                const double ref = w.at(v0 + r, i);
                const float h = __half2float(__ushort_as_half(r16[(size_t) r * k + i]));
                wr = std::max(wr, std::fabs(ref - h) / (std::fabs(ref) + 1e-3));
                if (w.fmt == x3::kFmtBF16 && rb[(size_t) r * k + i] != w.wb[(size_t) (v0 + r) * k + i]) bf_exact = false;
            }
        check(wr < 1e-3, "reconstruct_rows F16");
        check(bf_exact, "reconstruct_rows BF16 of BF16: exact");
        std::printf("fmt %s k %5d n %5d: worst %.2e of the products' magnitude; rows %.1e; %s\n",
                    w.fmt == x3::kFmtFP8 ? "FP8 " : "BF16", k, n, worst, wr, fails ? "FAILING" : "ok");
    }
    // several jobs of both formats in one call
    {
        const W a = make(x3::kFmtFP8, 5120, 1280, rng), b = make(x3::kFmtBF16, 5120, 512, rng);
        const x3::Mat ma = a.mat(), mb = b.mat();
        std::vector<float> x(5 * 5120);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (auto& v : x) v = g(rng);
        float *dx = dev(x), *ya, *yb, *ya1, *yb1;
        cudaMalloc(&ya, 5 * 1280 * 4), cudaMalloc(&yb, 5 * 512 * 4), cudaMalloc(&ya1, 5 * 1280 * 4), cudaMalloc(&yb1, 5 * 512 * 4);
        x3::MvJob j[2];
        j[0].m = &ma, j[0].xf = dx, j[0].y = ya, j[0].ncol = 5, j[0].xf_cs = 5120, j[0].y_cs = 1280;
        j[1].m = &mb, j[1].xf = dx, j[1].y = yb, j[1].ncol = 5, j[1].xf_cs = 5120, j[1].y_cs = 512;
        check(x3::mv(j, 2, s), "mv two");
        x3::MvJob o0 = j[0], o1 = j[1];
        o0.y = ya1, o1.y = yb1;
        check(x3::mv(&o0, 1, s) && x3::mv(&o1, 1, s), "mv apart");
        cudaStreamSynchronize(s);
        std::vector<float> h0(5 * 1280), h1(5 * 1280), g0(5 * 512), g1(5 * 512);
        cudaMemcpy(h0.data(), ya, h0.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h1.data(), ya1, h1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(g0.data(), yb, g0.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(g1.data(), yb1, g1.size() * 4, cudaMemcpyDeviceToHost);
        check(h0 == h1 && g0 == g1, "jobs together: each its own bits");
    }
    // speed: a 32768 x 1280 FP8 (wq_b) and a 5120 x 8192 (wo_b) at 1 and 6 activations
    for (const auto& sh : {std::array<int, 2>{1280, 32768}, std::array<int, 2>{8192, 5120}}) {
        const W w = make(x3::kFmtFP8, sh[0], sh[1], rng);
        const x3::Mat m = w.mat();
        float *dx, *dy;
        cudaMalloc(&dx, (size_t) 8 * w.k * 4), cudaMalloc(&dy, (size_t) 8 * w.n * 4);
        cudaMemset(dx, 0, (size_t) 8 * w.k * 4);
        for (int nc : {1, 6}) {
            x3::MvJob j;
            j.m = &m, j.xf = dx, j.y = dy, j.ncol = nc, j.xf_cs = w.k, j.y_cs = w.n;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0), cudaEventCreate(&e1);
            for (int i = 0; i < 3; ++i) x3::mv(&j, 1, s);
            cudaEventRecord(e0, s);
            for (int i = 0; i < 50; ++i) x3::mv(&j, 1, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = ms * 1000 / 50;
            std::printf("FP8 k %5d n %5d, %d activations: %.1f us (%.0f GB/s of weights)\n", w.k, w.n, nc, us,
                        (double) w.k * w.n / (us * 1e3));
        }
    }
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
