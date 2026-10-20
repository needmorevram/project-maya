// src/ds4/ds4_model.cu - DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp).
//
// The layer, for one token at position p (DeepSeek's inference/model.py, Block.forward with the lagged pre-mix):
//   [Engram, layers 1 and 14]  h += gate(h, key) * value from 24 n-gram rows of the FP8 tables
//   a_pre, a_post, a_comb = hc_mixes(h);  x = rmsnorm(sum_s pre[s] h[s])   (pre: the previous layer's f_pre)
//   attention(x): q from the low-rank wq_a / wq_b, one shared K = V row per token in a 128-slot window ring, plus
//                 (compress ratio > 0) up to 512 compressed rows the indexer picks; a learned sink per head; the
//                 output's rotary part rotated back, rounded to FP8, through the grouped wo_a and wo_b
//   h = post * attn + comb^T h;  f_* = hc_mixes(h);  x = rmsnorm(sum_s a_pre[s] h[s])
//   moe(x): 6 of 384 routed experts (sqrt-softplus scores, biased top-k) + the shared expert
//   h = post * moe + comb^T h;  pre = f_pre
// and after the last layer logits = head(rmsnorm(sum_s pre[s] h[s])).
#include "ds4_model.hpp"

#include "ds4_kernels.cuh"
#include "ds4_pool.hpp"
#include "ds4_uring.hpp"
#include "ds4_vision.cuh"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/sampler.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <queue>
#include <array>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace strata::ds4 {
namespace x3 = strata::kernels::exl3;
namespace kc = strata::kernels::cpu;
namespace gf = strata::kernels::glmf;

#define CK(call)                                                                            \
    do {                                                                                    \
        cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                       \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// part p of np of `whole` (a multiple of 128) in 128-blocks, as even as they go (whole / 128 % np parts a block more:
// those from part rot on, cyclically - so a group's layers, each its own rot, even out what each member holds):
// tensor parallel's columns of an FFN or of the head - an even split is what whole / np gave before
void split128(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<int> cpu_list(const std::string& s) {   // "0-3,8,10-11"
    std::vector<int> v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty() || part == "\n") continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
// each NUMA node's physical cores (the first logical CPU of each core), in order
std::vector<std::vector<int>> node_cores() {
    std::vector<std::vector<int>> nodes;
    for (int n = 0; n < 64; ++n) {
        const std::string cl = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cl.empty()) break;
        std::vector<int> cores;
        for (int c : cpu_list(cl)) {
            const std::vector<int> sib = cpu_list(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            if (sib.empty() || sib[0] == c) cores.push_back(c);
        }
        if (!cores.empty()) nodes.push_back(cores);
    }
    if (nodes.empty()) {
        std::vector<int> all;
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency() / 2); ++c) all.push_back((int) c);
        nodes.push_back(all);
    }
    return nodes;
}
// anonymous memory on one NUMA node (node < 0: wherever the first touch puts it), transparent huge pages
void* node_map(size_t bytes, int node, int n_nodes) {
    void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    madvise(m, bytes, MADV_HUGEPAGE);
    if (node >= 0 && n_nodes > 1) {
        unsigned long mask[4] = {0, 0, 0, 0};
        mask[node / 64] = 1ul << (node % 64);
        syscall(SYS_mbind, m, bytes, 2 /* MPOL_BIND */, mask, 256, 0);
    }
    return m;
}
void pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
}
// the RAM this process can still get: MemAvailable, and below it the tightest cgroup v2 limit on the way to the root
// (its limit less what is charged, the page cache counted as free: the kernel takes it back); ~0: not known
uint64_t ram_avail() {
    uint64_t avail = ~uint64_t{0};
    {
        std::istringstream mi(read_file("/proc/meminfo"));
        std::string k;
        uint64_t kb = 0;
        while (mi >> k >> kb) {
            if (k == "MemAvailable:") avail = kb << 10;
            mi.ignore(64, '\n');
        }
    }
    std::string cg = read_file("/proc/self/cgroup");   // "0::/user.slice/..."
    const size_t at = cg.find("0::");
    if (at == std::string::npos) return avail;
    cg = cg.substr(at + 3);
    cg = cg.substr(0, cg.find('\n'));
    for (;;) {
        const std::string dir = "/sys/fs/cgroup" + (cg == "/" ? std::string() : cg);
        const std::string mx = read_file(dir + "/memory.max");
        if (!mx.empty() && mx.compare(0, 3, "max") != 0) {
            const uint64_t lim = std::strtoull(mx.c_str(), nullptr, 10);
            const uint64_t cur = std::strtoull(read_file(dir + "/memory.current").c_str(), nullptr, 10);
            uint64_t file = 0;
            std::istringstream st(read_file(dir + "/memory.stat"));
            std::string k;
            uint64_t v = 0;
            while (st >> k >> v)
                if (k == "file") file = v;
            const uint64_t used = cur > file ? cur - file : 0;
            avail = std::min(avail, lim > used ? lim - used : 0);
        }
        if (cg.empty() || cg == "/") break;
        const size_t sl = cg.rfind('/');
        cg = sl == 0 || sl == std::string::npos ? "/" : cg.substr(0, sl);
    }
    return avail;
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            u = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f32_to_f16(float f) {   // round to nearest even (a BF16 value: exact where F16 holds it)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u, E = (u >> 23) & 0xffu;
    uint32_t m = u & 0x7fffffu;
    const int e = (int) E - 127 + 15;
    if (E == 0xffu) return (uint16_t) (s | 0x7c00u | (m != 0 ? 0x200u : 0u));
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    if (e <= 0) {   // (an F16 subnormal)
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = (uint32_t) e << 10 | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;   // (a carry into the exponent: the next binade, or inf)
    return (uint16_t) (s | r);
}

// ------------------------------------------------------------------ the pack
struct Tensor {
    std::string dtype;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Pack {
    std::string dir, model_dir;
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> files;
    std::vector<int> fds;
    std::unordered_map<std::string, Tensor> t;
    int eg_vocab = 0, eg_layers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<int64_t> eg_mult, eg_primes, eg_off;

    std::vector<std::pair<uint8_t*, size_t>> maps;   // file -> its read-only mapping (map(): the experts in place)

    ~Pack() {
        for (auto& m : maps)
            if (m.first != nullptr) munmap(m.first, m.second);
        for (int fd : fds)
            if (fd >= 0) ::close(fd);
    }
    // file i mapped whole, read-only (once; not thread-safe: called while loading)
    const uint8_t* map(int file) {
        if (maps.size() < fds.size()) maps.resize(fds.size(), {nullptr, 0});
        auto& m = maps[(size_t) file];
        if (m.first == nullptr) {
            struct stat sb;
            if (::fstat(fds[(size_t) file], &sb) != 0 || sb.st_size <= 0) return nullptr;
            void* p = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_SHARED, fds[(size_t) file], 0);
            if (p == MAP_FAILED) return nullptr;
            m = {(uint8_t*) p, (size_t) sb.st_size};
        }
        return m.first;
    }

    bool load(const std::string& d, std::string& err) {
        dir = d;
        {
            std::ifstream f(d + "/ds4.txt");
            if (!f) {
                err = d + "/ds4.txt: not found (run tools/ds4/pack.py --model <model>)";
                return false;
            }
            std::string line;
            while (std::getline(f, line)) {
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) continue;
                kv[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        model_dir = str("model_dir");
        std::FILE* f = std::fopen((d + "/tensors.txt").c_str(), "r");
        if (f == nullptr) {
            err = d + "/tensors.txt: not found";
            return false;
        }
        std::vector<char> buf(1 << 16);
        while (std::fgets(buf.data(), (int) buf.size(), f) != nullptr) {
            char* s = buf.data();
            if (s[0] == 'f') {   // file <i> <name>
                char name[4096];
                int i = 0;
                if (std::sscanf(s, "file %d %4095s", &i, name) == 2) {
                    if ((int) files.size() <= i) files.resize((size_t) i + 1);
                    files[(size_t) i] = name;
                }
                continue;
            }
            if (s[0] != 't') continue;
            char* save = nullptr;
            ::strtok_r(s, " \n", &save);
            const char* name = ::strtok_r(nullptr, " \n", &save);
            const char* dt = ::strtok_r(nullptr, " \n", &save);
            const char* fi = ::strtok_r(nullptr, " \n", &save);
            const char* off = ::strtok_r(nullptr, " \n", &save);
            const char* nb = ::strtok_r(nullptr, " \n", &save);
            if (name == nullptr || nb == nullptr) continue;
            Tensor x;
            x.dtype = dt;
            x.file = std::atoi(fi);
            x.off = std::strtoull(off, nullptr, 10);
            x.bytes = std::strtoull(nb, nullptr, 10);
            while (const char* dim = ::strtok_r(nullptr, " \n", &save)) x.shape.push_back(std::atoll(dim));
            t.emplace(name, std::move(x));
        }
        std::fclose(f);
        for (const std::string& fn : files) {
            const std::string p = model_dir + "/" + fn;
            const int fd = ::open(p.c_str(), O_RDONLY);
            if (fd < 0) {
                err = p + ": cannot open";
                return false;
            }
            fds.push_back(fd);
        }
        // the Engram tables
        std::FILE* e = std::fopen((d + "/engram.bin").c_str(), "rb");
        if (e == nullptr) {
            err = d + "/engram.bin: not found";
            return false;
        }
        char magic[4];
        int32_t hdr[6];
        bool ok = std::fread(magic, 1, 4, e) == 4 && std::memcmp(magic, "DSE1", 4) == 0 &&
                  std::fread(hdr, 4, 6, e) == 6;
        if (ok) {
            eg_vocab = hdr[0], eg_layers = hdr[1], eg_ngram = hdr[2], eg_heads = hdr[3], eg_hd = hdr[4], eg_pad = hdr[5];
            const int cols = (eg_ngram - 1) * eg_heads;
            eg_map.resize((size_t) eg_vocab);
            eg_mult.resize((size_t) eg_layers * eg_ngram);
            eg_primes.resize((size_t) eg_layers * cols);
            eg_off.resize((size_t) eg_layers * cols);
            ok = std::fread(eg_map.data(), 4, eg_map.size(), e) == eg_map.size() &&
                 std::fread(eg_mult.data(), 8, eg_mult.size(), e) == eg_mult.size() &&
                 std::fread(eg_primes.data(), 8, eg_primes.size(), e) == eg_primes.size() &&
                 std::fread(eg_off.data(), 8, eg_off.size(), e) == eg_off.size();
        }
        std::fclose(e);
        if (!ok) {
            err = d + "/engram.bin: unreadable";
            return false;
        }
        return true;
    }

    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v;
        std::stringstream ss(str(k));
        std::string x;
        while (std::getline(ss, x, ','))
            if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    bool read_at(int file, uint64_t off, uint64_t n, void* dst) const {
        uint8_t* p = (uint8_t*) dst;
        while (n > 0) {
            const ssize_t r = ::pread(fds[(size_t) file], p, std::min<uint64_t>(n, 1ull << 30), (off_t) off);
            if (r <= 0) return false;
            p += r, off += (uint64_t) r, n -= (uint64_t) r;
        }
        return true;
    }
    bool read(const Tensor& x, void* dst) const { return read_at(x.file, x.off, x.bytes, dst); }
};

// ------------------------------------------------------------------ geometry
struct Geo {
    int vocab = 0, dim = 0, nl = 0, heads = 0, hd = 0, rd = 0, q_lora = 0, o_lora = 0, o_groups = 0, n_exp = 0,
        k_exp = 0, ff = 0, win = 0, hc = 0, sinkhorn = 0, ix_heads = 0, ix_hd = 0, ix_topk = 0, cand_src = -1,
        cand_topk = 0, cand_blk = 0, orig_ctx = 0, eos = 1;
    float eps = 1e-20f, hc_eps = 1e-6f, route_scale = 1.0f, lim = 0.0f;
    double rope_theta = 1e4, comp_theta = 1e4, factor = 1.0, beta_fast = 32, beta_slow = 1;
    std::vector<int> ratios, kv_src, ix_src, eg_layers;
    // the DSpark drafter (the checkpoint's mtp.* blocks): a block of [seed, noise x (block - 1)] through n_mtp blocks
    int n_mtp = 0, d_block = 0, d_noise = 0, d_rank = 0, d_n_exp = 0, d_k = 0;
    std::vector<int> d_targets;

    void from(const Pack& p) {
        vocab = (int) p.num("vocab_size", 0), dim = (int) p.num("hidden_size", 0);
        nl = (int) p.num("num_hidden_layers", 0), heads = (int) p.num("num_attention_heads", 0);
        hd = (int) p.num("head_dim", 0), rd = (int) p.num("qk_rope_head_dim", 0);
        q_lora = (int) p.num("q_lora_rank", 0), o_lora = (int) p.num("o_lora_rank", 0);
        o_groups = (int) p.num("o_groups", 0), n_exp = (int) p.num("n_routed_experts", 0);
        k_exp = (int) p.num("num_experts_per_tok", 0), ff = (int) p.num("moe_intermediate_size", 0);
        win = (int) p.num("sliding_window", 0), hc = (int) p.num("hc_mult", 0);
        sinkhorn = (int) p.num("hc_sinkhorn_iters", 0), ix_heads = (int) p.num("index_n_heads", 0);
        ix_hd = (int) p.num("index_head_dim", 0), ix_topk = (int) p.num("index_topk", 0);
        cand_src = (int) p.num("candidate_source_layer_id", -1), cand_topk = (int) p.num("candidate_topk_blocks", 0);
        cand_blk = (int) p.num("candidate_block_size", 0);
        orig_ctx = (int) p.num("rope_scaling.original_max_position_embeddings", 0);
        eos = (int) p.num("eos_token_id", 1);
        eps = (float) p.num("rms_norm_eps", 1e-20), hc_eps = (float) p.num("hc_eps", 1e-6);
        route_scale = (float) p.num("routed_scaling_factor", 1.0), lim = (float) p.num("swiglu_limit", 0.0);
        rope_theta = p.num("rope_theta", 1e4), comp_theta = p.num("compress_rope_theta", 1e4);
        factor = p.num("rope_scaling.factor", 1.0), beta_fast = p.num("rope_scaling.beta_fast", 32);
        beta_slow = p.num("rope_scaling.beta_slow", 1);
        ratios = p.list("compress_ratios"), kv_src = p.list("kv_source_layer_ids");
        ix_src = p.list("index_source_layer_ids"), eg_layers = p.list("engram_layer_ids");
        d_block = (int) p.num("dspark_block_size", 0), d_noise = (int) p.num("dspark_noise_token_id", 0);
        d_rank = (int) p.num("dspark_markov_rank", 0), d_n_exp = (int) p.num("dspark_n_routed_experts", 0);
        d_k = (int) p.num("dspark_num_experts_per_tok", 0), d_targets = p.list("dspark_target_layer_ids");
        n_mtp = d_block > 0 && (int) ratios.size() > nl && p.get("mtp.0.attn.wq_a.trellis") != nullptr
                    ? (int) ratios.size() - nl : 0;
    }
};

// DeepSeek's precompute_freqs_cis frequencies (YaRN when original_seq_len > 0), in F32 as torch computes them
std::vector<float> rope_freqs(int dim, int original, double base, double factor, double beta_fast, double beta_slow) {
    std::vector<float> f((size_t) dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float e = (float) (2 * i) / (float) dim;
        f[(size_t) i] = 1.0f / std::pow((float) base, e);
    }
    if (original > 0) {
        auto corr = [&](double rot) {
            return dim * std::log(original / (rot * 2 * M_PI)) / (2 * std::log(base));
        };
        const double low = std::max(std::floor(corr(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corr(beta_slow)), (double) dim - 1);
        const float den = (float) std::max(high - low, 1e-3);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / den;
            ramp = std::min(1.0f, std::max(0.0f, ramp));
            const float smooth = 1.0f - ramp;
            f[(size_t) i] = f[(size_t) i] / (float) factor * (1.0f - smooth) + f[(size_t) i] * smooth;
        }
    }
    return f;
}

// ------------------------------------------------------------------ devices, layers
struct Dev {
    int id = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    // a window's hc_mix + hc_split beside the layer (their results are read only at its hc_post; the pre norm takes the
    // lagged pre): their stream, the stream's go (st -> st2) and its done (st2 -> st)
    cudaStream_t st2 = nullptr;
    cudaEvent_t hc_go = nullptr, hc_done = nullptr;
    size_t free0 = 0, total = 0, weights = 0, experts = 0, caches = 0, scratch = 0;
    // (the Monitor's map: the prompt chunks' buffers, out of scratch; of the weights, the copies a group's members
    // each hold of the tensors they do not split - what tensor parallel costs beyond one copy, its members' share)
    size_t prompt = 0, tp_w = 0;
    std::vector<void*> allocs;
    // a window's scratch, rows of up to kWin tokens (the routed experts' rows: kWin * kPicks)
    float *h = nullptr, *h2 = nullptr, *x = nullptr, *qr = nullptr, *q = nullptr, *kvr = nullptr, *ckr = nullptr,
          *cgr = nullptr, *latent = nullptr, *ikr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr,
          *bm = nullptr, *o = nullptr, *oa = nullptr, *ao = nullptr, *mix = nullptr, *a_pre = nullptr,
          *a_post = nullptr, *a_comb = nullptr, *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr,
          *glog = nullptr, *rw = nullptr, *moe_out = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *sh_out = nullptr, *cpu_part = nullptr, *eg_x = nullptr, *eg_kv = nullptr, *logits = nullptr,
          *freqs_win = nullptr, *freqs_comp = nullptr, *xe = nullptr, *xg = nullptr, *xu = nullptr, *xhh = nullptr,
          *ed = nullptr;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *tok = nullptr;
    TopkScratch tks;                                   // a window's indexer top k over long rows (topk's segments)
    unsigned long long* plan_ptr = nullptr;           // the decode path's VRAM experts (0: on the CPU lane)
    uint8_t* eg_rows = nullptr;
    uint16_t *emb_rows = nullptr, *ext = nullptr;   // the window's embedding rows; the window attention's rows
    // the hand-off to the next device through pinned host memory (these cards copy peer to peer only through the
    // driver's staging, ~1.7 ms a copy): h | pre | topk | cand
    uint8_t* bounce = nullptr;
    cudaEvent_t ev_out = nullptr;
    float* tap = nullptr;   // a window's stream means before a DSpark target layer
    // threads / tensor parallelism: its group and its index in it, the group's all-reduces it took part in, xfer_put's
    // block counter, the hand-off to the next group (host-mapped: h | pre | topk | cand, then its flag), the window
    // MoE's grouping scratch
    int grp = 0, mi = 0;
    unsigned ar_seq = 0;
    unsigned* xcnt = nullptr;
    // the head group's members (the last layer's group): each its columns (split128 of V) of the head and the norms;
    // logits_part: a member's share of the logits (the lead's: where the others' land before they join its rows)
    x3::Mat head;
    float *norm_w = nullptr, *d_norm = nullptr, *logits_part = nullptr;
    unsigned head_seq = 0;
    // the drafter's markov bias over the head group (markov_share): this member's rows [mk_v0, mk_v0 + mk_vn) of the
    // markov head, the previous token's E row, the drafted tokens (the lead's: d_ints)
    uint16_t *mk_share = nullptr, *mk_erow = nullptr;
    int* mk_ints = nullptr;
    int mk_v0 = 0, mk_vn = 0;
    uint8_t* hand = nullptr;
    unsigned* hand_flag = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
};

// the chunked prompt path's buffers on a device: a chunk of C tokens through a layer at once (the attention in slabs of
// A queries, the indexer in slabs of AI), its GEMMs through FP16 reconstructions of the weights and cuBLAS
constexpr int kEgSlots = 3;   // a prompt's Engram rows: chunks read ahead (Model::Impl::eg_reader)

struct Chunk {
    int C = 0, A = 0, AI = 0;
    float *h = nullptr, *x = nullptr, *acc = nullptr, *qr = nullptr, *kvr = nullptr, *ckr = nullptr, *cgr = nullptr,
          *latent = nullptr, *ikr = nullptr, *q = nullptr, *o = nullptr, *oa = nullptr, *iq = nullptr, *iw = nullptr,
          *scores = nullptr, *bm = nullptr, *mix = nullptr, *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr,
          *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr, *glog = nullptr, *rw = nullptr,
          *sg = nullptr, *su = nullptr, *y = nullptr, *eg_x = nullptr, *eg_kv = nullptr;
    uint16_t *x16 = nullptr, *qr16 = nullptr, *o16 = nullptr, *oa16 = nullptr, *lat16 = nullptr, *sh16 = nullptr,
             *h16 = nullptr, *eg_x16 = nullptr, *ext = nullptr, *w16 = nullptr, *emb = nullptr;
    size_t w16_elems = 0;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *bounds = nullptr, *row_tok = nullptr, *pos = nullptr;
    uint8_t* eg_rows = nullptr;
    int* bounds_pin = nullptr;   // pinned: the sort's bounds, read back
    // the CPU lane's experts streamed in: ring_n VRAM slots of ring_stride bytes in kParts parts, a batch a part,
    // kParts batches in flight on the copy stream (filled / freed: each part's last copy, its last use)
    static constexpr int kParts = 4;
    uint8_t* ring = nullptr;
    size_t ring_stride = 0;
    int ring_n = 0;
    cudaStream_t cs = nullptr;
    cudaEvent_t filled[kParts] = {}, freed[kParts] = {};
    // in place (file_experts): each streamed expert's file region read into pinned staging (ring_n slots of
    // stage_stride bytes, a part's slots free again once its `filled` copies are done) by the readers, in parallel,
    // and DMA'd from there - a copy from the files' pages goes through the driver's staging, one thread at ~1.3 GB/s
    uint8_t* stage = nullptr;
    size_t stage_stride = 0;
    std::unique_ptr<Pool> readers;
    cublasHandle_t blas = nullptr;
    // host-mapped: a chunk's hand-off to the next group (h | pre | topk | cand), two in flight (slot = chunk % 2), each
    // with its flag, and the next group's members' acknowledgements [slot][member] (32 apart)
    uint8_t* hand[2] = {nullptr, nullptr};
    unsigned* hand_flag[2] = {nullptr, nullptr};
    unsigned* hand_ack = nullptr;
    unsigned arb_seq = 0;        // the group's chunk all-reduces
    ArAux ar_aux;                // (allreduce_dma's pieces: its copies out on a stream of their own)
    // the attention's sums slab by slab beside the next slab's compute: a stream of their own (its own ArAux), and
    // the main stream waits for the last before the residual (DS4_AR_OVERLAP=0: on the main stream, whole)
    cudaStream_t ar_s = nullptr;
    ArAux ar_aux2;
    cudaEvent_t ar_go = nullptr, ar_done = nullptr;
    // the MoE's sum in kPendRanges row ranges on ar_s, the next layer here taking each range's residual (and its
    // first, row-wise steps) as it comes: pend while that is owed, pend_ev a range's sum done
    static constexpr int kPendRanges = 4;
    bool pend = false;
    int pend_T = 0;
    cudaEvent_t pend_ev[kPendRanges] = {};
    // pinned, two chunks in flight: the embedding rows (the first layer's device), the Engram rows (an Engram layer's)
    uint16_t* emb_host[2] = {nullptr, nullptr};
    cudaEvent_t emb_ev[2] = {nullptr, nullptr}, eg_ev[kEgSlots] = {};   // (eg_ev: its copies from each Engram slot)
    double stream_bytes = 0;     // the experts streamed in, in all
};

struct Layer {
    int il = 0, dev = 0, ratio = 0;
    // its group (tensor parallel: nm members, each a copy of the device state in mates, member mi's share of the
    // heads, of wo_a's groups, of wo_b's rows and of the FFNs: gate / up columns, down rows; its part of an expert
    // laid out as its classes' hlay, hblob bytes - the host state, the CPU lane and the full blob layout stay with
    // member 0)
    int grp = 0, nm = 1, mi = 0;
    std::vector<Layer> mates;
    int ff0 = 0, ffn = 0;     // this member's FFN columns [ff0, ff0 + ffn) (split128 of n_ff; the whole one alone)
    // this member's heads [ah0, ah0 + ahn) and wo_a groups [og0, og0 + ogn) - all of them alone; in a group of more
    // members than the heads split among evenly (head_members: 9, 7 or 6 over 64 heads), the first ones none: their
    // attention adds nothing to the sum
    int ah0 = 0, ahn = 0, og0 = 0, ogn = 0;
    // the chunked prompt path: each expert's virtual slot (its VRAM slot, or n_slots + its place among the CPU lane's
    // experts, which a chunk streams in that order) - pinned on member 0, a device copy on each member
    int* vs_pin = nullptr;
    int* vs_dev = nullptr;
    std::vector<int> host_list;
    std::string name;         // "layers.N." or, for the DSpark drafter's blocks, "mtp.N."
    int n_exp = 0, k_exp = 0; // routed experts, and how many a token takes (the drafter's: 128, 3)
    bool mtp = false;         // a DSpark block (index stage = il - n_layers)
    bool kv_src = false, ix_src = false, owns_k = false, cand_src = false, uses_cand = false;
    int src = -1;      // the kv source whose compressed KV this layer reads (index into Impl::srcs)
    int ik_src = -1;   // the kv source whose index keys its indexer reads
    int eg = -1;       // engram layer index, or -1
    x3::Mat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_bias = nullptr, *eg_w = nullptr,
          *gate_bias_vl = nullptr;   // (the image tokens' routing bias, with the vision tower)
    uint16_t *gate_w = nullptr, *i_wproj = nullptr;
    uint16_t* win = nullptr;                    // BF16 [win][hd]
    float *c_kv = nullptr, *c_score = nullptr;  // the compressor's partial group (ratio > 1)
    // routed experts
    int n_vram = 0;
    uint8_t* vram_base = nullptr;
    std::vector<int> slot;                      // expert -> VRAM slot or -1
    std::vector<uint8_t*> host;                 // expert -> host (non-null: on the CPU lane; its halves in hh)
    // the CPU lane keeps an expert as NH column parts, part X on NUMA node X: gate / up columns [X n_ff / NH, ..),
    // down columns [X dim / NH, ..), each a standalone EXL3 matrix (its own trellis, the full suh, its part of svh)
    struct Half {
        size_t trellis[3] = {}, suh[3] = {}, svh[3] = {}, bytes = 0;
        int k[3] = {}, n[3] = {};
    };
    // the experts' layouts by class: one for each (gate, up, down) K the layer's experts come in - an EXL3 pack's
    // layers one, a SAGE pack's several (it gives each matrix its own K; ecls: each expert's class) - the whole blob
    // (lay, blob bytes; a mixed layer's opens with its BlobHdr, lay.hdr), a group member's part of it (hlay, hblob:
    // each member's own), the CPU lane's parts (hl), where its pieces lie in an expert's file region; in a mixed layer
    // the VRAM slots go class by class, [slot0, slot0 + nsl) the class's at vram_base + voff (a member's own)
    struct XCls {
        x3::ExpertLayout lay, hlay;
        size_t blob = 0, hblob = 0;
        Half hl;
        size_t piece_src[3][3] = {}, piece_bytes[3][3] = {};
        int slot0 = 0, nsl = 0;
        size_t voff = 0;
    };
    std::vector<XCls> cls;
    std::vector<uint8_t> ecls;                   // expert -> its class (empty: all class 0)
    std::vector<std::array<uint8_t*, 4>> hh;     // expert -> its parts on the nodes
    // the expert cache: one VRAM slot and one set of host parts kept free, so an expert can move in while the one it
    // replaces moves out; heat is how much each expert was used lately (the profile's share to start with)
    int n_slots = 0, spare_vram = -1;
    std::array<uint8_t*, 4> spare_host = {nullptr, nullptr, nullptr, nullptr};
    std::vector<float> heat;
    std::vector<int> cpu_hits;                  // CPU-lane experts used since the cache last looked
    bool swap_pending = false;
    int sw_in = -1;                             // the expert a pending swap brings in (the recent slots leave it be)
    unsigned long long* tab_pin = nullptr;      // pinned: the table entries a swap writes
    // the recent slots (DS4_RECENT): rc_n VRAM slots from rc_base, each a copy of a CPU-lane expert the layer used
    // lately - its RAM parts stay where they are, so one goes out by its table entry alone; least recently used out
    int rc_base = -1, rc_n = 0, rc_off = 0;     // (rc_off: the layer's slots in Impl::rc_done)
    std::vector<int> rc_exp, rc_fill;           // a slot's expert (-1: none), the expert its copy is bringing (-1: none)
    std::vector<unsigned> rc_gen;               // a filling slot's fill (the gen rc_done says when done)
    std::deque<std::vector<int>> rc_hist;       // the CPU-lane experts of the layer's last ticks (a fill's rule)
    std::vector<unsigned> rc_stamp;             // the tick that last used it
    std::vector<cudaEvent_t> rc_ev;             // [slot][member]: the fill's copies done
    unsigned long long* rc_pin = nullptr;       // pinned: [slot][member] the slot's address, then a 0
    // a speculative window's backups (trunk layers): the ring rows it replaced, and a compressor's state before it and
    // the window's raw rows (kv, then score) to replay the accepted ones
    uint16_t* ring_bak = nullptr;
    float *c_snap = nullptr, *c_rows = nullptr;
    unsigned long long* vram_tab = nullptr;     // device [n_exp]
    std::vector<std::array<x3::Mat, 3>> vm;     // expert -> its gate, up, down as EXL3 views of its VRAM blob
    std::vector<uint64_t> region_off;           // expert -> file region (offset, bytes, file)
    std::vector<uint64_t> region_bytes;
    std::vector<int> region_file;
    size_t dense_bytes = 0;
};

struct Src {
    int layer = 0, ratio = 1;
    int64_t cap = 0;
    // per device (nullptr where no layer reads it): the compressed KV and the index keys as the packed FP4 rows they
    // are (Finish::pdst; kv_row() / ik_row() bytes a row: a quarter of BF16's and a little)
    std::vector<uint8_t*> ckv, ik;
    uint8_t *h_ckv = nullptr, *h_ik = nullptr;   // pinned: a window's new rows, on their way to the other devices
    cudaEvent_t ev_ckv = nullptr, ev_ik = nullptr;
    // threads: whether devices outside the source's group read it, and the host-mapped hand-off of a window's new rows
    // (ckv rows, then ik rows at kWin * hd) and its flag
    bool remote = false;
    uint8_t* x_stage = nullptr;
    unsigned* x_flag = nullptr;
    // pipelined chunks: a second staging slot (slot 1; slot 0 is x_stage), its flag, and the readers' acknowledgements
    // [slot][device] (32 apart); readers: the devices outside the source's group that read it
    uint8_t* x_stage1 = nullptr;
    unsigned* x_flag1 = nullptr;
    unsigned* x_ack = nullptr;
    std::vector<int> readers;
    uint8_t* stage(int slot) const { return slot ? x_stage1 : x_stage; }
    unsigned* flag(int slot) const { return slot ? x_flag1 : x_flag; }
};

// a host thread that enqueues one device's work (threads mode)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    bool idle() {
        std::lock_guard<std::mutex> lk(mu);
        return !busy;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};

struct Group {
    std::vector<int> mem;   // device indices, member order (member 0 the lead)
    ArGroup ar, arb;        // a window's all-reduces, a chunk's
    // the head group: the members' logits shares on their way to the lead, two slots (by seq), their flags and the
    // lead's acknowledgements (host-mapped)
    float* lg_stage = nullptr;
    unsigned *lg_flag = nullptr, *lg_ack = nullptr;
    std::shared_ptr<HostReducer> hred;   // a chunk's all-reduces summed on the host (5 members or more)
};

constexpr int kMaxL = 64;   // layers (the trunk's and the drafter's) a window plan covers

// what a window's layers need from the host, decided before they run (threads enqueue the layers concurrently): the
// mailbox seq of each layer (async: the device publishes its routing), the indexer's selection sizes and the candidate
// blocks as each layer sees them, the window's hand-off seq
struct WinPlan {
    bool par = false, async = false;
    unsigned win = 0;
    int k = 0;              // (a run of chunks) the chunk's index: its hand-off slot k % 2
    unsigned vl = 0;        // (a window) bit t: token t is inside a picture's span (routed with the image bias)
    // (a prompt's window, Options::prefill_keep's approximate tail) skip: before the tail - layer n_pre its KV source's
    // part alone, the layers after it nothing; lo: the tail's start, the consumer layers' window floor (0: none)
    bool skip = false;
    int64_t lo = 0;
    unsigned seq[kMaxL] = {};
    int topk[kMaxL][kWin] = {}, cand[kMaxL][kWin] = {};
};

}  // namespace

// ------------------------------------------------------------------ the model
struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    std::vector<Dev> devs;
    std::vector<Layer> L;
    std::vector<Src> srcs;
    std::vector<Group> groups;
    std::vector<std::unique_ptr<Worker>> workers;   // threads mode: one per device
    bool par = false;                 // threads mode
    unsigned win_seq = 0;             // the windows the threads ran (the hand-offs' seq)
    std::atomic<unsigned> win_now{0};   // (the last decode window's win_seq, for the copy threads)
    // the chunked prompt path (threads mode): chunk tokens a pass, runs shorter than chunk_min in windows
    int chunk = 0, chunk_min = 512;
    std::vector<std::unique_ptr<Chunk>> chunks;   // per device
    size_t eg_cap = kWin;             // tokens h_eg holds
    int tap_cap = kWin;               // rows h_taps holds (a chunk seeds the drafter with its last 128)
    // YOCO (DeepInfer's n_prefill_layer): the first layer of compress ratio 1 - it and the layers after it (the
    // consumers) read the KV and index keys its compressor makes from its input, so a long run's rows before the tail
    // need only its compressor there (Options::prefill_keep); -1: no such split
    int n_pre = -1;
    bool win_skip = false;            // the next window's WinPlan::skip / lo (Model::feed's windows of a truncated prompt)
    int64_t win_lo = 0;
    bool logits_all = false;          // the last feed was a chunk with logits for every row (chunk_logits)
    std::vector<float> chunk_logits;
    std::function<void(int64_t)> progress;   // a run of chunks: the tokens done so far (the main thread, while it waits)
    std::vector<cudaEvent_t> chunk_done;     // on the last layer's device: chunk k through every layer
    std::unique_ptr<std::atomic<int>[]> chunk_rec;   // chunk k's event recorded
    WinPlan wplan;
    int head_dev = 0;
    x3::Mat head;
    float* norm_w = nullptr;          // on the head device
    // the drafter: main_proj / main_norm (on its first block's device), its final norm, the markov head and the
    // confidence head (on the head device: the drafts' logits come from the trunk's head)
    x3::Mat d_main_proj;
    float *d_main_norm = nullptr, *d_norm = nullptr, *d_conf = nullptr;
    uint16_t *d_mk_embed = nullptr, *d_mk_head = nullptr;   // BF16 [vocab][rank], F16 [vocab][rank]
    // a head group of two or more: the markov steps over its members (each its rows of the bias and an argmax of its
    // share, the global one through host-mapped slots [step][member] {value, index, seq}) in place of the lead's whole
    // rows - E host-mapped (each member reads the picked token's row), a draft's seq in mk_seq
    bool mk_split = false;
    uint16_t* h_mk_embed = nullptr;
    unsigned* mk_slots = nullptr;
    unsigned* mk_lists = nullptr;     // host-mapped [step][member] kSpecList: a sampled draft's shares' top k (spec_pick)
    unsigned mk_seq = 0;
    int d_dev = 0;                    // the drafter's first block's device
    bool spec_win = false;            // the window being run is a draft's verification (layer() keeps backups)
    float* h_taps = nullptr;          // pinned [kWin][targets][dim]: the last window's taps
    cudaEvent_t ev_seed = nullptr;    // the drafter's seeding read h_taps: the next window's taps wait for it
    float* d_confv = nullptr;         // on the head device
    int* d_ints = nullptr;            // on the head device: [0, 16) the draft chain, [16, 32) greedy picks, [32, 48) the
                                      // drafts verified, [48, 64) their verdicts
    float conf_threshold = std::getenv("DS4_DSPARK_CONF") ? (float) std::atof(std::getenv("DS4_DSPARK_CONF")) : 0.5f;
    // speculative sampling (a sampled request): the trunk's token the draft with probability min(1, p/q) under the
    // request's chain (top_k, top_p, min_p, temperature), else a draw from max(0, p - q) - each kept token distributed
    // as the trunk's own draw.  The drafts the drafter's argmax (q a point at each: kept with p's probability), or with
    // DS4_DRAFT_SAMPLE=1 (a request's draft=sample) drawn from its distribution, their kept lists in d_spec[i] - a
    // draft then standing sum min(p, q) of the time instead of p(argmax q): +1-3.5 points a position on one prompt's
    // rows, but no more tokens a round on 40 chat answers (2.446 against 2.454, 3.5 bits, --tp 9, temperature 1.0)
    const bool draft_sample_env =
        std::getenv("DS4_DRAFT_SAMPLE") != nullptr && std::atoi(std::getenv("DS4_DRAFT_SAMPLE")) != 0;
    bool draft_sample = draft_sample_env;   // (a request's draft=greedy|sample: set_draft_sample)
    strata::kernels::SpecList *d_spec = nullptr, *h_spec = nullptr;   // [kWin]: the head device's, pinned
    bool spec_lists = false;          // the last draft's lists are in d_spec
    // DS4_SPEC_DIAG=1 (sampled drafts): each verified position's chance to stand both ways on the same rows - p(argmax q)
    // (a greedy draft) and sum min(p, q) (a drawn one) - by position, printed every 64 rounds
    bool spec_diag = std::getenv("DS4_SPEC_DIAG") != nullptr;
    double diag_g[kWin] = {}, diag_s[kWin] = {};
    long long diag_n[kWin] = {}, diag_rounds = 0;
    // DS4_DRAFT_TIMES=1: a draft's host time by phase (the blocks enqueued, the head, the markov steps, the wait), every
    // 64 drafts
    bool draft_times = std::getenv("DS4_DRAFT_TIMES") != nullptr;
    // DS4_TOPK_CHECK=1: each window's indexer top k (the segmented one, TopkScratch) again a block a row, the two
    // compared - the calls and the rows that differed, every 256 calls
    bool topk_check = std::getenv("DS4_TOPK_CHECK") != nullptr;
    std::atomic<long long> tkc_calls{0}, tkc_bad{0};
    // DS4_INDEX_CHECK=1: each window's indexer scores (the tensor cores' kernel) again on the CUDA cores, into the score
    // rows past the window's - the -inf the same, and the largest difference of the row's largest, every 256 calls
    bool index_check = std::getenv("DS4_INDEX_CHECK") != nullptr;
    std::atomic<long long> ixc_calls{0}, ixc_bad{0};
    std::atomic<uint32_t> ixc_worst{0};   // (a float's bits)
    void index_check_rows(Dev& d, int il, const float* q, const float* w, const uint8_t* kp, int T, int n, const int* cand,
                          int blk, const Rows& lim, long long ss, cudaStream_t s) {
        if (2 * T > kWin || n <= 0) return;
        float* ref = d.scores + (size_t) T * ss;
        index_score(q, w, nullptr, kp, g.ix_heads, g.ix_hd, T, n, cand, (int) cand_stride(), blk, lim, ref, ss, s, false);
        std::vector<float> a((size_t) T * n), b((size_t) T * n);
        cudaMemcpy2DAsync(a.data(), (size_t) n * 4, d.scores, (size_t) ss * 4, (size_t) n * 4, (size_t) T,
                          cudaMemcpyDeviceToHost, s);
        cudaMemcpy2DAsync(b.data(), (size_t) n * 4, ref, (size_t) ss * 4, (size_t) n * 4, (size_t) T, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        bool bad = false;
        float worst = 0.0f;
        for (int t = 0; t < T; ++t) {
            float mx = 0.0f;
            for (int i = 0; i < n; ++i)
                if (std::isfinite(b[(size_t) t * n + i])) mx = std::max(mx, std::fabs(b[(size_t) t * n + i]));
            for (int i = 0; i < n; ++i) {
                const float x = a[(size_t) t * n + i], y = b[(size_t) t * n + i];
                if (std::isfinite(x) != std::isfinite(y)) bad = true;
                else if (std::isfinite(x) && mx > 0.0f) worst = std::max(worst, std::fabs(x - y) / mx);
            }
        }
        if (bad || worst > 1e-5f) ++ixc_bad;
        uint32_t cur = ixc_worst.load();
        while (worst > __builtin_bit_cast(float, cur) && !ixc_worst.compare_exchange_weak(cur, __builtin_bit_cast(uint32_t, worst))) {}
        const long long nc = ++ixc_calls;
        if (nc % 256 == 0 || bad)
            std::fprintf(stderr, "index check: %lld calls, %lld off (layer %d, n %d), worst %.2e of a row's largest\n", nc,
                         ixc_bad.load(), il, n, __builtin_bit_cast(float, ixc_worst.load()));
    }
    double dt_acc[4] = {};
    long long dt_n = 0;

    uint16_t* embed_pinned = nullptr;
    std::vector<float> freqs_win, freqs_comp;
    std::vector<std::unique_ptr<Pool>> pools;   // one per NUMA node, its workers on that node's cores
    std::vector<std::vector<int>> cores;        // each node's physical cores
    int NH = 1;                                 // the CPU lane's parts of an expert (the NUMA nodes)
    // the CPU lane's experts read in place from the model's files (one part, the files' layout; the page cache keeps
    // what fits) instead of copied into RAM, and how much of them the load reads ahead (the most-routed first)
    bool file_experts = false;
    uint64_t warm_bytes = 0;
    static constexpr uint64_t kRamMargin = 8ull << 30;   // RAM left to the rest (the system, the page cache's churn)
    uint8_t* file_part(const Layer& l, int e) const {
        return pk.maps[(size_t) l.region_file[(size_t) e]].first + l.region_off[(size_t) e];
    }
    // host side of the CPU lane, a window's rows
    float *h_x = nullptr, *h_w = nullptr, *h_part = nullptr;
    int* h_idx = nullptr;
    uint8_t* h_eg = nullptr;          // the window's Engram rows: [engram layer][token] of [cols][hd] then the scales
    uint16_t* h_emb = nullptr;        // the window's embedding rows
    std::vector<float> c_xh, c_xsb, c_acc, c_xhd, c_xsd, c_accd, c_yd;
    // the state
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    struct Check {                    // the state at a position: every window ring, the compressors' partial groups
        int64_t pos = -1;
        std::vector<std::vector<uint16_t>> win;
        std::vector<std::vector<float>> comp;
    };
    Check ck[2];                      // the end of the last prompt, and the position before it
    std::vector<uint32_t> prof;       // [nl][n_exp] routed-expert hits
    Stats st;
    int cur_topk_n[kWin] = {}, cur_cand_n[kWin] = {};   // what the latest indexer / candidate source left, per query
                                                        // (plan_window's: the windows' plans are made in order)
    int n_logits = 0;                 // the last forward's logit rows
    // the decode path (windows of one): the device publishes each layer's routing into reqs, the service thread
    // answers in resps with the CPU lane's part, and the device waits for the answer only before the expert sum
    Req* reqs = nullptr;
    Resp* resps = nullptr;
    unsigned seq_issued = 0;
    std::atomic<int> layer_of[kRing];
    std::thread service;
    std::atomic<bool> quit{false};
    bool sync_decode = std::getenv("DS4_SYNC") != nullptr;
    // a window of one through moe_gate_up / moe_down (DS4_MOE_TOKEN=0: win_experts, whose down rows run
    // before the wait for the CPU lane - the same bits, no faster)
    bool moe_token_path = std::getenv("DS4_MOE_TOKEN") == nullptr || std::atoi(std::getenv("DS4_MOE_TOKEN")) != 0;
    int async_max_t = std::getenv("DS4_ASYNC_T") ? std::atoi(std::getenv("DS4_ASYNC_T")) : 8;   // windows up to this
    // the expert cache: swaps prepared on threads of their own (DS4_CACHE_THREADS, 4: each its staging buffers and
    // copy streams), switched between windows
    struct Swap {
        int layer = 0, in = -1, out = -1;
        unsigned win = 0;   // (win_now at the tick that queued it)
    };
    static constexpr int kMaxSwaps = 8;
    // the swap rule: a CPU-lane expert in when its heat passes the coldest VRAM one's x margin + bonus; heat x decay
    // every 256 tokens; at most `inflight` swaps under way (DS4_CACHE_MARGIN / _BONUS / _DECAY / _INFLIGHT)
    static float env_f(const char* k, float d) { return std::getenv(k) ? (float) std::atof(std::getenv(k)) : d; }
    const float c_margin = env_f("DS4_CACHE_MARGIN", 1.25f), c_bonus = env_f("DS4_CACHE_BONUS", 1.0f),
                c_decay = env_f("DS4_CACHE_DECAY", 0.9f);
    const int c_inflight = std::max(1, (int) env_f("DS4_CACHE_INFLIGHT", (float) kMaxSwaps));
    // the recent slots: DS4_RECENT a trunk layer (0: none; 32 where the prompts stream the experts by DMA - the serve
    // protocol at 3.5 bits, 8 / 16 / 24 / 32 of them: 59.0 / 61.5 / 62.2 / 62.9 tok/s, a routing trace's VRAM-neutral
    // optimum ~32), at most DS4_RECENT_FILLS (2) new ones a layer a tick
    int recent_n = 0;
    const int recent_fills = std::max(1, (int) env_f("DS4_RECENT_FILLS", 2.0f));
    // a fill once the expert was on the CPU lane DS4_RECENT_NEED (2) times in the layer's last DS4_RECENT_WIN (8)
    // ticks: a one-off pick stays out (a routing trace of 12 answers, 16 slots: 8.9% of picks on the CPU lane for
    // 0.29 fills a layer and tick, against 7.7% for 0.95 filling every one)
    const int recent_need = std::max(1, (int) env_f("DS4_RECENT_NEED", 2.0f));
    const int recent_win = std::max(1, (int) env_f("DS4_RECENT_WIN", 8.0f));
    std::vector<cudaStream_t> rc_cs;            // a device's copy stream for the fills
    std::vector<cudaEvent_t> rc_go;             // (the fills after the main stream's work so far)
    // the fills off the windows' way: a thread of their own issues each fill's copies (on rc_cs, after rc_go) and
    // marks it done (rc_done[layer's rc_off + slot] = the fill's gen) once its copies are; a tick's table entries go
    // in one launch a device (tab_set, from rc_upd: pinned, [2 x kRcUpd])
    struct RcJob {
        int layer = 0, slot = 0, expert = 0;
        unsigned gen = 0, win = 0;
    };
    // the expert cache's copies on a device (the swaps', the fills') off its group's turn in the windows: over the
    // GPU's PCIe link beside a window's all-reduces they slowed each ~6x (a decode capture's 1405 all-reduces: ~190 us
    // after the last member's arrival with a copy under way on a member, ~29 without).  A device's turn over: its
    // last layer of a window (idle_ev, idle_win the window).  The head's group is idle from the tick on (the next
    // window starts with the first group), so its copies go at once; another group's once its part of a window after
    // the tick is done.  DS4_COPY_IDLE=0: every copy at once
    std::vector<cudaEvent_t> idle_ev;
    std::unique_ptr<std::atomic<unsigned>[]> idle_win;
    bool copy_open(int di, unsigned w) const {
        if (idle_ev.empty() || devs[(size_t) di].grp == devs[(size_t) head_dev].grp) return true;
        const unsigned iw = idle_win[(size_t) di].load(std::memory_order_acquire);
        return (int) (iw - w) > 0 && (int) (iw - win_now.load(std::memory_order_acquire)) >= 0;
    }
    void copy_after_idle(int di, cudaStream_t cs) {   // (di's device current)
        if (!idle_ev.empty()) cudaStreamWaitEvent(cs, idle_ev[(size_t) di], 0);
    }
    static constexpr int kRcUpd = 8192;
    std::mutex rc_mu;
    std::condition_variable rc_cv;
    std::deque<RcJob> rc_q;
    std::thread rc_th;
    std::unique_ptr<std::atomic<unsigned>[]> rc_done;
    std::vector<unsigned long long*> rc_upd;
    std::atomic<unsigned> rc_clock{1};
    int64_t rc_fills_done = 0, rc_fills_started = 0, rc_ticks = 0;
    double rc_tick_ms = 0;
    std::mutex sw_mu;
    std::condition_variable sw_cv;
    std::deque<Swap> sw_queue;
    std::vector<Swap> sw_ready;
    struct SwapBufs {
        uint8_t *a = nullptr, *b = nullptr, *h = nullptr;
        std::vector<cudaStream_t> st;   // a copy stream a device
    };
    std::vector<SwapBufs> sw_bufs;
    std::vector<std::thread> cache_ths;
    int sw_inflight = 0;
    int64_t decay_tokens = 0;
    bool check_swaps = std::getenv("DS4_CHECK_SWAPS") != nullptr;
    std::vector<float> host_logits;
    std::vector<void*> host_allocs;
    std::FILE* dump_h = std::getenv("DS4_DUMP_H") ? std::fopen(std::getenv("DS4_DUMP_H"), "wb") : nullptr;
    // DS4_DUMP_L: (threads) every layer's residual streams as records [layer, p0, T, T x hc x dim floats], window or
    // chunk path (tests: where the two part)
    std::FILE* dump_l = std::getenv("DS4_DUMP_L") ? std::fopen(std::getenv("DS4_DUMP_L"), "wb") : nullptr;
    std::mutex dump_mu;
    void dump_layer(Dev& d, const float* h, int il, int64_t p0, int T) {
        std::vector<float> v((size_t) T * g.hc * g.dim);
        cudaSetDevice(d.id);
        cudaMemcpyAsync(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        std::lock_guard<std::mutex> lk(dump_mu);
        const int64_t hdr[3] = {il, p0, T};
        std::fwrite(hdr, 8, 3, dump_l);
        std::fwrite(v.data(), 4, v.size(), dump_l);
        std::fflush(dump_l);
    }

    ~Impl() {
        for (auto& w : workers) w->stop();
        for (size_t i = 0; i < chunks.size(); ++i)
            if (chunks[i] && chunks[i]->blas) {
                cudaSetDevice(devs[i].id);
                cublasDestroy(chunks[i]->blas);
            }
        quit.store(true);
        if (service.joinable()) service.join();
        {
            std::lock_guard<std::mutex> lk(eg_th_mu);
        }
        eg_th_cv.notify_all();
        eg_done_cv.notify_all();
        if (eg_th.joinable()) eg_th.join();
        for (auto& f : eg_fd) {
            if (f.second.first >= 0) ::close(f.second.first);
            if (f.second.second >= 0) ::close(f.second.second);
        }
        std::free(eg_bounce);
        {
            std::lock_guard<std::mutex> lk(sw_mu);
        }
        sw_cv.notify_all();
        for (std::thread& t : cache_ths)
            if (t.joinable()) t.join();
        {
            std::lock_guard<std::mutex> lk(rc_mu);
        }
        rc_cv.notify_all();
        if (rc_th.joinable()) rc_th.join();
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            for (void* p : d.allocs) cudaFree(p);
            if (d.st) cudaStreamDestroy(d.st);
            if (d.ev) cudaEventDestroy(d.ev);
        }
        for (void* p : host_allocs) cudaFreeHost(p);
        for (Layer& l : L)
            for (uint8_t* b : l.host)
                if (b != nullptr) {
                    (void) b;   // host blobs live in one mapping per layer, unmapped below
                }
        for (auto& m : host_maps) {
            if (chunk > 0) cudaHostUnregister(m.first);
            munmap(m.first, m.second);
        }
    }
    std::vector<std::pair<void*, size_t>> host_maps;

    template <class T>
    bool dalloc(Dev& d, T*& p, size_t n, std::string& err, size_t* acct = nullptr) {
        void* v = nullptr;
        cudaSetDevice(d.id);
        const size_t b = align256(n * sizeof(T));
        if (cudaMalloc(&v, b) != cudaSuccess) {
            err = "cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        // zeroed before it is handed out: d.st does not order with the default stream, and the weights' uploads into
        // these buffers are cudaMemcpy - a zeroing still queued could land on them afterwards (a load's weights, some of
        // them zeros, other ones each start)
        cudaMemsetAsync(v, 0, b, d.st);
        cudaStreamSynchronize(d.st);
        d.allocs.push_back(v);
        if (acct) *acct += b;
        else d.scratch += b;
        p = (T*) v;
        return true;
    }
    static Layer& member(Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    static const Layer& member(const Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    // bytes of a VRAM slot of l's experts (a member's part of each, in a group)
    // expert e's class, and the bytes of its blob on a member's device (its part in a group)
    static const Layer::XCls& xc(const Layer& l, int e) { return l.cls[l.ecls.empty() ? 0 : (size_t) l.ecls[(size_t) e]]; }
    static Layer::XCls& xc(Layer& l, int e) { return l.cls[l.ecls.empty() ? 0 : (size_t) l.ecls[(size_t) e]]; }
    static bool mixed(const Layer& l) { return l.cls.size() > 1; }
    static size_t cunit(const Layer& x, const Layer::XCls& c) { return x.nm > 1 ? c.hblob : c.blob; }
    static size_t unit(const Layer& x, int e) { return cunit(x, xc(x, e)); }
    static size_t unit_max(const Layer& x) {
        size_t b = 0;
        for (const Layer::XCls& c : x.cls) b = std::max(b, cunit(x, c));
        return b;
    }
    // every expert's whole blob, and the VRAM slot s's blob on member x's device
    static size_t all_blobs(const Layer& l) {
        size_t b = 0;
        for (int e = 0; e < l.n_exp; ++e) b += xc(l, e).blob;
        return b;
    }
    static size_t slots_bytes(const Layer& x) {   // (a member's VRAM slots, every class's)
        const Layer::XCls& c = x.cls.back();
        return c.voff + (size_t) c.nsl * cunit(x, c);
    }
    static uint8_t* slot_ptr(const Layer& x, int s) {
        for (const Layer::XCls& c : x.cls)
            if (s >= c.slot0 && s < c.slot0 + c.nsl) return x.vram_base + c.voff + (size_t) (s - c.slot0) * cunit(x, c);
        return x.vram_base + (size_t) s * cunit(x, x.cls[0]);   // (one class: its slots run on)
    }
    // the members of a group of nm that take heads: the most the heads and the wo_a groups split among evenly (64
    // heads and 8 groups: 9 -> 8, 7 or 6 -> 4, 5 -> 4, 3 -> 2); the first members the rest, only their FFN share
    int head_members(int nm) const {
        int na = nm;
        while (na > 1 && (g.heads % na != 0 || g.o_groups % na != 0)) --na;
        return na;
    }
    // a packed row's bytes: the compressed KV's (FP4, an E4M3 scale a 16) and the index keys' (FP4, a power of two a 32)
    size_t kv_row() const { return fp4_row_bytes(g.hd, 2); }
    size_t ik_row() const { return fp4_row_bytes(g.ix_hd, 3); }
    // host-mapped memory (the same address on the host and every device), zeroed
    template <class T>
    bool hmap(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "host-mapped memory (" + std::to_string((n * sizeof(T)) >> 20) + " MiB)";
            return false;
        }
        std::memset(v, 0, n * sizeof(T) + 64);
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocPortable) != cudaSuccess) {
            err = "cudaHostAlloc " + std::to_string((n * sizeof(T)) >> 20) + " MiB failed";
            return false;
        }
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }

    // ---- weights
    bool upload(Dev& d, const Tensor& t, void* dst, std::string& err) {
        std::vector<uint8_t> tmp(t.bytes);
        if (!pk.read(t, tmp.data())) {
            err = "read failed";
            return false;
        }
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dst, tmp.data(), t.bytes, cudaMemcpyHostToDevice));
        return true;
    }
    // an EXL3 matrix onto the device: whole, or (np > 1) part `part` of its columns (cols) or of its rows - a standalone
    // matrix of its own (the rotations' 128-blocks stay whole)
    bool mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
             bool cols = true, int rot = 0) {
        if (pk.get(pre + ".trellis") == nullptr && pk.get(pre + ".weight") != nullptr)
            return dmat(d, pre, m, err, acct, part, np, cols, rot);
        const Tensor* tr = pk.get(pre + ".trellis");
        const Tensor* su = pk.get(pre + ".suh");
        const Tensor* sv = pk.get(pre + ".svh");
        if (tr == nullptr || su == nullptr || sv == nullptr || tr->shape.size() != 3) {
            err = pre + ": not an EXL3 matrix in the pack";
            return false;
        }
        if (pk.get(pre + ".mul1") == nullptr) {
            err = pre + ": only the mul1 codebook is covered";
            return false;
        }
        m.k = (int) tr->shape[0] * 16;
        m.n = (int) tr->shape[1] * 16;
        m.K = (int) tr->shape[2] / 16;
        m.cb = 2;
        m.ld = 0;
        if (np > 1) {
            const int whole = cols ? m.n : m.k;
            int at = 0, w = 0;
            if (whole % 128 != 0 || whole / 128 < np) {
                err = pre + ": " + std::to_string(whole) + (cols ? " columns" : " rows") + " do not split in " +
                      std::to_string(np) + " parts of 128-blocks";
                return false;
            }
            split128(whole, np, part, at, w, rot);
            std::vector<uint8_t> t0(tr->bytes), u0(su->bytes), v0(sv->bytes);
            if (!pk.read(*tr, t0.data()) || !pk.read(*su, u0.data()) || !pk.read(*sv, v0.data())) {
                err = pre + ": read failed";
                return false;
            }
            const size_t tile = tr->bytes / ((size_t) (m.k / 16) * (m.n / 16));
            std::vector<uint8_t> t1, u1, v1;
            if (cols) {   // the tiles [c0 / 16, (c0 + w) / 16) of each 16-row slice, svh [c0, c0 + w)
                const size_t c0 = (size_t) at, nt = (size_t) m.n / 16, nth = (size_t) w / 16;
                t1.resize((size_t) (m.k / 16) * nth * tile);
                for (size_t sl = 0; sl < (size_t) m.k / 16; ++sl)
                    std::memcpy(t1.data() + sl * nth * tile, t0.data() + (sl * nt + c0 / 16) * tile, nth * tile);
                u1 = u0;
                v1.assign(v0.begin() + (long) (c0 * 2), v0.begin() + (long) ((c0 + w) * 2));
                m.n = w;
            } else {      // the 16-row slices [r0 / 16, (r0 + w) / 16), suh [r0, r0 + w)
                const size_t r0 = (size_t) at, row = (size_t) (m.n / 16) * tile;
                t1.assign(t0.begin() + (long) (r0 / 16 * row), t0.begin() + (long) ((r0 + w) / 16 * row));
                u1.assign(u0.begin() + (long) (r0 * 2), u0.begin() + (long) ((r0 + w) * 2));
                v1 = v0;
                m.k = w;
            }
            uint32_t* dt = nullptr;
            uint16_t *ds = nullptr, *dv = nullptr;
            if (!dalloc(d, dt, t1.size() / 4, err, acct) || !dalloc(d, ds, u1.size() / 2, err, acct) ||
                !dalloc(d, dv, v1.size() / 2, err, acct))
                return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(dt, t1.data(), t1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ds, u1.data(), u1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv, v1.data(), v1.size(), cudaMemcpyHostToDevice));
            m.trellis = dt;
            m.suh = ds;
            m.svh = dv;
            return x3::supported(m) || (err = pre + ": a part not covered by the EXL3 kernels", false);
        }
        uint32_t* dt = nullptr;
        uint16_t *ds = nullptr, *dv = nullptr;
        if (!dalloc(d, dt, tr->bytes / 4, err, acct) || !dalloc(d, ds, su->bytes / 2, err, acct) ||
            !dalloc(d, dv, sv->bytes / 2, err, acct))
            return false;
        if (!upload(d, *tr, dt, err) || !upload(d, *su, ds, err) || !upload(d, *sv, dv, err)) return false;
        m.trellis = dt;
        m.suh = ds;
        m.svh = dv;
        if (!x3::supported(m)) {
            err = pre + ": k " + std::to_string(m.k) + " n " + std::to_string(m.n) + " K " + std::to_string(m.K) +
                  " not covered by the EXL3 kernels";
            return false;
        }
        return true;
    }
    // a plain weight onto the device (x3::Mat's kFmtFP8 - DeepSeek's FP8 E4M3 [n][k] and its UE8M0 scale per 32 x 32
    // block, `.scale` - or kFmtBF16): whole, or part `part` of its outputs (cols: rows of the [n][k]) or of its inputs,
    // of the rows [r0, r0 + nr) when nr > 0 (a group of wo_a's) - a standalone matrix of its own
    bool dmat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
              bool cols = true, int rot = 0, int r0 = 0, int nr = 0) {
        const Tensor* t = pk.get(pre + ".weight");
        const Tensor* ts = pk.get(pre + ".scale");
        const bool fp8 = t != nullptr && t->dtype == "F8_E4M3";
        if (t == nullptr || t->shape.size() != 2 || !(fp8 || t->dtype == "BF16") ||
            (fp8 && (ts == nullptr || ts->dtype != "F8_E8M0" || ts->shape.size() != 2 ||
                     ts->shape[0] * 32 != t->shape[0] || ts->shape[1] * 32 != t->shape[1]))) {
            err = pre + ": not an FP8 E4M3 (with UE8M0 32 x 32 scales) or BF16 matrix in the pack";
            return false;
        }
        const size_t eb = fp8 ? 1 : 2, K = (size_t) t->shape[1];
        int n = nr > 0 ? nr : (int) t->shape[0];
        const int R0 = nr > 0 ? r0 : 0;
        if (nr > 0 && (r0 % 32 != 0 || nr % 32 != 0 || r0 + nr > t->shape[0])) {
            err = pre + ": rows " + std::to_string(r0) + " + " + std::to_string(nr) + " out of its blocks";
            return false;
        }
        int at = 0, w = cols ? n : (int) K;
        if (np > 1) {
            const int whole = cols ? n : (int) K;
            if (whole % 128 != 0 || whole / 128 < np) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " +
                      std::to_string(np) + " parts of 128-blocks";
                return false;
            }
            split128(whole, np, part, at, w, rot);
        }
        const int rows = cols ? w : n, row0 = R0 + (cols ? at : 0), k = cols ? (int) K : w, c0 = cols ? 0 : at;
        std::vector<uint8_t> raw((size_t) rows * K * eb), sraw(fp8 ? (size_t) (rows / 32) * (K / 32) : 0);
        if (!pk.read_at(t->file, t->off + (uint64_t) row0 * K * eb, raw.size(), raw.data()) ||
            (fp8 && !pk.read_at(ts->file, ts->off + (uint64_t) (row0 / 32) * (K / 32), sraw.size(), sraw.data()))) {
            err = pre + ": read failed";
            return false;
        }
        if (k != (int) K) {   // (the inputs [c0, c0 + k) of each row, and of each scale row)
            std::vector<uint8_t> a((size_t) rows * k * eb), b(fp8 ? (size_t) (rows / 32) * (k / 32) : 0);
            for (int r = 0; r < rows; ++r)
                std::memcpy(a.data() + (size_t) r * k * eb, raw.data() + ((size_t) r * K + c0) * eb, (size_t) k * eb);
            for (int r = 0; fp8 && r < rows / 32; ++r)
                std::memcpy(b.data() + (size_t) r * (k / 32), sraw.data() + (size_t) r * (K / 32) + c0 / 32, (size_t) k / 32);
            raw.swap(a);
            sraw.swap(b);
        }
        m = x3::Mat{};
        m.fmt = fp8 ? x3::kFmtFP8 : x3::kFmtBF16;
        m.k = k;
        m.n = rows;
        uint8_t *dw = nullptr, *dsc = nullptr;
        if (!dalloc(d, dw, raw.size(), err, acct) || (fp8 && !dalloc(d, dsc, sraw.size(), err, acct))) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dw, raw.data(), raw.size(), cudaMemcpyHostToDevice));
        if (fp8) CK(cudaMemcpy(dsc, sraw.data(), sraw.size(), cudaMemcpyHostToDevice));
        m.w = dw;
        m.sc = dsc;
        if (!x3::supported(m)) {
            err = pre + ": k " + std::to_string(m.k) + " n " + std::to_string(m.n) + " not covered by the plain kernels";
            return false;
        }
        return true;
    }
    // wo_a's group gi: a slice of the EXL3 pack's, or rows [gi o_lora, (gi + 1) o_lora) of the plain weight
    bool wo_a(Dev& d, const std::string& A, int gi, x3::Mat& m, std::string& err, size_t* acct) {
        if (pk.get(A + "wo_a.weight") != nullptr) return dmat(d, A + "wo_a", m, err, acct, 0, 1, true, 0, gi * g.o_lora, g.o_lora);
        return mat(d, A + "wo_a.slice." + std::to_string(gi), m, err, acct);
    }
    // a float tensor (F32 / F16 / BF16) as F32 on the device
    bool f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want = 0) {
        const Tensor* t = pk.get(name);
        if (t == nullptr) {
            err = name + ": not in the pack";
            return false;
        }
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        size_t n = 0;
        std::vector<float> v;
        if (t->dtype == "F32") {
            n = t->bytes / 4;
            v.resize(n);
            std::memcpy(v.data(), raw.data(), t->bytes);
        } else if (t->dtype == "F16" || t->dtype == "BF16") {
            n = t->bytes / 2;
            v.resize(n);
            const uint16_t* s = (const uint16_t*) raw.data();
            for (size_t i = 0; i < n; ++i) v[i] = t->dtype == "F16" ? f16_to_f32(s[i]) : bf16_to_f32(s[i]);
        } else {
            err = name + ": dtype " + t->dtype + " is not a float";
            return false;
        }
        if (want != 0 && n != want) {
            err = name + ": " + std::to_string(n) + " values, expected " + std::to_string(want);
            return false;
        }
        if (!dalloc(d, p, n, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
        return true;
    }
    // an F16 tensor's bits on the device (a BF16 one rounded to F16: the 1.62-bit SAGE pack's router and heads)
    bool f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct) {
        const Tensor* t = pk.get(name);
        if (t == nullptr || (t->dtype != "F16" && t->dtype != "BF16")) {
            err = name + ": not an F16 (or BF16) tensor in the pack";
            return false;
        }
        if (!dalloc(d, p, t->bytes / 2, err, acct)) return false;
        if (t->dtype == "F16") return upload(d, *t, p, err);
        std::vector<uint16_t> v(t->bytes / 2);
        if (!pk.read(*t, v.data())) {
            err = name + ": read failed";
            return false;
        }
        for (uint16_t& x : v) x = f32_to_f16(bf16_to_f32(x));
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, v.data(), t->bytes, cudaMemcpyHostToDevice));
        return true;
    }

    size_t mat_bytes(const std::string& pre) const {
        size_t b = 0;
        static const std::vector<const char*> exl3 = {".trellis", ".suh", ".svh"}, plain = {".weight", ".scale"};
        for (const char* s : pk.get(pre + ".trellis") != nullptr ? exl3 : plain)
            if (const Tensor* t = pk.get(pre + s)) b += align256(t->bytes);
        return b;
    }
    size_t tensor_bytes(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return t ? align256(t->bytes * (t->dtype == "F32" ? 1 : 2)) : 0;
    }

    // ---- expert regions and blob layout
    bool expert_layout(Layer& l, std::string& err) {
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        const char* pieces[3] = {".trellis", ".suh", ".svh"};
        l.region_off.resize((size_t) l.n_exp);
        l.region_bytes.resize((size_t) l.n_exp);
        l.region_file.resize((size_t) l.n_exp);
        // each expert's (gate, up, down) K: its class
        std::map<std::array<int, 3>, int> by;
        std::vector<std::array<int, 3>> ks((size_t) l.n_exp);
        for (int e = 0; e < l.n_exp; ++e) {
            const std::string E = P + std::to_string(e) + ".";
            for (int r = 0; r < 3; ++r) {
                const Tensor* tr = pk.get(E + parts[r] + ".trellis");
                if (tr == nullptr || tr->shape.size() != 3 || pk.get(E + parts[r] + ".mul1") == nullptr) {
                    err = E + parts[r] + ": not an EXL3 (mul1) matrix";
                    return false;
                }
                ks[(size_t) e][(size_t) r] = (int) tr->shape[2] / 16;
            }
            by.emplace(ks[(size_t) e], 0);
        }
        const bool mix = by.size() > 1;
        if (by.size() > 255) {
            err = P + ": " + std::to_string(by.size()) + " expert layouts";
            return false;
        }
        l.cls.assign(by.size(), Layer::XCls{});
        l.ecls.clear();
        if (mix) l.ecls.resize((size_t) l.n_exp);
        int ci = 0;
        for (auto& kv : by) kv.second = ci++;
        std::vector<int> first(by.size(), -1);
        for (int e = 0; e < l.n_exp; ++e) {
            const int c = by[ks[(size_t) e]];
            if (mix) l.ecls[(size_t) e] = (uint8_t) c;
            if (first[(size_t) c] < 0) first[(size_t) c] = e;
        }
        // each class's blob layout (a mixed layer's after its header) from its first expert
        for (size_t c = 0; c < l.cls.size(); ++c) {
            Layer::XCls& X = l.cls[c];
            const std::string E = P + std::to_string(first[c]) + ".";
            size_t off = mix ? x3::kBlobHdr : 0;
            for (int r = 0; r < 3; ++r) {
                X.lay.K[r] = ks[(size_t) first[c]][(size_t) r];
                X.lay.trellis[r] = off;
                off += align256(pk.get(E + parts[r] + ".trellis")->bytes);
                X.lay.suh[r] = off;
                off += align256(pk.get(E + parts[r] + ".suh")->bytes);
                X.lay.svh[r] = off;
                off += align256(pk.get(E + parts[r] + ".svh")->bytes);
            }
            X.lay.cb = 2;
            X.lay.hdr = mix ? 1 : 0;
            X.blob = off;
        }
        for (int e = 0; e < l.n_exp; ++e) {
            const std::string E = P + std::to_string(e) + ".";
            Layer::XCls& X = xc(l, e);
            const bool lead = first[(size_t) (mix ? l.ecls[(size_t) e] : 0)] == e;
            uint64_t lo = UINT64_MAX, hi = 0;
            int file = -1;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const Tensor* t = pk.get(E + parts[r] + pieces[q]);
                    if (t == nullptr) {
                        err = E + parts[r] + pieces[q] + ": missing";
                        return false;
                    }
                    if (file < 0) file = t->file;
                    if (t->file != file) {
                        err = E + ": split across files";
                        return false;
                    }
                    lo = std::min(lo, t->off);
                    hi = std::max(hi, t->off + t->bytes);
                    if (lead) X.piece_bytes[r][q] = t->bytes;
                    else if (t->bytes != X.piece_bytes[r][q]) {
                        err = E + parts[r] + ": not the shape of its class's first expert";
                        return false;
                    }
                }
            l.region_off[(size_t) e] = lo;
            l.region_bytes[(size_t) e] = hi - lo;
            l.region_file[(size_t) e] = file;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const size_t src = pk.get(E + parts[r] + pieces[q])->off - lo;
                    if (lead) X.piece_src[r][q] = src;
                    else if (src != X.piece_src[r][q]) {
                        err = E + ": the pieces lie differently from its class's first expert's";
                        return false;
                    }
                }
        }
        return true;
    }
    // a mixed layer's blob header (BlobHdr): the layout lay at the blob's first bytes
    static void put_hdr(const x3::ExpertLayout& lay, uint8_t* blob) {
        if (!lay.hdr) return;
        x3::BlobHdr h{};
        for (int r = 0; r < 3; ++r) {
            h.suh[r] = lay.suh[r];
            h.svh[r] = lay.svh[r];
            h.tr[r] = lay.trellis[r];
            h.K[r] = lay.K[r];
        }
        h.cb = lay.cb;
        std::memset(blob, 0, x3::kBlobHdr);
        std::memcpy(blob, &h, sizeof h);
    }
    // the region of expert e (read from the file) -> its blob
    void repack(const Layer& l, int e, const uint8_t* region, uint8_t* blob) const {
        const Layer::XCls& X = xc(l, e);
        put_hdr(X.lay, blob);
        for (int r = 0; r < 3; ++r) {
            const size_t dst[3] = {X.lay.trellis[r], X.lay.suh[r], X.lay.svh[r]};
            for (int q = 0; q < 3; ++q) std::memcpy(blob + dst[q], region + X.piece_src[r][q], X.piece_bytes[r][q]);
        }
    }

    // the region of expert e -> its CPU-lane part X (columns [X n / NH, (X + 1) n / NH) of each matrix)
    void repack_part(const Layer& l, int e, const uint8_t* region, int X, uint8_t* dst) const {
        const Layer::XCls& C = xc(l, e);
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * C.lay.K[r];
            const int k = C.hl.k[r], nt = C.hl.n[r] * NH / 16, nth = C.hl.n[r] / 16;
            const uint8_t* tr = region + C.piece_src[r][0];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + C.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + C.hl.suh[r], region + C.piece_src[r][1], (size_t) k * 2);
            std::memcpy(dst + C.hl.svh[r], region + C.piece_src[r][2] + (size_t) X * C.hl.n[r] * 2, (size_t) C.hl.n[r] * 2);
        }
    }

    // ---- loading
    bool load(const Options& opt, std::string& err) {
        o = opt;
        const double t0 = now_ms();
        if (!pk.load(o.pack, err)) return false;
        g.from(pk);
        if (g.dim == 0 || g.nl == 0 || (int) g.ratios.size() < g.nl) {
            err = o.pack + "/ds4.txt: incomplete geometry";
            return false;
        }
        if (g.hd != 512 || g.ix_hd != 128 || g.k_exp > 8 || g.hc > 8) {
            err = "geometry not covered (head_dim " + std::to_string(g.hd) + ", index_head_dim " +
                  std::to_string(g.ix_hd) + ", experts per token " + std::to_string(g.k_exp) + ")";
            return false;
        }
        freqs_win = rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow);
        freqs_comp = rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow);

        // devices
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            err = "no CUDA device";
            return false;
        }
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < ndev; ++i) ids.push_back(i);
        devs.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            Dev& d = devs[i];
            d.id = ids[i];
            CK(cudaSetDevice(d.id));
            CK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
            CK(cudaStreamCreateWithFlags(&d.st2, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&d.hc_go, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&d.hc_done, cudaEventDisableTiming));
            if (!x3::prepare(d.st)) {
                err = "exl3 scratch on CUDA" + std::to_string(d.id);
                return false;
            }
            CK(cudaMemGetInfo(&d.free0, &d.total));
        }
        // the groups: runs of o.tp devices of one kind (their memory), a device unlike the next alone - or, o.tp the
        // number of GPUs, one group of all of them whatever their kind (its tightest member sets the experts' room)
        par = o.threads || o.tp > 1;
        const bool tp_all = o.tp > 1 && o.tp == (int) devs.size();
        for (size_t i = 0; i < devs.size();) {
            Group G;
            G.mem.push_back((int) i);
            size_t j = i + 1;
            while ((int) G.mem.size() < o.tp && j < devs.size() && (tp_all || devs[j].total == devs[i].total))
                G.mem.push_back((int) j++);
            if ((int) G.mem.size() < o.tp && G.mem.size() > 1) {   // a short run at the end: devices of their own
                for (int m : G.mem) {
                    Group one;
                    one.mem.push_back(m);
                    groups.push_back(one);
                }
            } else groups.push_back(G);
            i = j;
        }
        for (size_t gi = 0; gi < groups.size(); ++gi)
            for (size_t m = 0; m < groups[gi].mem.size(); ++m) {
                devs[(size_t) groups[gi].mem[m]].grp = (int) gi;
                devs[(size_t) groups[gi].mem[m]].mi = (int) m;
            }
        // whether the CPU lane's experts are read in place (--mmap-experts; auto: the pack's expert bytes beyond what the
        // GPUs can take - their free memory less the other weights and the prompt path's buffers - do not fit the RAM
        // this process can get beside kRamMargin), decided here because it sets the prompt chunk's default: in place
        // every chunk streams every expert VRAM does not hold from the files, so the larger the chunk the fewer times
        if (o.mmap_experts >= 0) file_experts = o.mmap_experts > 0;
        else {
            double exp_b = 0, other_b = 0, room = 0;
            for (const auto& kt : pk.t) {
                const std::string& n = kt.first;
                if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                    (!o.dspark && n.rfind("mtp.", 0) == 0))
                    continue;
                (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
            }
            for (const Dev& d : devs)
                room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) - (par ? 1.5 : 0.4) * (1 << 30)) *
                        o.vram_experts;
            file_experts = exp_b - std::max(0.0, room - other_b) + (double) kRamMargin > (double) ram_avail();
        }
        // (4096 in RAM: at --tp 9 a 16K prompt 896 -> 1336 tok/s and a 64K one 1132 -> 1651, its buffers ~240 experts
        // a GPU out of VRAM - decode 54.4 -> 52.3 tok/s at 3.5 bits; 8192 leaves the CPU lane more than the RAM)
        chunk = par ? (o.prefill_chunk >= 0 ? o.prefill_chunk : file_experts ? kVlRows : 4096) : 0;
        if (const char* v = std::getenv("DS4_CHUNK")) chunk = par ? std::atoi(v) : 0;
        if (const char* v = std::getenv("DS4_CHUNK_MIN")) chunk_min = std::max(1, std::atoi(v));
        if (const char* v = std::getenv("DS4_PREFILL_KEEP")) o.prefill_keep = std::atoi(v);
        recent_n = std::max(0, (int) env_f("DS4_RECENT", chunk > 0 && !file_experts ? 32.0f : 0.0f));
        if (chunk > 0) {
            chunk = std::min(kVlRows, std::max(256, chunk / 16 * 16));   // (route()'s image bits cover a chunk)
            eg_cap = (size_t) chunk;
            tap_cap = std::max(kWin, 128);
        }
        // the vision tower on the first GPU, before the split: the layers get what it leaves
        if (o.vision && !load_vision(err)) return false;
        if (o.vision_only) return true;

        // layers: roles and the sources they read
        if (!o.dspark) g.n_mtp = 0;
        L.resize((size_t) (g.nl + g.n_mtp));
        int last_src = -1;
        for (int il = 0; il < g.nl + g.n_mtp; ++il) {
            Layer& l = L[(size_t) il];
            l.il = il;
            l.mtp = il >= g.nl;
            l.name = l.mtp ? "mtp." + std::to_string(il - g.nl) + "." : "layers." + std::to_string(il) + ".";
            l.n_exp = l.mtp ? g.d_n_exp : g.n_exp;
            l.k_exp = l.mtp ? g.d_k : g.k_exp;
            l.ratio = g.ratios[(size_t) il];
            if (l.mtp) {   // window attention only: no compressor, indexer or Engram
                if (l.ratio != 0 || l.n_exp <= 0 || l.k_exp <= 0 || l.k_exp > kPicks) {
                    err = l.name + ": a DSpark block of compress ratio " + std::to_string(l.ratio) + ", " +
                          std::to_string(l.n_exp) + " experts / " + std::to_string(l.k_exp) + " is not covered";
                    return false;
                }
                if (!expert_layout(l, err)) return false;
                continue;
            }
            l.kv_src = std::find(g.kv_src.begin(), g.kv_src.end(), il) != g.kv_src.end();
            l.ix_src = std::find(g.ix_src.begin(), g.ix_src.end(), il) != g.ix_src.end();
            l.owns_k = l.kv_src;
            l.cand_src = il == g.cand_src;
            l.uses_cand = g.cand_src >= 0 && g.cand_src < il && l.ix_src;
            if (l.kv_src) {
                Src s;
                s.layer = il;
                s.ratio = l.ratio;
                s.cap = (o.max_context + l.ratio - 1) / l.ratio + 1;
                srcs.push_back(s);
                last_src = (int) srcs.size() - 1;
            }
            if (l.ratio > 0) {
                if (last_src < 0 || srcs[(size_t) last_src].ratio != l.ratio) {
                    err = "layer " + std::to_string(il) + ": compress ratio " + std::to_string(l.ratio) +
                          " without a KV source of that ratio before it";
                    return false;
                }
                l.src = last_src;
                l.ik_src = last_src;
            }
            auto eg = std::find(g.eg_layers.begin(), g.eg_layers.end(), il);
            if (eg != g.eg_layers.end()) l.eg = (int) (eg - g.eg_layers.begin());
            if (!expert_layout(l, err)) return false;
        }
        // a SAGE pack's layers (each expert matrix its own K: several layouts a layer) keep their experts where the
        // load puts them - no cache swaps, no recent slots (both move experts between slots of one size)
        if (std::any_of(L.begin(), L.end(), [](const Layer& l) { return mixed(l); })) {
            if (o.expert_cache || recent_n > 0)
                std::fprintf(stderr, "ds4: the experts come in several layouts a layer: the expert cache and the recent "
                                     "slots are off\n");
            o.expert_cache = false;
            recent_n = 0;
        }

        // YOCO's split: the first layer of ratio 1 a KV source, no KV source and no Engram layer after it (where they come
        // later the consumers' rows would matter to them)
        for (int il = 0; il < g.nl && n_pre < 0; ++il)
            if (L[(size_t) il].ratio == 1 && L[(size_t) il].kv_src) n_pre = il;
        for (int il = n_pre + 1; n_pre >= 0 && il < g.nl; ++il)
            if (L[(size_t) il].kv_src || L[(size_t) il].eg >= 0) n_pre = -1;
        if (n_pre >= 0 && L[(size_t) n_pre].eg >= 0) n_pre = -1;

        // the split: each layer's dense weights + its experts, the head on the last GPU
        for (Layer& l : L) {
            const std::string& P = l.name;
            size_t b = 0;
            if (l.mtp && l.il == g.nl) b += mat_bytes("mtp.0.main_proj");
            for (const char* m : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1",
                                  "ffn.shared_experts.w2", "ffn.shared_experts.w3", "attn.compressor.wkv",
                                  "attn.compressor.wgate", "attn.indexer.wq_b", "attn.indexer.wk", "engram.wkv"})
                b += mat_bytes(P + m);
            for (int gi = 0; gi < g.o_groups; ++gi) b += mat_bytes(P + "attn.wo_a.slice." + std::to_string(gi));
            b += mat_bytes(P + "attn.wo_a");   // (a plain weight's: one tensor)
            b += tensor_bytes(P + "hc_attn_fn") + tensor_bytes(P + "hc_ffn_fn") + tensor_bytes(P + "ffn.gate.weight");
            b += (size_t) g.win * g.hd * 2 + (2u << 20);
            l.dense_bytes = b;
        }
        const size_t head_b = mat_bytes("head") + (size_t) g.vocab * 4 + (16u << 20);
        const size_t per_dev_fixed = (size_t) o.reserve_mib << 20;
        const size_t kv_per_src_dev = 0;   // counted after the split
        (void) kv_per_src_dev;
        std::vector<double> avail(groups.size(), 0.0);
        double avail_sum = 0;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            for (int m : groups[gi].mem)
                avail[gi] += std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed - 400.0 * (1 << 20));
            avail_sum += avail[gi];
        }
        double need_sum = (double) head_b;
        for (const Layer& l : L) need_sum += (double) l.dense_bytes + (double) all_blobs(l);
        {
            // contiguous layer ranges, each group's share of the total by its free memory
            size_t d = 0;
            double cum = 0, target = avail[0] / avail_sum * need_sum;
            std::vector<Layer*> order;   // the drafter's blocks first (the first GPU), then the trunk's layers
            for (Layer& l : L)
                if (l.mtp) order.push_back(&l);
            for (Layer& l : L)
                if (!l.mtp) order.push_back(&l);
            for (Layer* lp : order) {
                Layer& l = *lp;
                const double lb = (double) l.dense_bytes + (double) all_blobs(l);
                if (cum + lb / 2 > target && d + 1 < groups.size()) {
                    ++d;
                    target += avail[d] / avail_sum * need_sum;
                }
                l.grp = (int) d;
                cum += lb;
            }
            // with the chunked prompt path: the contiguous split that streams the least into its busiest member (its
            // VRAM short of its layers' experts, per member) - a chunk's groups run as a pipeline, its slowest stage
            // sets the pace; a token's decode passes every layer either way, the experts in VRAM the same in all
            if (chunk > 0 && groups.size() > 1 && groups.size() <= 6 && std::getenv("DS4_SPLIT") == nullptr) {
                const int G = (int) groups.size(), NL = g.nl;
                std::vector<double> cap(groups.size());
                for (int gi = 0; gi < G; ++gi) {
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    double mn = 1e300;
                    for (int m : groups[(size_t) gi].mem)
                        mn = std::min(mn, std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed -
                                                            400.0 * (1 << 20) - 1.1 * (1 << 30)));   // (the chunk's buffers)
                    cap[(size_t) gi] = mn * nm;
                }
                for (const Layer& l : L)
                    if (l.mtp) cap[0] -= (double) l.dense_bytes + (double) all_blobs(l);
                cap[(size_t) G - 1] -= (double) head_b;
                std::vector<double> pe(NL + 1, 0.0), pd(NL + 1, 0.0);   // prefix sums: experts, dense
                for (int il = 0; il < NL; ++il) {
                    pe[(size_t) il + 1] = pe[(size_t) il] + (double) all_blobs(L[(size_t) il]);
                    pd[(size_t) il + 1] = pd[(size_t) il] + (double) L[(size_t) il].dense_bytes;
                }
                auto cost = [&](int gi, int a, int b) {   // per-member streaming of layers [a, b) on group gi
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    const double dense = (pd[(size_t) b] - pd[(size_t) a]) * (nm > 1 ? 1.3 : 1.0);
                    return std::max(0.0, pe[(size_t) b] - pe[(size_t) a] - (cap[(size_t) gi] - dense)) / nm;
                };
                // best[gi][b]: the least max cost of layers [0, b) over groups 0..gi (each at least one layer)
                std::vector<std::vector<double>> best((size_t) G, std::vector<double>((size_t) NL + 1, 1e300));
                std::vector<std::vector<int>> cut((size_t) G, std::vector<int>((size_t) NL + 1, 0));
                for (int b = 1; b <= NL; ++b) best[0][(size_t) b] = cost(0, 0, b);
                for (int gi = 1; gi < G; ++gi)
                    for (int b = gi + 1; b <= NL; ++b)
                        for (int a = gi; a < b; ++a) {
                            const double v = std::max(best[(size_t) gi - 1][(size_t) a], cost(gi, a, b));
                            if (v < best[(size_t) gi][(size_t) b]) {
                                best[(size_t) gi][(size_t) b] = v;
                                cut[(size_t) gi][(size_t) b] = a;
                            }
                        }
                int b = NL;
                for (int gi = G - 1; gi >= 0; --gi) {
                    const int a = gi > 0 ? cut[(size_t) gi][(size_t) b] : 0;
                    for (int il = a; il < b; ++il) L[(size_t) il].grp = gi;
                    b = a;
                }
            }
            // DS4_SPLIT=a,b,..: the trunk layers each group takes (the drafter stays with the first)
            if (const char* sp = std::getenv("DS4_SPLIT")) {
                std::vector<int> n;
                for (const char* q = sp; *q;) {
                    n.push_back(std::atoi(q));
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
                int il = 0;
                for (size_t gi = 0; gi < n.size() && gi < groups.size(); ++gi)
                    for (int j = 0; j < n[gi] && il < g.nl; ++j) L[(size_t) il++].grp = (int) gi;
                for (; il < g.nl; ++il) L[(size_t) il].grp = (int) groups.size() - 1;
            }
            for (Layer& l : L) {
                l.dev = groups[(size_t) l.grp].mem[0];
                l.nm = l.mtp ? 1 : (int) groups[(size_t) l.grp].mem.size();   // (the drafter's blocks: on the lead alone)
            }
            head_dev = L[(size_t) g.nl - 1].dev;
            d_dev = g.n_mtp > 0 ? L[(size_t) g.nl].dev : 0;
        }
        // the groups' members: copies of each layer's device state, a member's part of an expert
        for (Layer& l : L) {
            l.ff0 = 0;
            l.ffn = g.ff;
            l.ah0 = l.og0 = 0;
            l.ahn = g.heads;
            l.ogn = g.o_groups;
            if (l.nm == 1) continue;
            const int na = head_members(l.nm);
            if (g.heads % na != 0 || g.o_groups % na != 0 || g.ff % 128 != 0 || g.ff / 128 < l.nm) {
                err = "tensor parallel " + std::to_string(l.nm) + ": the heads, output groups or FFN do not split";
                return false;
            }
            for (int m = 1; m < l.nm; ++m) {
                Layer c = l;
                c.mates.clear();
                c.mi = m;
                c.dev = groups[(size_t) l.grp].mem[(size_t) m];
                l.mates.push_back(std::move(c));
            }
            // each member's FFN columns (split128: as even as the 128-blocks go) and its part of an expert's layout
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const int ha = m - (l.nm - na);   // (its place among the members with heads)
                x.ahn = ha < 0 ? 0 : g.heads / na;
                x.ah0 = ha < 0 ? 0 : ha * x.ahn;
                x.ogn = ha < 0 ? 0 : g.o_groups / na;
                x.og0 = ha < 0 ? 0 : ha * x.ogn;
                split128(g.ff, l.nm, m, x.ff0, x.ffn, l.il % l.nm);   // (rot: the odd blocks go round the members)
                for (Layer::XCls& c : x.cls) {   // (each class's part: a mixed layer's after its header too)
                    size_t off = c.lay.hdr ? x3::kBlobHdr : 0;
                    for (int r = 0; r < 3; ++r) {
                        const int k = r < 2 ? g.dim : x.ffn, n = r < 2 ? x.ffn : g.dim;
                        c.hlay.K[r] = c.lay.K[r];
                        c.hlay.trellis[r] = off;
                        off += align256(x3::trellis_bytes(k, n, c.lay.K[r]));
                        c.hlay.suh[r] = off;
                        off += align256((size_t) k * 2);
                        c.hlay.svh[r] = off;
                        off += align256((size_t) n * 2);
                    }
                    c.hlay.cb = c.lay.cb;
                    c.hlay.hdr = c.lay.hdr;
                    c.hblob = off;
                }
            }
        }
        // which devices hold each source's compressed KV / index keys
        for (Src& s : srcs) {
            s.ckv.assign(devs.size(), nullptr);
            s.ik.assign(devs.size(), nullptr);
        }
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                const Layer& l = member(l0, m);
                if (l.src >= 0) {
                    srcs[(size_t) l.src].ckv[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.src].layer].grp) srcs[(size_t) l.src].remote = true;
                }
                if (l.ix_src && l.ik_src >= 0) {
                    srcs[(size_t) l.ik_src].ik[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.ik_src].layer].grp) srcs[(size_t) l.ik_src].remote = true;
                }
            }

        // per device: the scratch, the caches, the dense weights
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            const int D = g.dim, H = g.hc;
            const size_t W = kWin, P = (size_t) kWin * kPicks, cap = (size_t) o.max_context + 64;
            bool ok = dalloc(d, d.h, W * H * D, err) && dalloc(d, d.h2, W * H * D, err) && dalloc(d, d.x, W * D, err) &&
                      dalloc(d, d.qr, W * g.q_lora, err) && dalloc(d, d.q, W * g.heads * g.hd, err) &&
                      dalloc(d, d.kvr, W * g.hd, err) && dalloc(d, d.ckr, W * g.hd, err) &&
                      dalloc(d, d.cgr, W * g.hd, err) && dalloc(d, d.latent, W * g.hd, err) &&
                      dalloc(d, d.ikr, W * g.ix_hd, err) && dalloc(d, d.iq, W * g.ix_heads * g.ix_hd, err) &&
                      dalloc(d, d.iw, W * 64, err) && dalloc(d, d.scores, W * cap, err) &&
                      dalloc(d, d.tks.val, kTksRows * tks_p() * g.ix_topk, err) &&
                      dalloc(d, d.tks.idx, kTksRows * tks_p() * g.ix_topk, err) &&
                      dalloc(d, d.tks.cnt, kTksRows * tks_p(), err) && dalloc(d, d.tks.kth, kTksRows * tks_p(), err) &&
                      dalloc(d, d.bm, W * bm_stride(), err) && dalloc(d, d.o, W * g.heads * g.hd, err) &&
                      dalloc(d, d.oa, W * g.o_groups * g.o_lora, err) && dalloc(d, d.ao, W * D, err) &&
                      dalloc(d, d.mix, W * 32, err) && dalloc(d, d.a_pre, W * 8, err) && dalloc(d, d.a_post, W * 8, err) &&
                      dalloc(d, d.a_comb, W * 64, err) && dalloc(d, d.f_pre, W * 8, err) &&
                      dalloc(d, d.f_post, W * 8, err) && dalloc(d, d.f_comb, W * 64, err) && dalloc(d, d.pre, W * 8, err) &&
                      dalloc(d, d.glog, W * 1024, err) && dalloc(d, d.rw, W * kPicks, err) &&
                      dalloc(d, d.ridx, W * kPicks, err) && dalloc(d, d.moe_out, W * D, err) &&
                      dalloc(d, d.sg, W * g.ff, err) && dalloc(d, d.su, W * g.ff, err) && dalloc(d, d.sh, W * g.ff, err) &&
                      dalloc(d, d.sh_out, W * D, err) && dalloc(d, d.cpu_part, W * D, err) &&
                      dalloc(d, d.freqs_win, 64, err) && dalloc(d, d.freqs_comp, 64, err) &&
                      dalloc(d, d.topk, W * topk_stride(), err) && dalloc(d, d.cand, W * cand_stride(), err) &&
                      dalloc(d, d.tok, 16, err) && dalloc(d, d.emb_rows, W * D, err) &&
                      dalloc(d, d.ext, ((size_t) g.win + W) * g.hd, err) && dalloc(d, d.xe, P * D, err) &&
                      dalloc(d, d.xg, P * g.ff, err) && dalloc(d, d.xu, P * g.ff, err) && dalloc(d, d.xhh, P * g.ff, err) &&
                      dalloc(d, d.ed, P * D, err) && dalloc(d, d.plan_ptr, P, err);
            if (!ok) return false;
            d.tks.T = (int) kTksRows, d.tks.P = (int) tks_p(), d.tks.K = g.ix_topk;
            if (!g.eg_layers.empty()) {
                const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
                ok = dalloc(d, d.eg_rows, W * cols * (pk.eg_hd + pk.eg_hd / 32), err) &&
                     dalloc(d, d.eg_x, W * cols * pk.eg_hd, err) && dalloc(d, d.eg_kv, W * (g.hc + 1) * D, err);
                if (!ok) return false;
            }
            if ((int) di == head_dev && !dalloc(d, d.logits, (size_t) kWin * g.vocab, err)) return false;
            if (!halloc(d.bounce, bounce_bytes(), err)) return false;
            if (!dalloc(d, d.xcnt, 4, err) || !hmap(d.hand, bounce_bytes(), err) || !hmap(d.hand_flag, 32, err) ||
                !dalloc(d, d.grp_ptr, P, err) || !dalloc(d, d.grp_start, P + 1, err) || !dalloc(d, d.grp_ent, P, err) ||
                !dalloc(d, d.n_groups, 4, err))
                return false;
            if (g.n_mtp > 0 && !dalloc(d, d.tap, (size_t) tap_cap * D, err)) return false;
            if (g.n_mtp > 0 && (int) di == head_dev &&
                (!dalloc(d, d_confv, kWin, err) || !dalloc(d, d_spec, kWin, err) || !halloc(h_spec, kWin, err) ||
                 !dalloc(d, d_ints, 64, err)))
                return false;
            CK(cudaEventCreateWithFlags(&d.ev_out, cudaEventDisableTiming));
            cudaSetDevice(d.id);
            CK(cudaMemcpy(d.freqs_win, freqs_win.data(), freqs_win.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d.freqs_comp, freqs_comp.data(), freqs_comp.size() * 4, cudaMemcpyHostToDevice));
            for (Src& s : srcs) {
                if (s.h_ckv == nullptr) {   // (its events belong to the source layer's device: it records them)
                    if (!halloc(s.h_ckv, (size_t) kWin * kv_row(), err) || !halloc(s.h_ik, (size_t) kWin * ik_row(), err))
                        return false;
                    CK(cudaSetDevice(devs[(size_t) L[(size_t) s.layer].dev].id));
                    CK(cudaEventCreateWithFlags(&s.ev_ckv, cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&s.ev_ik, cudaEventDisableTiming));
                    CK(cudaSetDevice(d.id));
                }
                if (s.ckv[di] != nullptr && !dalloc(d, s.ckv[di], (size_t) s.cap * kv_row(), err, &d.caches)) return false;
                if (s.ik[di] != nullptr && !dalloc(d, s.ik[di], (size_t) s.cap * ik_row(), err, &d.caches))
                    return false;
            }
        }
        for (Src& sr : srcs) {
            if (sr.remote && (!hmap(sr.x_stage, x_cap() * (g.hd + g.ix_hd) * 2, err) || !hmap(sr.x_flag, 32, err)))
                return false;
            if (sr.remote && chunk > 0 &&
                (!hmap(sr.x_stage1, x_cap() * (g.hd + g.ix_hd) * 2, err) || !hmap(sr.x_flag1, 32, err) ||
                 !hmap(sr.x_ack, (size_t) 2 * devs.size() * 32, err)))
                return false;
            const int sg = L[(size_t) sr.layer].grp;
            for (size_t di = 0; di < devs.size(); ++di)
                if ((sr.ckv[di] != nullptr || sr.ik[di] != nullptr) && devs[di].grp != sg) sr.readers.push_back((int) di);
        }
        for (Group& G : groups) {
            G.ar.nm = (int) G.mem.size();
            G.ar.slots = 4;
            G.ar.stride = (size_t) kWin * g.dim;
            if (G.ar.nm > 1 && (!hmap(G.ar.data, (size_t) G.ar.nm * G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.flags, (size_t) G.ar.nm * G.ar.slots * 32, err)))
                return false;
            // three members or more: the reduce-scatter's sums and its per-block flags (allreduce)
            if (G.ar.nm > 2 && (!hmap(G.ar.sums, (size_t) G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.bflags, (size_t) 2 * G.ar.nm * G.ar.slots * kArBlocks * 8, err)))
                return false;
            if (G.ar.nm > 1 && &G == &groups[(size_t) devs[(size_t) head_dev].grp] &&
                (!hmap(G.lg_stage, (size_t) 2 * (G.ar.nm - 1) * kWin * ((g.vocab / 128 + G.ar.nm - 1) / G.ar.nm * 128), err) ||
                 !hmap(G.lg_flag, (size_t) 2 * (G.ar.nm - 1) * 32, err) || !hmap(G.lg_ack, (size_t) 2 * 32, err)))
                return false;
            G.arb = G.ar;   // (a chunk's: two slots of chunk rows)
            G.arb.slots = 2;
            G.arb.stride = (size_t) chunk * g.dim;
            if (chunk > 0 && G.arb.nm > 1 && (!hmap(G.arb.data, (size_t) G.arb.nm * G.arb.slots * G.arb.stride, err) ||
                                              !hmap(G.arb.flags, (size_t) G.arb.nm * G.arb.slots * 32, err)))
                return false;
            if (chunk > 0 && G.arb.nm > 2 &&
                (!hmap(G.arb.sums, (size_t) G.arb.slots * G.arb.stride, err) ||
                 !hmap(G.arb.bflags, (size_t) 2 * G.arb.nm * G.arb.slots * kArBlocks * 8, err)))
                return false;
            // 5 members or more: a chunk's sums by the host (DS4_AR_HOST: the members from which, 0 off; threads
            // DS4_AR_HOST_THREADS, 32)
            const char* hm = std::getenv("DS4_AR_HOST");
            const int host_from = hm != nullptr ? std::atoi(hm) : 5;
            if (chunk > 0 && host_from > 0 && G.arb.nm >= host_from && G.arb.nm <= 16 && G.arb.sums != nullptr) {
                const char* ht = std::getenv("DS4_AR_HOST_THREADS");
                std::vector<int> ids;
                for (int m : G.mem) ids.push_back(devs[(size_t) m].id);
                G.hred = std::make_shared<HostReducer>(G.arb, ids, G.arb.stride, ht != nullptr ? std::max(1, std::atoi(ht)) : 32);
                if (!G.hred->ok()) G.hred.reset();   // (the sums by the copy engines then)
            }
        }
        if (g.n_mtp > 0) {
            if (!halloc(h_taps, (size_t) tap_cap * g.d_targets.size() * g.dim, err)) return false;
            CK(cudaSetDevice(devs[(size_t) d_dev].id));
            CK(cudaEventCreateWithFlags(&ev_seed, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_seed, devs[(size_t) d_dev].st));
        }
        // the dense weights, each device's layers by a thread of its own
        {
            std::vector<std::string> errs(devs.size());
            std::vector<std::thread> th;
            for (size_t di = 0; di < devs.size(); ++di)
                th.emplace_back([this, di, &errs] {
                    std::string e;
                    if (!load_dense((int) di, e)) errs[di] = e;
                });
            for (auto& t : th) t.join();
            for (auto& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
        }
        if (o.verbose)
            std::fprintf(stderr, "ds4: dense weights loaded in %.1f s\n", (now_ms() - t0) / 1000.0);
        // the markov steps over a head group of two or more (DS4_MK_SPLIT=0: the lead's whole rows, as before): E
        // host-mapped, the steps' slots
        if (g.n_mtp > 0 && head_nm() > 1 &&
            !(std::getenv("DS4_MK_SPLIT") != nullptr && std::atoi(std::getenv("DS4_MK_SPLIT")) == 0)) {
            const Tensor* te = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.embed.weight");
            if (te == nullptr || te->bytes != (uint64_t) g.vocab * g.d_rank * 2) {
                err = "the markov embed: not a BF16 [vocab][rank] table";
                return false;
            }
            if (head_nm() > kSpecMembers) {
                err = "the markov steps over a head group of more than " + std::to_string(kSpecMembers) + " members";
                return false;
            }
            if (!hmap(h_mk_embed, (size_t) g.vocab * g.d_rank, err) || !pk.read(*te, h_mk_embed) ||
                !hmap(mk_slots, (size_t) kWin * head_nm() * 4, err) ||
                !hmap(mk_lists, (size_t) kWin * head_nm() * kSpecList, err)) {
                if (err.empty()) err = "the markov embed: read failed";
                return false;
            }
            devs[(size_t) head_dev].mk_ints = d_ints;   // (the lead's drafted tokens: where the confidence head reads them)
            mk_split = true;
        }

        // the embedding table, pinned on the host
        {
            const Tensor* t = pk.get("embed.weight");
            if (t == nullptr || t->dtype != "BF16" || t->bytes != (uint64_t) g.vocab * g.dim * 2) {
                err = "embed.weight: not a BF16 [vocab][dim] table";
                return false;
            }
            if (!halloc(embed_pinned, (size_t) g.vocab * g.dim, err)) return false;
            if (!pk.read(*t, embed_pinned)) {
                err = "embed.weight: read failed";
                return false;
            }
        }
        // the host side of the CPU lane and the Engram rows
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
        if (!halloc(h_x, (size_t) kWin * g.dim, err) || !halloc(h_w, (size_t) kWin * kPicks, err) ||
            !halloc(h_part, (size_t) kWin * g.dim, err) || !halloc(h_idx, (size_t) kWin * kPicks, err) ||
            !halloc(h_emb, (size_t) std::max(kWin, chunk) * g.dim, err) ||
            !halloc(h_eg, (size_t) std::max(1, pk.eg_layers) * eg_cap * cols * (pk.eg_hd + pk.eg_hd / 32) + 64, err))
            return false;
        prof.assign((size_t) (g.nl + g.n_mtp) * g.n_exp, 0u);
        if (!o.profile.empty()) {
            std::FILE* f = std::fopen(o.profile.c_str(), "rb");
            if (f != nullptr) {
                std::vector<uint32_t> p(prof.size());
                const size_t n = std::fread(p.data(), 4, p.size(), f);
                if (n == p.size() || n == (size_t) g.nl * g.n_exp) std::copy(p.begin(), p.begin() + (long) n, prof.begin());
                std::fclose(f);
            }
        }

        // the chunked prompt path's buffers (before the experts: they take what is left)
        if (chunk > 0 && !alloc_chunks(err)) return false;
        // the CPU lane's experts read in place (decided above): how much of them the load reads ahead, from what VRAM
        // now leaves over
        if (file_experts) {
            double room = 0, all = 0;   // the VRAM place_experts will find for experts; every expert's bytes
            for (Dev& d : devs) {
                cudaSetDevice(d.id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                room += std::max(0.0, (double) fr - (double) ((size_t) o.reserve_mib << 20)) * o.vram_experts;
            }
            for (const Layer& l : L) all += (double) all_blobs(l);
            const double need = std::max(0.0, all - room);
            const uint64_t avail = ram_avail();
            warm_bytes = avail > kRamMargin ? std::min<uint64_t>(avail - kRamMargin, (uint64_t) need) : 0;
            if (avail == ~uint64_t{0}) warm_bytes = (uint64_t) need;
            std::fprintf(stderr,
                         "ds4: the experts VRAM leaves over (~%.1f GB) are read in place from the model's files%s "
                         "(%.1f GB of RAM free to keep them: the most-routed %.1f GB read ahead; prompts in chunks of %d)\n",
                         need / 1e9, o.mmap_experts > 0 ? " (--mmap-experts on)" : ", not copied into RAM",
                         avail == ~uint64_t{0} ? 0.0 : avail / 1e9, warm_bytes / 1e9, chunk);
        }
        // the CPU lane: one pool of workers per NUMA node, each expert split in NH column parts (part X on node X); in
        // place, one part (the files' layout) and one pool over every node's cores (the pages lie where the cache put them)
        cores = node_cores();
        NH = file_experts ? 1 : (int) std::min<size_t>(2, cores.size());
        if (NH > 1 && ((g.ff / NH) % 128 != 0 || (g.dim / NH) % 128 != 0)) NH = 1;
        for (int X = 0; X < (int) cores.size() && X < std::max(NH, 1); ++X) {
            std::vector<int> c = cores[(size_t) X];
            if (file_experts)
                for (size_t Y = 1; Y < cores.size(); ++Y) c.insert(c.end(), cores[Y].begin(), cores[Y].end());
            if (X == 0 && !c.empty()) c.erase(c.begin());   // the host thread's core (it helps node 0's pool)
            int nw = (int) c.size();
            if (o.cpu_threads > 0) nw = std::max(X == 0 ? 0 : 1, std::min(nw, o.cpu_threads / NH - (X == 0 ? 1 : 0)));
            pools.push_back(std::make_unique<Pool>(nw, c, 20000));   // spin through a token's layers (20 ms)
        }
        {
            const size_t E = (size_t) g.dim, FF = (size_t) g.ff, GQ = E / 256, DQ = FF / 128, S = kSlots;
            c_xh.resize((size_t) NH * S * 2 * E);
            c_xsb.resize((size_t) NH * S * 2 * (E / 128));
            c_acc.resize((size_t) NH * S * 2 * GQ * (FF / NH));
            c_xhd.resize(S * FF);
            c_xsd.resize(S * (FF / 128));
            c_accd.resize((size_t) NH * S * DQ * (E / NH));
            c_yd.resize(S * E);
        }
        // the experts: VRAM slots in what each device has left, the rest on the host
        if (!place_experts(err)) return false;
        if (!load_experts(err)) return false;
        if (chunk > 0) {
            // the CPU lane's parts pinned (the chunks stream them to the GPUs by DMA), each layer's virtual slots
            const double tr = now_ms();
            for (auto& m : host_maps)
                if (cudaHostRegister(m.first, m.second, cudaHostRegisterPortable) != cudaSuccess) {
                    err = "pinning the CPU lane's experts (" + std::to_string(m.second >> 20) + " MiB) failed";
                    return false;
                }
            if (o.verbose) std::fprintf(stderr, "ds4: CPU lane's experts pinned in %.1f s\n", (now_ms() - tr) / 1000.0);
            if (file_experts) {   // in place: each device's staging for its ring, and its readers (DS4_READERS, 8)
                size_t mr = 0;
                for (const Layer& l : L)
                    if (!l.mtp)
                        for (uint64_t r : l.region_bytes) mr = std::max<size_t>(mr, r);
                const char* nr = std::getenv("DS4_READERS");
                const int n_readers = std::max(1, nr != nullptr ? std::atoi(nr) : 8);
                for (auto& cp : chunks) {
                    if (cp == nullptr || cp->ring == nullptr) continue;
                    cp->stage_stride = (mr + 4095) & ~(size_t) 4095;
                    if (!halloc(cp->stage, (size_t) cp->ring_n * cp->stage_stride, err)) return false;
                    cp->readers = std::make_unique<Pool>(n_readers - 1, std::vector<int>{}, 50);
                }
            }
            for (Layer& l : L) {
                if (l.mtp) continue;
                if (!halloc(l.vs_pin, (size_t) l.n_exp, err)) return false;
                for (int m = 0; m < l.nm; ++m) {
                    Layer& x = member(l, m);
                    if (!dalloc(devs[(size_t) x.dev], x.vs_dev, (size_t) l.n_exp, err)) return false;
                }
            }
        }
        {
            void* m = nullptr;
            if (cudaHostAlloc(&m, sizeof(Req) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's requests";
                return false;
            }
            host_allocs.push_back(m);
            reqs = (Req*) m;
            if (cudaHostAlloc(&m, sizeof(Resp) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's answers";
                return false;
            }
            host_allocs.push_back(m);
            resps = (Resp*) m;
            std::memset((void*) reqs, 0, sizeof(Req) * kRing);
            std::memset((void*) resps, 0, sizeof(Resp) * kRing);
            for (auto& x : layer_of) x.store(-1);
            const int core = !cores.empty() && !cores[0].empty() ? cores[0][0] : -1;
            service = std::thread([this, core] {
                if (core >= 0) pin_self(core);
                serve_lane();
            });
        }
        if (par && (!std::getenv("DS4_COPY_IDLE") || std::atoi(std::getenv("DS4_COPY_IDLE")) != 0)) {
            idle_ev.assign(devs.size(), nullptr);
            idle_win.reset(new std::atomic<unsigned>[devs.size()]);
            for (size_t di = 0; di < devs.size(); ++di) {
                cudaSetDevice(devs[di].id);
                CK(cudaEventCreateWithFlags(&idle_ev[di], cudaEventDisableTiming));
                idle_win[di].store(0u);
            }
        }
        if (o.expert_cache) {
            size_t maxb = 0;
            for (const Layer& l : L)
                for (const Layer::XCls& c : l.cls) maxb = std::max(maxb, c.blob);
            const int nt = std::max(1, std::min(c_inflight, std::getenv("DS4_CACHE_THREADS") ? std::atoi(std::getenv("DS4_CACHE_THREADS")) : 4));
            sw_bufs.resize((size_t) nt);
            for (SwapBufs& B : sw_bufs) {
                if (!halloc(B.a, maxb, err) || !halloc(B.b, maxb, err) || !halloc(B.h, maxb, err)) return false;
                for (Dev& d : devs) {
                    cudaSetDevice(d.id);
                    cudaStream_t cs = nullptr;
                    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
                    B.st.push_back(cs);
                }
            }
            for (SwapBufs& B : sw_bufs) cache_ths.emplace_back([this, &B] { cache_loop(B); });
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
        }
        if (par)
            for (size_t di = 0; di < devs.size(); ++di) {
                workers.push_back(std::make_unique<Worker>());
                workers.back()->start();
            }
        if (o.verbose) std::fprintf(stderr, "ds4: loaded in %.1f s\n%s", (now_ms() - t0) / 1000.0, describe().c_str());
        reset();
        return true;
    }


    // ---- the chunked prompt path
    size_t x_cap() const { return (size_t) std::max<int>(kWin, chunk + 2); }   // a window's / chunk's new source rows
    size_t chunk_hand_bytes() const {
        return (size_t) chunk * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }

    bool alloc_chunks(std::string& err) {
        chunks.resize(devs.size());
        const int D = g.dim, H = g.hc, HD = g.hd;
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            const size_t s0 = d.scratch;   // (this device's chunk buffers: d.prompt, not d.scratch)
            chunks[di] = std::make_unique<Chunk>();
            Chunk& c = *chunks[di];
            const int nm = (int) groups[(size_t) d.grp].mem.size();
            const int NHL = g.heads / head_members(nm), HQ = NHL * HD, OG = g.o_groups / head_members(nm);
            int FF = 0, ff0_ = 0;
            split128(g.ff, nm, 0, ff0_, FF);   // (the widest member's share: member 0 takes the odd blocks)
            const size_t C = (size_t) chunk;
            c.C = chunk;
            c.A = std::max(64, (int) (((size_t) 8 << 20) / (size_t) HQ) / 64 * 64);
            // (the indexer's slab: its scores rows are the context wide - about 64 MB of them)
            c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
            c.w16_elems = (size_t) 8 << 20;
            bool has_eg = false, has_first = false;
            size_t stride = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di && !l.mtp) {
                        has_eg = has_eg || l.eg >= 0;
                        has_first = has_first || l.il == 0;
                        stride = std::max(stride, align256(unit_max(member(l, m))));
                    }
            const size_t A = (size_t) c.A, AI = (size_t) c.AI, ss = (size_t) o.max_context + 64;
            bool ok = dalloc(d, c.h, C * H * D, err) && dalloc(d, c.x, C * D, err) && dalloc(d, c.acc, C * D, err) &&
                      dalloc(d, c.x16, C * D, err) && dalloc(d, c.qr, C * g.q_lora, err) &&
                      dalloc(d, c.qr16, C * g.q_lora, err) && dalloc(d, c.kvr, C * HD, err) &&
                      dalloc(d, c.ckr, C * HD, err) && dalloc(d, c.cgr, C * HD, err) &&
                      dalloc(d, c.latent, (C + 2) * HD, err) && dalloc(d, c.lat16, (C + 2) * HD, err) &&
                      dalloc(d, c.ikr, (C + 2) * g.ix_hd, err) && dalloc(d, c.ext, ((size_t) g.win + C) * HD, err) &&
                      dalloc(d, c.q, A * HQ, err) && dalloc(d, c.o, A * HQ, err) && dalloc(d, c.o16, A * HQ, err) &&
                      dalloc(d, c.oa, A * OG * g.o_lora, err) && dalloc(d, c.oa16, A * OG * g.o_lora, err) &&
                      dalloc(d, c.iq, AI * g.ix_heads * g.ix_hd, err) && dalloc(d, c.iw, C * 64, err) &&
                      dalloc(d, c.scores, AI * ss, err) && dalloc(d, c.bm, AI * bm_stride(), err) &&
                      dalloc(d, c.topk, C * topk_stride(), err) && dalloc(d, c.cand, C * cand_stride(), err) &&
                      dalloc(d, c.mix, C * 32, err) && dalloc(d, c.a_pre, C * 8, err) && dalloc(d, c.a_post, C * 8, err) &&
                      dalloc(d, c.a_comb, C * 64, err) && dalloc(d, c.f_pre, C * 8, err) &&
                      dalloc(d, c.f_post, C * 8, err) && dalloc(d, c.f_comb, C * 64, err) && dalloc(d, c.pre, C * 8, err) &&
                      dalloc(d, c.glog, C * 1024, err) && dalloc(d, c.ridx, C * kPicks, err) &&
                      dalloc(d, c.rw, C * kPicks, err) && dalloc(d, c.bounds, 2048, err) &&
                      dalloc(d, c.row_tok, C * kPicks, err) && dalloc(d, c.pos, C * kPicks, err) &&
                      dalloc(d, c.sg, C * FF, err) && dalloc(d, c.su, C * FF, err) && dalloc(d, c.sh16, C * FF, err) &&
                      dalloc(d, c.h16, C * FF, err) && dalloc(d, c.y, C * D + 32, err) && dalloc(d, c.w16, c.w16_elems, err) &&
                      halloc(c.bounds_pin, 2048, err);
            if (!ok) return false;
            if (has_first && !dalloc(d, c.emb, C * D, err)) return false;
            if (has_eg) {
                const size_t ES = 256, per = (size_t) cols * (ehd + ehd / 32);
                if (!dalloc(d, c.eg_x, ES * cols * ehd, err) || !dalloc(d, c.eg_x16, ES * cols * ehd, err) ||
                    !dalloc(d, c.eg_kv, ES * (H + 1) * D, err) || !dalloc(d, c.eg_rows, C * per, err))
                    return false;
            }
            if (stride > 0) {   // DS4_RING_MB (400) of slots in kParts batches
                const size_t mb = std::getenv("DS4_RING_MB") ? (size_t) std::atoi(std::getenv("DS4_RING_MB")) : 400;
                c.ring_stride = stride;
                c.ring_n = std::max(Chunk::kParts, (int) ((mb << 20) / stride) / Chunk::kParts * Chunk::kParts);
                if (!dalloc(d, c.ring, (size_t) c.ring_n * stride, err)) return false;
            }
            CK(cudaStreamCreateWithFlags(&c.cs, cudaStreamNonBlocking));
            for (int i = 0; i < Chunk::kParts; ++i) {
                CK(cudaEventCreateWithFlags(&c.filled[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.freed[i], cudaEventDisableTiming));
            }
            if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(c.blas, d.st) != CUBLAS_STATUS_SUCCESS) {
                err = "cuBLAS on CUDA" + std::to_string(d.id);
                return false;
            }
            for (int sl = 0; sl < 2; ++sl) {
                if (d.mi == 0 && (!hmap(c.hand[sl], chunk_hand_bytes(), err) || !hmap(c.hand_flag[sl], 32, err))) return false;
                if (has_first && !halloc(c.emb_host[sl], C * D, err)) return false;
                CK(cudaEventCreateWithFlags(&c.emb_ev[sl], cudaEventDisableTiming));
            }
            if (d.mi == 0 && !hmap(c.hand_ack, (size_t) 2 * 8 * 32, err)) return false;
            for (int sl = 0; sl < kEgSlots; ++sl) CK(cudaEventCreateWithFlags(&c.eg_ev[sl], cudaEventDisableTiming));
            CK(cudaStreamCreateWithFlags(&c.ar_aux.s2, cudaStreamNonBlocking));
            for (cudaEvent_t& e : c.ar_aux.ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            CK(cudaStreamCreateWithFlags(&c.ar_s, cudaStreamNonBlocking));
            CK(cudaStreamCreateWithFlags(&c.ar_aux2.s2, cudaStreamNonBlocking));
            for (cudaEvent_t& e : c.ar_aux2.ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&c.ar_go, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&c.ar_done, cudaEventDisableTiming));
            for (cudaEvent_t& e : c.pend_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            d.prompt += d.scratch - s0;
            d.scratch = s0;
        }
        // the prompt's Engram rows: the shared slots, the reader's IO pool (DS4_EG_READERS, 48), the devices that copy
        eg_copies = 0;
        eg_devs.clear();
        for (const Layer& l : L)
            if (!l.mtp && l.eg >= 0)
                for (int m = 0; m < l.nm; ++m) {
                    ++eg_copies;
                    if (std::find(eg_devs.begin(), eg_devs.end(), member(l, m).dev) == eg_devs.end())
                        eg_devs.push_back(member(l, m).dev);
                }
        if (eg_copies > 0) {
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            for (int sl = 0; sl < kEgSlots; ++sl)
                if (!halloc(eg_shared[sl], (size_t) chunk * pk.eg_layers * per, err)) return false;
            const char* nr = std::getenv("DS4_EG_READERS");
            eg_pool = std::make_unique<Pool>(std::max(1, nr != nullptr ? std::atoi(nr) : 48), std::vector<int>{}, 0);
        }
        return true;
    }

    // y[r][c] = x16[r] . W[:, c] (+ beta y) for r < rows: W reconstructed to FP16 in whole 128-column blocks, cuBLAS
    bool gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
              std::string& err) {
        if (rows <= 0) return true;
        const int cols = std::max(128, (int) std::min<size_t>((size_t) m.n, c.w16_elems / (size_t) m.k) / 128 * 128);
        for (int c0 = 0; c0 < m.n; c0 += cols) {
            const int nc = std::min(cols, m.n - c0);
            x3::reconstruct_rows(nc == m.n ? m : x3::view(m, c0, nc), c.w16, false, d.st);
            const float one = 1.0f;
            const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, c.w16, CUDA_R_16F,
                                                   m.k, x16, CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                err = "cublasGemmEx: status " + std::to_string((int) st);
                return false;
            }
        }
        return true;
    }
    // y[r][c] = alpha x16[r] . W[c] for F16 weights W [n][k] (the router, the indexer's weights_proj)
    bool gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows, float alpha,
                   float* y, int ldy, std::string& err) {
        const float zero = 0.0f;
        const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F, k,
                                               x16, CUDA_R_16F, ldx, &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            err = "cublasGemmEx (F16 weights): status " + std::to_string((int) st);
            return false;
        }
        return true;
    }
    // hc_mix over a chunk: the dots by an F32 GEMM, then the rms scale
    bool hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes, std::string& err) {
        const int hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, fn, hcd, h, hcd, &zero, mixes, 32) !=
            CUBLAS_STATUS_SUCCESS) {
            err = "cublasSgemm (mHC mixes)";
            return false;
        }
        hc_mix_scale(h, hcd, n_mix, T, g.eps, mixes, d.st);
        return true;
    }

    // expert e of l (on the CPU lane: its NH column parts) -> member m's VRAM copy at dst, in the GPU layout (the full
    // blob, or a group member's part), by DMA from the pinned parts on the copy stream (in place: from `src`, its file
    // region read into pinned staging)
    bool stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                       const uint8_t* src = nullptr) {
        const auto& parts = l.hh[(size_t) e];
        const Layer::XCls& C = xc(l, e);   // (its class's layout: the prompt's sets read the offsets, no header)
        const int D = g.dim, FF = g.ff;
        auto cp2 = [&](uint8_t* d0, size_t dp, const uint8_t* s0, size_t sp, size_t w, size_t h) {
            return cudaMemcpy2DAsync(d0, dp, s0, sp, w, h, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        auto cp = [&](uint8_t* d0, const uint8_t* s0, size_t n) {
            return cudaMemcpyAsync(d0, s0, n, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        bool ok = true;
        if (NH == 1) {   // one part: the whole expert in exllamav3's layout, its pieces at hl's offsets
            const uint8_t* p = src != nullptr ? src : parts[0];
            if (l.nm == 1) {
                for (int r = 0; r < 3; ++r)
                    ok = ok && cp(dst + C.lay.trellis[r], p + C.hl.trellis[r], C.piece_bytes[r][0]) &&
                         cp(dst + C.lay.suh[r], p + C.hl.suh[r], (size_t) C.hl.k[r] * 2) &&
                         cp(dst + C.lay.svh[r], p + C.hl.svh[r], (size_t) C.hl.n[r] * 2);
            } else {   // member m's part (its FFN columns [f0, f0 + fn)), cut as blob_to_half cuts a blob
                const Layer& x = member(l, m);
                const int f0 = x.ff0, fn = x.ffn;
                for (int r = 0; r < 2; ++r) {
                    const size_t tile = (size_t) 32 * C.lay.K[r], nt = (size_t) FF / 16, nth = (size_t) fn / 16;
                    ok = ok && cp2(dst + xc(x, e).hlay.trellis[r], nth * tile, p + C.hl.trellis[r] + (size_t) (f0 / 16) * tile,
                                   nt * tile, nth * tile, (size_t) D / 16) &&
                         cp(dst + xc(x, e).hlay.suh[r], p + C.hl.suh[r], (size_t) D * 2) &&
                         cp(dst + xc(x, e).hlay.svh[r], p + C.hl.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
                }
                const size_t row = (size_t) (D / 16) * 32 * C.lay.K[2];
                ok = ok && cp(dst + xc(x, e).hlay.trellis[2], p + C.hl.trellis[2] + (size_t) (f0 / 16) * row,
                              (size_t) (fn / 16) * row) &&
                     cp(dst + xc(x, e).hlay.suh[2], p + C.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2) &&
                     cp(dst + xc(x, e).hlay.svh[2], p + C.hl.svh[2], (size_t) D * 2);
            }
        } else if (l.nm == 1) {   // the full blob: each matrix's columns from the two parts side by side
            for (int r = 0; r < 3; ++r) {
                const size_t tile = (size_t) 32 * C.lay.K[r];
                const int k = C.hl.k[r], nh = C.hl.n[r];   // part: k x nh
                for (int X = 0; X < 2; ++X) {
                    ok = ok && cp2(dst + C.lay.trellis[r] + (size_t) X * (nh / 16) * tile, (size_t) (2 * nh / 16) * tile,
                                   parts[(size_t) X] + C.hl.trellis[r], (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile,
                                   (size_t) (k / 16));
                    ok = ok && cp(dst + C.lay.svh[r] + (size_t) X * nh * 2, parts[(size_t) X] + C.hl.svh[r], (size_t) nh * 2);
                }
                ok = ok && cp(dst + C.lay.suh[r], parts[0] + C.hl.suh[r], (size_t) k * 2);
            }
        } else {   // member m (FFN columns [f0, f0 + fn)): gate / up from the parts holding them; down rows [f0, ..) of
                   // both parts' columns
            const Layer& x = member(l, m);
            const int f0 = x.ff0, fn = x.ffn;
            for (int r = 0; r < 2; ++r) {
                const size_t tile = (size_t) 32 * C.lay.K[r];
                const int hn = C.hl.n[r];   // part X: columns [X hn, (X + 1) hn)
                for (int X = 0; X < 2; ++X) {
                    const int a = std::max(f0, X * hn), b = std::min(f0 + fn, (X + 1) * hn);
                    if (a >= b) continue;
                    ok = ok && cp2(dst + xc(x, e).hlay.trellis[r] + (size_t) ((a - f0) / 16) * tile, (size_t) (fn / 16) * tile,
                                   parts[(size_t) X] + C.hl.trellis[r] + (size_t) ((a - X * hn) / 16) * tile,
                                   (size_t) (hn / 16) * tile, (size_t) ((b - a) / 16) * tile, (size_t) (D / 16)) &&
                         cp(dst + xc(x, e).hlay.svh[r] + (size_t) (a - f0) * 2, parts[(size_t) X] + C.hl.svh[r] + (size_t) (a - X * hn) * 2,
                            (size_t) (b - a) * 2);
                }
                ok = ok && cp(dst + xc(x, e).hlay.suh[r], parts[0] + C.hl.suh[r], (size_t) D * 2);
            }
            const size_t tile = (size_t) 32 * C.lay.K[2];
            const int nh = D / 2;
            for (int X = 0; X < 2; ++X) {
                ok = ok && cp2(dst + xc(x, e).hlay.trellis[2] + (size_t) X * (nh / 16) * tile, (size_t) (D / 16) * tile,
                               parts[(size_t) X] + C.hl.trellis[2] + (size_t) (f0 / 16) * (nh / 16) * tile,
                               (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile, (size_t) (fn / 16));
                ok = ok && cp(dst + xc(x, e).hlay.svh[2] + (size_t) X * nh * 2, parts[(size_t) X] + C.hl.svh[2], (size_t) nh * 2);
            }
            ok = ok && cp(dst + xc(x, e).hlay.suh[2], parts[0] + C.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
        return ok;
    }

    // a device's streamed experts: batches of ring_n / kParts of each of its layers' CPU-lane experts, chunk by chunk
    // in layer order; the copies run at most kParts batches ahead of the compute
    struct Batch {
        int k = 0, il = 0, j0 = 0, n = 0;   // chunk, layer, its host_list range
    };
    struct Streamer {
        std::vector<Batch> b;
        size_t cp = 0, kp = 0;        // the next batch to copy, to compute
        std::atomic<size_t> done{0};  // cp, for read_ahead's thread
    };
    // in place (file_experts): the batches' file regions read ahead of their copies - POSIX_FADV_WILLNEED, the kernel
    // reading them in the background in large requests, where the copies alone fault the pages in a few at a time -
    // at most DS4_READAHEAD_MB (2048) beyond the batch being copied; each region by its layer's first member alone
    void read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const {
        static const double cap = (std::getenv("DS4_READAHEAD_MB") ? std::atof(std::getenv("DS4_READAHEAD_MB")) : 2048.0) *
                                  1048576.0;
        std::deque<std::pair<size_t, double>> issued;   // (batch, bytes) not yet copied
        double ahead = 0;
        for (size_t q = 0; q < S.b.size() && !stop.load(std::memory_order_relaxed);) {
            const size_t done = S.done.load(std::memory_order_relaxed);
            while (!issued.empty() && issued.front().first < done) {
                ahead -= issued.front().second;
                issued.pop_front();
            }
            if (ahead > cap) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            const Batch& bt = S.b[q];
            const Layer& l0 = L[(size_t) bt.il];
            double nb = 0;
            if (l0.dev == di)
                for (int i = 0; i < bt.n; ++i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    posix_fadvise(pk.fds[(size_t) l0.region_file[(size_t) e]], (off_t) l0.region_off[(size_t) e],
                                  (off_t) l0.region_bytes[(size_t) e], POSIX_FADV_WILLNEED);
                    nb += (double) l0.region_bytes[(size_t) e];
                }
            issued.push_back({q++, nb});
            ahead += nb;
        }
    }
    bool copy_ahead(Dev& d, Chunk& c, Streamer& S, int di, std::string& err) {
        const int P = Chunk::kParts;
        while (S.cp < S.b.size() && S.cp < S.kp + P) {
            const Batch& bt = S.b[S.cp];
            const int half = (int) (S.cp % P);
            const Layer& l0 = L[(size_t) bt.il];
            int mi = 0;
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di) mi = m;
            if (S.cp >= (size_t) P) CK(cudaStreamWaitEvent(c.cs, c.freed[half], 0));
            uint8_t* sb = nullptr;   // in place: the batch's regions in the part's staging slots
            if (c.stage != nullptr) {
                if (S.cp >= (size_t) P) CK(cudaEventSynchronize(c.filled[half]));   // (its last batch's copies out)
                sb = c.stage + (size_t) half * (c.ring_n / P) * c.stage_stride;
                std::atomic<bool> bad{false};
                c.readers->run(bt.n, [&](int i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    if (!pk.read_at(l0.region_file[(size_t) e], l0.region_off[(size_t) e], l0.region_bytes[(size_t) e],
                                    sb + (size_t) i * c.stage_stride))
                        bad.store(true);
                });
                if (bad.load()) {
                    err = "reading layer " + std::to_string(l0.il) + "'s experts from the model's files failed";
                    return false;
                }
            }
            for (int i = 0; i < bt.n; ++i) {
                uint8_t* dst = c.ring + ((size_t) half * (c.ring_n / P) + i) * c.ring_stride;
                if (!stream_expert(l0, mi, l0.host_list[(size_t) (bt.j0 + i)], dst, c.cs, err,
                                   sb != nullptr ? sb + (size_t) i * c.stage_stride : nullptr))
                    return false;
                c.stream_bytes += (double) unit(member(l0, mi), l0.host_list[(size_t) (bt.j0 + i)]);
            }
            CK(cudaEventRecord(c.filled[half], c.cs));
            ++S.cp;
            S.done.store(S.cp, std::memory_order_relaxed);
        }
        return true;
    }

    // the experts [i0, i1) of a sorted set (bounds b on the host, d_bounds on the device) over the chunk's rows, in
    // pieces of at most C rows, each added into acc in pick order: the VRAM slots (ring false: slot i's blob), or the
    // CPU lane's experts streamed to the ring (blobs at wbase + (i - wb0) * stride; i - n_slots their place in L0's
    // host_list) - a mixed layer's class by class (the slots go class by class, host_list too), each its own layout
    bool run_set(Dev& d, Chunk& c, const Layer& L0, const Layer& l, bool ring, const uint8_t* wbase, int wb0,
                 size_t stride, const int* d_bounds, const int* b, int i0, int i1, int T, std::string& err) {
        while (i0 < i1) {
            int ci = 0, c1 = i1;
            if (mixed(l)) {
                if (ring) {
                    ci = L0.ecls[(size_t) L0.host_list[(size_t) (i0 - L0.n_slots)]];
                    c1 = i0 + 1;
                    while (c1 < i1 && L0.ecls[(size_t) L0.host_list[(size_t) (c1 - L0.n_slots)]] == ci) ++c1;
                } else {
                    while (!(i0 >= l.cls[(size_t) ci].slot0 && i0 < l.cls[(size_t) ci].slot0 + l.cls[(size_t) ci].nsl)) ++ci;
                    c1 = std::min(i1, l.cls[(size_t) ci].slot0 + l.cls[(size_t) ci].nsl);
                }
            }
            const Layer::XCls& C = l.cls[(size_t) ci];
            if (!run_set_one(d, c, l, C, ring ? wbase + (size_t) (i0 - wb0) * stride : slot_ptr(l, i0), i0,
                             ring ? stride : cunit(l, C), d_bounds, b, i0, c1, T, err))
                return false;
            i0 = c1;
        }
        return true;
    }
    // [i0, i1) of one class C, blobs at wbase + (i - wb0) * stride
    bool run_set_one(Dev& d, Chunk& c, const Layer& l, const Layer::XCls& C, const uint8_t* wbase, int wb0,
                     size_t stride, const int* d_bounds, const int* b, int i0, int i1, int T, std::string& err) {
        const int D = g.dim, FF = l.ffn;
        while (i0 < i1) {
            int j = i0 + 1;
            while (j < i1 && b[j + 1] - b[i0] <= c.C) ++j;
            const int a = b[i0], nrows = b[j] - a;
            int max_rows = 0;
            for (int i = i0; i < j; ++i) max_rows = std::max(max_rows, b[i + 1] - b[i]);
            if (nrows > 0) {
                x3::SetArgs sa;
                sa.wbase = wbase + (size_t) (i0 - wb0) * stride;   // (wbase: set member wb0's blob)
                sa.stride = stride;
                sa.lay = l.nm > 1 ? C.hlay : C.lay;
                sa.lay.hdr = 0;   // (one layout: the offsets)
                sa.n_exp = j - i0;
                sa.bounds = d_bounds + i0;
                sa.row0 = a;
                sa.max_rows = max_rows;
                sa.n_embd = D;
                sa.n_ff = FF;
                sa.limit = g.lim;
                x3::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                x3::set_down(sa, c.h16, FF, c.y, D, d.st);
                moe_combine(c.acc, c.y, c.pos, c.rw, T, l.k_exp, a, a + nrows, D, d.st);
            }
            i0 = j;
        }
        (void) err;
        return true;
    }

    bool load_dense(int di, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        size_t* acct = &d.weights;
        for (Layer& l0 : L)
          for (int mi = 0; mi < l0.nm; ++mi) {
            Layer& l = member(l0, mi);
            if (l.dev != di) continue;
            // (a group's member: its share of the heads (wq_b columns, the sinks), of wo_a's groups and wo_b's rows,
            // of the shared expert's FFN; the rest whole)
            const int nm = l0.nm, na = g.heads / std::max(1, l.ahn), ha = l.ah0 / std::max(1, l.ahn);
            const std::string& P = l.name;
            const std::string A = P + "attn.";
            // (the member's shares in sp; what d.weights gains besides them every member holds whole: tp_w)
            const size_t w0 = d.weights;
            size_t sp = 0;
            bool ok = mat(d, A + "wq_a", l.wq_a, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wq_b", l.wq_b, err, &sp, ha, na, true)) &&
                      mat(d, A + "wkv", l.wkv, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wo_b", l.wo_b, err, &sp, ha, na, false)) &&
                      f32(d, A + "q_norm.weight", l.q_norm, err, acct, (size_t) g.q_lora) &&
                      f32(d, A + "kv_norm.weight", l.kv_norm, err, acct, (size_t) g.hd) &&
                      f32(d, A + "attn_sink", l.sink, err, acct, (size_t) g.heads) &&
                      f32(d, P + "attn_norm.weight", l.attn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "ffn_norm.weight", l.ffn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "hc_attn_fn", l.hca_fn, err, acct) && f32(d, P + "hc_attn_scale", l.hca_scale, err, acct) &&
                      f32(d, P + "hc_attn_base", l.hca_base, err, acct) && f32(d, P + "hc_ffn_fn", l.hcf_fn, err, acct) &&
                      f32(d, P + "hc_ffn_scale", l.hcf_scale, err, acct) &&
                      f32(d, P + "hc_ffn_base", l.hcf_base, err, acct) &&
                      f16raw(d, P + "ffn.gate.weight", l.gate_w, err, acct) &&
                      f32(d, P + "ffn.gate.bias", l.gate_bias, err, acct, (size_t) l.n_exp) &&
                      (l.mtp || img_tok < 0 || pk.get(P + "ffn.gate.bias_vl") == nullptr ||
                       f32(d, P + "ffn.gate.bias_vl", l.gate_bias_vl, err, acct, (size_t) l.n_exp)) &&
                      mat(d, P + "ffn.shared_experts.w1", l.s1, err, &sp, mi, nm, true, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w2", l.s2, err, &sp, mi, nm, false, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w3", l.s3, err, &sp, mi, nm, true, l.il % nm);
            if (!ok) return false;
            l.sink += (size_t) l.ah0;
            for (int gi = 0; gi < l.ogn; ++gi)
                if (!wo_a(d, A, l.og0 + gi, l.wo_a[gi], err, &sp)) return false;
            if (l.kv_src) {
                const std::string C = A + "compressor.";
                if (!mat(d, C + "wkv", l.c_wkv, err, acct) || !f32(d, C + "norm.weight", l.c_norm, err, acct)) return false;
                if (l.ratio > 1 && !mat(d, C + "wgate", l.c_wgate, err, acct)) return false;
                if (l.ratio > 1 && (!dalloc(d, l.c_kv, (size_t) l.ratio * g.hd, err) ||
                                    !dalloc(d, l.c_score, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
            if (l.ix_src) {
                const std::string I = A + "indexer.";
                if (!mat(d, I + "wq_b", l.i_wq_b, err, acct) || !f16raw(d, I + "weights_proj.weight", l.i_wproj, err, acct))
                    return false;
                if (l.owns_k && (!mat(d, I + "wk", l.i_wk, err, acct) || !f32(d, I + "k_norm.weight", l.ik_norm, err, acct)))
                    return false;
            }
            if (l.eg >= 0) {
                if (!mat(d, P + "engram.wkv", l.eg_wkv, err, acct)) return false;
                float *qw = nullptr, *kw = nullptr;
                if (!f32(d, P + "engram.q_weight", qw, err, acct) || !f32(d, P + "engram.k_weight", kw, err, acct))
                    return false;
                std::vector<float> a((size_t) g.hc * g.dim), b(a.size());
                CK(cudaMemcpy(a.data(), qw, a.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b.data(), kw, b.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];   // only ever used as a product
                CK(cudaMemcpy(qw, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
                l.eg_w = qw;
            }
            if (!dalloc(d, l.win, (size_t) g.win * g.hd, err, &d.caches)) return false;
            if (g.n_mtp > 0 && !l.mtp) {
                if (!dalloc(d, l.ring_bak, (size_t) kWin * g.hd, err)) return false;
                if (l.kv_src && l.ratio > 1 && (!dalloc(d, l.c_snap, (size_t) 2 * l.ratio * g.hd, err) ||
                                                !dalloc(d, l.c_rows, (size_t) 2 * kWin * g.hd, err)))
                    return false;
            }
            if (!dalloc(d, l.vram_tab, (size_t) l.n_exp, err, acct)) return false;
            if (nm > 1) d.tp_w += (d.weights - w0) / (size_t) nm * (size_t) (nm - 1);
            d.weights += sp;
          }
        if (d.grp == devs[(size_t) head_dev].grp) {   // the head: each member of its group a share of the columns
            const int hn = head_nm();
            if (!mat(d, "head", d.head, err, acct, d.mi, hn, true) || !f32(d, "norm.weight", d.norm_w, err, acct, (size_t) g.dim))
                return false;
            if (g.n_mtp > 0 && !f32(d, "mtp." + std::to_string(g.n_mtp - 1) + ".norm.weight", d.d_norm, err, acct, (size_t) g.dim))
                return false;
            if (!dalloc(d, d.logits_part, (size_t) kWin * ((g.vocab / 128 + hn - 1) / hn * 128), err)) return false;
            if (g.n_mtp > 0 && hn > 1) {   // the markov steps over the group (mk_split): this member's rows of the head
                const Tensor* th = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.head.weight");
                const int R = g.d_rank;
                split128(g.vocab, hn, d.mi, d.mk_v0, d.mk_vn);
                if (th == nullptr || (th->dtype != "F16" && th->dtype != "BF16") || th->bytes != (uint64_t) g.vocab * R * 2) {
                    err = "the markov head: not an F16 (or BF16) [vocab][rank] matrix";
                    return false;
                }
                std::vector<uint16_t> rows((size_t) d.mk_vn * R);
                if (!pk.read_at(th->file, th->off + (uint64_t) d.mk_v0 * R * 2, rows.size() * 2, rows.data())) {
                    err = "the markov head: read failed";
                    return false;
                }
                if (th->dtype == "BF16")
                    for (uint16_t& x : rows) x = f32_to_f16(bf16_to_f32(x));
                if (!dalloc(d, d.mk_share, rows.size(), err, acct) || !dalloc(d, d.mk_erow, (size_t) R, err) ||
                    !dalloc(d, d.mk_ints, (size_t) kWin + 2, err))
                    return false;
                CK(cudaMemcpy(d.mk_share, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
            }
        }
        if (di == head_dev) {
            head = d.head;
            norm_w = d.norm_w;
            d_norm = d.d_norm;
            if (g.n_mtp > 0) {
                const std::string Z = "mtp." + std::to_string(g.n_mtp - 1) + ".";
                if (!f32(d, Z + "confidence_head.proj.weight", d_conf, err, acct, (size_t) (g.dim + g.d_rank)))
                    return false;
                const Tensor* te = pk.get(Z + "markov_head.embed.weight");
                const Tensor* th = pk.get(Z + "markov_head.head.weight");
                if (te == nullptr || th == nullptr || te->dtype != "BF16" || (th->dtype != "F16" && th->dtype != "BF16")) {
                    err = Z + "markov_head: not the BF16 embed and F16 (or BF16) head expected";
                    return false;
                }
                if (!dalloc(d, d_mk_embed, te->bytes / 2, err, acct) || !upload(d, *te, d_mk_embed, err) ||
                    !f16raw(d, Z + "markov_head.head.weight", d_mk_head, err, acct))
                    return false;
            }
        }
        if (g.n_mtp > 0 && di == d_dev) {
            if (!mat(d, "mtp.0.main_proj", d_main_proj, err, acct) ||
                !f32(d, "mtp.0.main_norm.weight", d_main_norm, err, acct, (size_t) g.dim))
                return false;
        }
        return true;
    }

    void set_vm(Layer& l, int e, int slot) {
        uint8_t* b = slot_ptr(l, slot);
        const Layer::XCls& C = xc(l, e);
        for (int r = 0; r < 3; ++r) {
            x3::Mat& m = l.vm[(size_t) e][(size_t) r];
            m.trellis = (const uint32_t*) (b + C.lay.trellis[r]);
            m.suh = (const uint16_t*) (b + C.lay.suh[r]);
            m.svh = (const uint16_t*) (b + C.lay.svh[r]);
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
            m.K = C.lay.K[r];
            m.cb = 2;
            m.ld = 0;
        }
    }
    // an expert's VRAM blob -> its CPU-lane part X, and its NH parts -> the blob (the cache's swaps)
    void blob_to_part(const Layer& l, int e, const uint8_t* blob, int X, uint8_t* dst) const {
        const Layer::XCls& C = xc(l, e);
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * C.lay.K[r];
            const int k = C.hl.k[r], nt = C.hl.n[r] * NH / 16, nth = C.hl.n[r] / 16;
            const uint8_t* tr = blob + C.lay.trellis[r];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + C.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + C.hl.suh[r], blob + C.lay.suh[r], (size_t) k * 2);
            std::memcpy(dst + C.hl.svh[r], blob + C.lay.svh[r] + (size_t) X * C.hl.n[r] * 2, (size_t) C.hl.n[r] * 2);
        }
    }
    void parts_to_blob(const Layer& l, int e, const std::array<uint8_t*, 4>& parts, uint8_t* blob) const {
        const Layer::XCls& C = xc(l, e);
        put_hdr(C.lay, blob);
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * C.lay.K[r];
            const int k = C.hl.k[r], nt = C.hl.n[r] * NH / 16, nth = C.hl.n[r] / 16;
            uint8_t* tr = blob + C.lay.trellis[r];
            for (int X = 0; X < NH; ++X)
                for (int sl = 0; sl < k / 16; ++sl)
                    std::memcpy(tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                                parts[(size_t) X] + C.hl.trellis[r] + (size_t) sl * nth * tile, (size_t) nth * tile);
            std::memcpy(blob + C.lay.suh[r], parts[0] + C.hl.suh[r], (size_t) k * 2);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + C.lay.svh[r] + (size_t) X * C.hl.n[r] * 2, parts[(size_t) X] + C.hl.svh[r],
                            (size_t) C.hl.n[r] * 2);
        }
    }

    // an expert's blob -> member m's part (gate / up columns [ff0, ff0 + ffn), down rows of the same), and back
    void blob_to_half(const Layer& l, int e, const uint8_t* blob, int m, uint8_t* dst) const {
        const Layer& x = member(l, m);
        const Layer::XCls &C = xc(l, e), &XC = xc(x, e);
        put_hdr(XC.hlay, dst);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * C.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(dst + XC.hlay.trellis[r] + sl * nth * tile,
                            blob + C.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile, nth * tile);
            std::memcpy(dst + XC.hlay.suh[r], blob + C.lay.suh[r], (size_t) g.dim * 2);
            std::memcpy(dst + XC.hlay.svh[r], blob + C.lay.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * C.lay.K[2];
        std::memcpy(dst + XC.hlay.trellis[2], blob + C.lay.trellis[2] + (size_t) (f0 / 16) * row, (size_t) (fn / 16) * row);
        std::memcpy(dst + XC.hlay.suh[2], blob + C.lay.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        std::memcpy(dst + XC.hlay.svh[2], blob + C.lay.svh[2], (size_t) g.dim * 2);
    }
    void half_to_blob(const Layer& l, int e, const uint8_t* src, int m, uint8_t* blob) const {
        const Layer& x = member(l, m);
        const Layer::XCls &C = xc(l, e), &XC = xc(x, e);
        if (m == 0) put_hdr(C.lay, blob);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * C.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(blob + C.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile,
                            src + XC.hlay.trellis[r] + sl * nth * tile, nth * tile);
            if (m == 0) std::memcpy(blob + C.lay.suh[r], src + XC.hlay.suh[r], (size_t) g.dim * 2);
            std::memcpy(blob + C.lay.svh[r] + (size_t) f0 * 2, src + XC.hlay.svh[r], (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * C.lay.K[2];
        std::memcpy(blob + C.lay.trellis[2] + (size_t) (f0 / 16) * row, src + XC.hlay.trellis[2], (size_t) (fn / 16) * row);
        std::memcpy(blob + C.lay.suh[2] + (size_t) f0 * 2, src + XC.hlay.suh[2], (size_t) fn * 2);
        if (m == 0) std::memcpy(blob + C.lay.svh[2], src + XC.hlay.svh[2], (size_t) g.dim * 2);
    }

    bool place_experts(std::string& err) {
        // per group: what its tightest member has left (each member holds its part of every VRAM expert of the
        // group's layers, its unit bytes a slot - counted at the members' mean: an uneven split's odd blocks go round
        // the members layer by layer, so each member's total comes to about the mean's)
        auto ucount = [](const Layer& l, int e) {
            double s = 0;
            for (int m = 0; m < l.nm; ++m) s += (double) unit(member(l, m), e);
            return s / l.nm;
        };
        // each layer's experts hottest first (the profile, then by index) and what the i-th of them takes a member (a
        // mixed layer's experts differ; the others' all ucount(l, 0)): take() the hottest of them as room goes, csum()
        // the first n's - an EXL3 pack's layers by the same arithmetic as ever
        std::vector<std::vector<int>> hot(L.size());
        std::vector<std::vector<double>> cst(L.size());
        for (const Layer& l : L) {
            std::vector<int>& o_ = hot[(size_t) l.il];
            o_.resize((size_t) l.n_exp);
            std::iota(o_.begin(), o_.end(), 0);
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            std::stable_sort(o_.begin(), o_.end(), [&](int a, int b) { return p[a] > p[b]; });
            for (int i = 0; i < l.n_exp; ++i) cst[(size_t) l.il].push_back(ucount(l, o_[(size_t) i]));
        }
        auto take = [&](Layer& l, double& room, int upto) {
            if (!mixed(l)) {
                l.n_vram = std::min(upto, (int) std::floor(room / ucount(l, 0)));
                room -= (double) l.n_vram * ucount(l, 0);
                return;
            }
            l.n_vram = 0;
            while (l.n_vram < upto && cst[(size_t) l.il][(size_t) l.n_vram] <= room) room -= cst[(size_t) l.il][(size_t) l.n_vram++];
        };
        auto csum = [&](const Layer& l, int n) {
            if (!mixed(l)) return (double) ucount(l, 0) * n;
            double b = 0;
            for (int i = 0; i < n; ++i) b += cst[(size_t) l.il][(size_t) i];
            return b;
        };
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            std::vector<double> room;   // each member's
            for (int m : groups[gi].mem) {
                cudaSetDevice(devs[(size_t) m].id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                room.push_back(std::max(0.0, ((double) fr - (double) ((size_t) o.reserve_mib << 20))) * o.vram_experts);
            }
            // the drafter's experts first (every draft uses its blocks) - its blocks are the lead's alone, so they take
            // the lead's room; the trunk's layers then what the tightest member has left
            double others = -1;   // the room of the group's tightest member but the lead (-1: the lead is alone)
            for (size_t m = 1; m < room.size(); ++m) others = others < 0 ? room[m] : std::min(others, room[m]);
            bool dseen = false;
            for (const Layer& l : L)
                if (l.grp == (int) gi && l.mtp)
                    for (int e = 0; e < l.n_exp; ++e) dseen = dseen || prof[(size_t) l.il * g.n_exp + e] > 0;
            if (others < 0 || !dseen || std::getenv("DS4_PLACE_UNIFORM") != nullptr) {   // all of them, as room goes
                for (Layer& l : L)
                    if (l.grp == (int) gi && l.mtp) take(l, room[0], l.n_exp);
            } else {
                // by the profile: each block's experts that take 99% of its routes always, the others only while the
                // lead keeps the room of the next tightest member (past it, each drafter expert's bytes on the lead
                // are as many trunk experts lost on every member: a trunk expert in VRAM takes its part on each)
                std::vector<std::vector<double>> sh(L.size());
                using Top = std::pair<double, int>;   // (the next expert's share a byte, layer)
                std::priority_queue<Top> pq;
                for (Layer& l : L) {
                    if (l.grp != (int) gi || !l.mtp) continue;
                    const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                    double sum = 0;
                    for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                    std::vector<double>& s = sh[(size_t) l.il];
                    for (int e = 0; e < l.n_exp; ++e) s.push_back(sum > 0 ? p[e] / sum : 1.0 / l.n_exp);
                    std::sort(s.begin(), s.end(), std::greater<double>());
                    int keep = 0;
                    for (double c = 0; keep < l.n_exp && c < 0.99; ++keep) c += s[(size_t) keep];
                    take(l, room[0], keep);
                    if (l.n_vram < l.n_exp) pq.push({s[(size_t) l.n_vram] / cst[(size_t) l.il][(size_t) l.n_vram], l.il});
                }
                while (!pq.empty()) {
                    const int il = pq.top().second;
                    pq.pop();
                    Layer& l = L[(size_t) il];
                    const double u = cst[(size_t) il][(size_t) l.n_vram];
                    if (room[0] - u < others) continue;
                    room[0] -= u;
                    if (++l.n_vram < l.n_exp)
                        pq.push({sh[(size_t) il][(size_t) l.n_vram] / cst[(size_t) il][(size_t) l.n_vram], il});
                }
            }
            const double budget = *std::min_element(room.begin(), room.end());
            double want = 0, seen = 0;
            for (const Layer& l : L)
                if (l.grp == (int) gi && !l.mtp) {
                    want += csum(l, l.n_exp);
                    for (int e = 0; e < l.n_exp; ++e) seen += prof[(size_t) l.il * g.n_exp + e];
                }
            if (seen > 0 && std::getenv("DS4_PLACE_UNIFORM") == nullptr) {
                // by the profile: the room to the experts that take the largest share of their layer's routes a byte
                // (every layer routes as often; one whose routes spread wider gets more of the room) - each layer's
                // shares smoothed a tenth toward even, so a layer the profile barely saw is not left without
                std::vector<std::vector<double>> sh(L.size());
                using Top = std::pair<double, int>;   // (the next expert's share a byte, layer)
                std::priority_queue<Top> pq;
                for (Layer& l : L) {
                    if (l.grp != (int) gi || l.mtp) continue;
                    l.n_vram = 0;
                    const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                    double sum = 0;
                    for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                    std::vector<double>& s = sh[(size_t) l.il];
                    for (int e = 0; e < l.n_exp; ++e)
                        s.push_back(0.9 * (sum > 0 ? p[e] / sum : 1.0 / l.n_exp) + 0.1 / l.n_exp);
                    std::sort(s.begin(), s.end(), std::greater<double>());
                    pq.push({s[0] / cst[(size_t) l.il][0], l.il});
                }
                double used = 0;
                while (!pq.empty()) {
                    const int il = pq.top().second;
                    pq.pop();
                    Layer& l = L[(size_t) il];
                    const double u = cst[(size_t) il][(size_t) l.n_vram];
                    if (used + u > budget) continue;   // (this layer's experts no longer fit; smaller ones may)
                    used += u;
                    if (++l.n_vram < l.n_exp)
                        pq.push({sh[(size_t) il][(size_t) l.n_vram] / cst[(size_t) il][(size_t) l.n_vram], il});
                }
            } else {   // as even as the room goes: each layer the same share of its experts
                const double frac = want > 0 ? std::min(1.0, budget / want) : 0.0;
                for (Layer& l : L) {
                    if (l.grp != (int) gi || l.mtp) continue;
                    l.n_vram = std::min(l.n_exp, (int) std::floor(frac * l.n_exp));
                    // never more than fits: the floor of each layer's share can still round up the total
                    while (l.n_vram > 0 && csum(l, l.n_vram) > budget) --l.n_vram;
                }
                double used = 0;
                for (Layer& l : L)
                    if (l.grp == (int) gi && !l.mtp) used += csum(l, l.n_vram);
                // what the floors left: whole experts, layer by layer
                for (Layer& l : L) {
                    if (l.grp != (int) gi || l.mtp) continue;
                    while (l.n_vram < l.n_exp && used + cst[(size_t) l.il][(size_t) l.n_vram] <= budget) {
                        used += cst[(size_t) l.il][(size_t) l.n_vram];
                        ++l.n_vram;
                    }
                }
            }
            if (o.verbose) {
                int nv = 0, ne = 0;
                for (const Layer& l : L)
                    if (l.grp == (int) gi) nv += l.n_vram, ne += l.n_exp;
                std::fprintf(stderr, "ds4: group %zu (%zu GPUs): %.2f GB a member for experts, %d of %d in VRAM\n", gi,
                             groups[gi].mem.size(), budget / 1e9, nv, ne);
                for (const Layer& l : L)
                    if (l.grp == (int) gi && std::getenv("DS4_PLACE_DEBUG"))
                        std::fprintf(stderr, "  layer %d%s: blob %.2f MB, mean unit %.2f MB, member 0 %.2f MB, %d of %d\n",
                                     l.il, l.mtp ? " (mtp)" : "", all_blobs(l) / 1e6 / l.n_exp, csum(l, l.n_exp) / 1e6 / l.n_exp,
                                     unit(member(l, 0), 0) / 1e6,
                                     l.n_vram, l.n_exp);
            }
        }
        for (Layer& l : L) {
            // the hottest experts of the profile first, then by index
            std::vector<int> order((size_t) l.n_exp);
            std::iota(order.begin(), order.end(), 0);
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
            l.slot.assign((size_t) l.n_exp, -1);
            l.host.assign((size_t) l.n_exp, nullptr);
            // a layer with experts on both sides keeps one VRAM slot free for the cache's swaps
            l.n_slots = std::min(l.n_vram, l.n_exp);
            if (o.expert_cache && l.n_slots >= 2 && l.n_slots < l.n_exp) {   // (every expert fits: nothing to swap)
                const int r = recent_n > 0 && !l.mtp && !file_experts && l.n_slots - 1 - recent_n >= 1 ? recent_n : 0;
                l.n_vram = l.n_slots - 1 - r;   // (the recent slots: [n_vram, n_vram + r), from the profile's coldest)
                l.spare_vram = l.n_slots - 1;
                l.rc_base = l.n_vram;
                l.rc_n = r;
            } else {
                l.n_vram = l.n_slots;
                l.spare_vram = -1;
            }
            if (mixed(l)) {   // the slots class by class (each class's in the hotness order), each member its own places
                for (Layer::XCls& c : l.cls) c.nsl = 0;
                for (int i = 0; i < l.n_vram; ++i) ++xc(l, order[(size_t) i]).nsl;
                int s0 = 0;
                for (Layer::XCls& c : l.cls) c.slot0 = s0, s0 += c.nsl;
                std::vector<int> at(l.cls.size(), 0);
                for (int i = 0; i < l.n_vram; ++i) {
                    const int e = order[(size_t) i], c = l.ecls[(size_t) e];
                    l.slot[(size_t) e] = l.cls[(size_t) c].slot0 + at[(size_t) c]++;
                }
            } else {
                for (int i = 0; i < l.n_vram; ++i) l.slot[(size_t) order[(size_t) i]] = i;
                l.cls[0].slot0 = 0;
                l.cls[0].nsl = l.n_slots;
            }
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                size_t off = 0;
                for (size_t c = 0; c < l.cls.size(); ++c) {
                    x.cls[c].slot0 = l.cls[c].slot0;
                    x.cls[c].nsl = l.cls[c].nsl;
                    x.cls[c].voff = off;
                    off += (size_t) x.cls[c].nsl * cunit(x, x.cls[c]);
                }
            }
            for (int m = 0; m < l.nm && l.n_slots > 0; ++m) {
                Layer& x = member(l, m);
                Dev& d = devs[(size_t) x.dev];
                cudaSetDevice(d.id);
                void* v = nullptr;
                const size_t vb = slots_bytes(x);
                if (cudaMalloc(&v, vb) != cudaSuccess) {
                    err = "cudaMalloc of layer " + std::to_string(l.il) + "'s " + std::to_string(l.n_slots) +
                          " experts on CUDA" + std::to_string(d.id) + " failed";
                    return false;
                }
                d.allocs.push_back(v);
                d.experts += vb;
                x.vram_base = (uint8_t*) v;
            }
            l.heat.assign((size_t) l.n_exp, 0.0f);
            {   // the profile as the starting heat: its counts scaled to ~512 hits a layer
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                const double sc = sum > 0 ? 512.0 / sum : 0.0;
                for (int e = 0; e < l.n_exp; ++e) l.heat[(size_t) e] = (float) (p[e] * sc);
            }
            const int n_host = l.n_exp - l.n_vram;
            for (Layer::XCls& C : l.cls) {   // the CPU lane's parts of an expert (in place: the whole expert, its pieces
                size_t off = 0;              // where the file has them), by class
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : g.ff, nf = r < 2 ? g.ff : g.dim, n = nf / NH;
                    C.hl.k[r] = k;
                    C.hl.n[r] = n;
                    C.hl.trellis[r] = file_experts ? C.piece_src[r][0] : off;
                    off += align256((size_t) (k / 16) * (n / 16) * 32 * C.lay.K[r]);
                    C.hl.suh[r] = file_experts ? C.piece_src[r][1] : off;
                    off += align256((size_t) k * 2);
                    C.hl.svh[r] = file_experts ? C.piece_src[r][2] : off;
                    off += align256((size_t) n * 2);
                }
                C.hl.bytes = off;
            }
            l.hh.assign((size_t) l.n_exp, {nullptr, nullptr, nullptr, nullptr});
            if (file_experts) {   // every expert's file mapped (the cache's swaps hand VRAM experts back to the files)
                for (int e = 0; e < l.n_exp; ++e)
                    if (pk.map(l.region_file[(size_t) e]) == nullptr) {
                        err = "mapping " + pk.files[(size_t) l.region_file[(size_t) e]] + " failed";
                        return false;
                    }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) {
                        l.hh[(size_t) e][0] = file_part(l, e);
                        l.host[(size_t) e] = l.hh[(size_t) e][0];
                    }
            } else if (n_host > 0) {
                size_t bytes = l.spare_vram >= 0 ? l.cls[0].hl.bytes : 0;   // (+ the cache's spare: one class)
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) bytes += xc(l, e).hl.bytes;
                for (int X = 0; X < NH; ++X) {
                    void* m = node_map(bytes, X, NH);
                    if (m == nullptr) {
                        err = "host memory for layer " + std::to_string(l.il) + "'s " + std::to_string(n_host) +
                              " CPU experts (" + std::to_string(bytes >> 20) + " MiB on node " + std::to_string(X) + ")";
                        return false;
                    }
                    host_maps.emplace_back(m, bytes);
                    size_t at = 0;
                    for (int e = 0; e < l.n_exp; ++e)
                        if (l.slot[(size_t) e] < 0) {
                            l.hh[(size_t) e][(size_t) X] = (uint8_t*) m + at;
                            at += xc(l, e).hl.bytes;
                        }
                    if (l.spare_vram >= 0) l.spare_host[(size_t) X] = (uint8_t*) m + at;
                }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = l.hh[(size_t) e][0];
            }
            l.vm.assign((size_t) l.n_exp, {});
            if (l.nm == 1)   // (the host-routed window path's views: a group's members take the device-routed path)
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0) set_vm(l, e, l.slot[(size_t) e]);
            if (!halloc(l.tab_pin, (size_t) 2 * l.nm, err)) return false;
            if (l.rc_n > 0) {
                l.rc_exp.assign((size_t) l.rc_n, -1);
                l.rc_fill.assign((size_t) l.rc_n, -1);
                l.rc_gen.assign((size_t) l.rc_n, 0u);
                l.rc_stamp.assign((size_t) l.rc_n, 0u);
                if (!halloc(l.rc_pin, (size_t) l.rc_n * l.nm + 1, err)) return false;
                l.rc_ev.assign((size_t) l.rc_n * l.nm, nullptr);
                for (int i = 0; i < l.rc_n; ++i)
                    for (int m = 0; m < l.nm; ++m) {
                        Layer& x = member(l, m);
                        l.rc_pin[(size_t) i * l.nm + m] = (unsigned long long) slot_ptr(x, l.rc_base + i);
                        CK(cudaSetDevice(devs[(size_t) x.dev].id));
                        CK(cudaEventCreateWithFlags(&l.rc_ev[(size_t) i * l.nm + m], cudaEventDisableTiming));
                    }
                l.rc_pin[(size_t) l.rc_n * l.nm] = 0ull;
            }
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0)
                        tab[(size_t) e] = (unsigned long long) slot_ptr(x, l.slot[(size_t) e]);
                cudaSetDevice(devs[(size_t) x.dev].id);
                CK(cudaMemcpy(x.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice));
            }
        }
        return true;
    }

    bool load_experts(std::string& err) {
        // every expert's region, in file order, by a few threads; VRAM ones through a pinned staging blob.  In place
        // (file_experts) the CPU lane's experts are not copied: the most-routed of them (the profile's share of their
        // layer) are read once, up to warm_bytes, so the page cache starts with them, and the VRAM ones' pages are
        // dropped from it again
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        for (const Layer& l : L)
            for (int e = 0; e < l.n_exp; ++e)
                if (!file_experts || l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        uint64_t warm = 0;
        if (file_experts) {
            std::vector<std::pair<double, Job>> cand;
            for (const Layer& l : L) {
                const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) cand.push_back({sum > 0 ? p[e] / sum : 0.0, Job{l.il, e}});
            }
            std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (const auto& c : cand) {
                const uint64_t b = L[(size_t) c.second.layer].region_bytes[(size_t) c.second.e];
                if (warm + b > warm_bytes) break;
                warm += b;
                jobs.push_back(c.second);
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.region_file[(size_t) a.e] != lb.region_file[(size_t) b.e])
                return la.region_file[(size_t) a.e] < lb.region_file[(size_t) b.e];
            return la.region_off[(size_t) a.e] < lb.region_off[(size_t) b.e];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        const int nth = 12;
        size_t maxr = 0, maxb = 0;
        for (const Layer& l : L) {
            for (const Layer::XCls& c : l.cls) maxb = std::max(maxb, c.blob);
            for (uint64_t r : l.region_bytes) maxr = std::max<size_t>(maxr, r);
        }
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&] {
                std::vector<uint8_t> region(maxr);
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                int cur_dev = -1;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    if (!pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], l.region_bytes[(size_t) e],
                                    region.data())) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "reading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    if (l.slot[(size_t) e] >= 0) {
                        repack(l, e, region.data(), stage);
                        bool up = true;
                        for (int m = 0; m < l.nm && up; ++m) {   // (a group: each member its part)
                            const Layer& x = member(l, m);
                            const Dev& d = devs[(size_t) x.dev];
                            if (cur_dev != d.id) {
                                cudaSetDevice(d.id);
                                cur_dev = d.id;
                            }
                            if (l.nm > 1) blob_to_half(l, e, stage, m, stage2);
                            up = cudaMemcpy(slot_ptr(x, l.slot[(size_t) e]), l.nm > 1 ? stage2 : stage, unit(x, e),
                                            cudaMemcpyHostToDevice) == cudaSuccess;
                        }
                        if (!up) {
                            std::lock_guard<std::mutex> lk(emu);
                            ferr = "uploading layer " + std::to_string(l.il) + " expert " + std::to_string(e);
                            failed.store(true);
                            break;
                        }
                        if (file_experts)
                            posix_fadvise(pk.fds[(size_t) l.region_file[(size_t) e]], (off_t) l.region_off[(size_t) e],
                                          (off_t) l.region_bytes[(size_t) e], POSIX_FADV_DONTNEED);
                    } else if (!file_experts) {
                        for (int X = 0; X < NH; ++X) repack_part(l, e, region.data(), X, l.hh[(size_t) e][(size_t) X]);
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.region_bytes[(size_t) e]) +
                                       (int64_t) l.region_bytes[(size_t) e];
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(),
                                     db / 1e9, db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        if (file_experts)
            std::fprintf(stderr, "ds4: experts loaded in %.1f s (%.1f GB of the CPU lane's read ahead)\n",
                         (now_ms() - t0) / 1000.0, warm / 1e9);
        return true;
    }

    std::string describe() const {
        std::ostringstream s;
        for (size_t di = 0; di < devs.size(); ++di) {
            const Dev& d = devs[di];
            int a = -1, b = -1, ev = 0, et = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di) {
                        if (a < 0) a = l.il;
                        b = l.il;
                        ev += l.n_vram;
                        et += l.n_exp;
                    }
            size_t fr = 0, tot = 0;
            cudaSetDevice(d.id);
            cudaMemGetInfo(&fr, &tot);
            const size_t gsz = groups.empty() ? 1 : groups[(size_t) d.grp].mem.size();
            char tp[64] = "";
            if (gsz > 1) std::snprintf(tp, sizeof tp, " (tensor parallel %zu, part %d)", gsz, d.mi);
            char line[384];
            std::snprintf(line, sizeof line,
                          "ds4: CUDA%d layers [%d, %d]%s%s: weights %.2f GB, experts %.2f GB (%d of %d in VRAM), caches "
                          "%.2f GB, %.0f MiB free\n",
                          d.id, a, b, tp, (int) di == head_dev ? " + head" : "", d.weights / 1e9, d.experts / 1e9, ev, et,
                          d.caches / 1e9, fr / 1048576.0);
            s << line;
        }
        int nh = 0;
        size_t hb_ = 0;
        for (const Layer& l : L) {
            nh += l.n_exp - l.n_vram;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0) hb_ += xc(l, e).blob;
        }
        char line[200];
        if (file_experts)
            std::snprintf(line, sizeof line,
                          "ds4: %d routed experts on the CPU lane (%.1f GB, in place in the model's files), %d threads\n",
                          nh, hb_ / 1e9, cpu_threads());
        else
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB of RAM, %d NUMA parts), %d threads\n", nh,
                          hb_ / 1e9, NH, cpu_threads());
        s << line;
        if (n_pre >= 0 && chunk > 0 && o.prefill_keep >= 0) {
            std::snprintf(line, sizeof line, "ds4: a long prompt's layers %d-%d over its last %d rows (%s)\n", n_pre, g.nl - 1,
                          tail_rows(1 << 30), o.prefill_keep > 0 ? "approximate: their windows cut there" : "exact");
            s << line;
        }
        return s.str();
    }

    int cpu_threads() const {
        int n = 0;
        for (const auto& pl : pools) n += pl->size() - 1;
        return n + 1;
    }

    void reset() {
        pos = 0;
        toks.clear();
        ctoks.clear();
        ck[0].pos = ck[1].pos = -1;
        for (Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                Layer& l = member(l0, m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemsetAsync(l.win, 0, (size_t) g.win * g.hd * 2, d.st);
                if (l.c_kv) cudaMemsetAsync(l.c_kv, 0, (size_t) l.ratio * g.hd * 4, d.st);
                if (l.c_score) cudaMemsetAsync(l.c_score, 0, (size_t) l.ratio * g.hd * 4, d.st);
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the state at pos into checkpoint `slot` (the window rings and the compressors' partial groups: the rest of the
    // state is indexed by position, and what lies past a restored position is written again)
    void checkpoint(int slot) {
        Check& c = ck[slot & 1];
        c.pos = pos;
        c.win.resize(L.size());
        c.comp.resize(L.size() * 2);
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            c.win[i].resize((size_t) g.win * g.hd);
            cudaMemcpyAsync(c.win[i].data(), l.win, c.win[i].size() * 2, cudaMemcpyDeviceToHost, d.st);
            if (l.c_kv) {
                c.comp[2 * i].resize((size_t) l.ratio * g.hd);
                c.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                cudaMemcpyAsync(c.comp[2 * i].data(), l.c_kv, c.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, d.st);
                cudaMemcpyAsync(c.comp[2 * i + 1].data(), l.c_score, c.comp[2 * i + 1].size() * 4,
                                cudaMemcpyDeviceToHost, d.st);
            }
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the checkpoint at or before position limit (the state's tokens agree with a new prompt up to there), or -1
    int best_check(int64_t limit) const {
        int b = -1;
        for (int i = 0; i < 2; ++i)
            if (ck[i].pos >= 0 && ck[i].pos <= limit && ck[i].pos <= pos && (b < 0 || ck[i].pos > ck[b].pos)) b = i;
        return b;
    }
    int64_t restorable(int64_t limit) const {
        const int b = best_check(limit);
        return b < 0 ? -1 : ck[b].pos;
    }

    bool restore(int64_t limit) {
        const int b = best_check(limit);
        if (b < 0) return false;
        const Check& c = ck[b];
        for (size_t i = 0; i < L.size(); ++i)
            for (int m = 0; m < L[i].nm; ++m) {   // (a group's members hold the same state: the lead's copy)
                Layer& l = member(L[i], m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemcpyAsync(l.win, c.win[i].data(), c.win[i].size() * 2, cudaMemcpyHostToDevice, d.st);
                if (l.c_kv) {
                    cudaMemcpyAsync(l.c_kv, c.comp[2 * i].data(), c.comp[2 * i].size() * 4, cudaMemcpyHostToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, c.comp[2 * i + 1].data(), c.comp[2 * i + 1].size() * 4,
                                    cudaMemcpyHostToDevice, d.st);
                }
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        pos = c.pos;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        for (Check& o2 : ck)   // a later checkpoint saw tokens the state no longer has
            if (o2.pos > pos) o2.pos = -1;
        return true;
    }

    size_t bounce_bytes() const {
        return (size_t) kWin * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }
    size_t topk_stride() const { return (size_t) std::max(g.ix_topk, 32); }
    size_t cand_stride() const { return (size_t) std::max(g.cand_topk, 32); }
    size_t bm_stride() const { return (size_t) (o.max_context + 64) / std::max(1, g.cand_blk) + 64; }
    static constexpr int kSlots = kWin * kPicks;   // the CPU lane's (token, expert) rows a window can need

    // ---- the vision tower (the checkpoint's vision.* / aligner.*) on the first GPU, its weights streamed from the
    // host; a prompt's pictures as their spans' rows (set_images)
    VisionTower vt;
    VisionGeo vg;
    int img_tok = -1;
    std::vector<uint16_t> img_mark[3];   // image_start, image_newline, image_end: BF16 [dim]
    std::vector<Image> imgs;             // the pictures of the prompt being fed, by start
    bool load_vision(std::string& err) {
        vg.layers = (int) pk.num("vision.num_hidden_layers", 0);
        const int tok = (int) pk.num("image_token_id", -1);
        if (vg.layers <= 0 || tok < 0 || pk.get("vision.patch_embed.proj.weight") == nullptr) return true;   // none
        vg.dim = (int) pk.num("vision.hidden_size", 1024), vg.heads = (int) pk.num("vision.num_attention_heads", 16);
        vg.inter = (int) pk.num("vision.intermediate_size", 2816), vg.patch = (int) pk.num("vision.patch_size", 14);
        vg.merge = (int) pk.num("vision.downsample_ratio", 3), vg.theta = (float) pk.num("vision.rope_theta", 1e4);
        vg.max_tokens = (int) pk.num("vision.max_image_tokens", 1024), vg.out = g.dim;
        const char* mark[3] = {"image_start", "image_newline", "image_end"};
        for (int i = 0; i < 3; ++i) {
            const Tensor* t = pk.get(mark[i]);
            img_mark[i].resize((size_t) g.dim);
            if (t == nullptr || t->dtype != "BF16" || t->bytes != (uint64_t) g.dim * 2 || !pk.read(*t, img_mark[i].data())) {
                err = std::string(mark[i]) + ": not a BF16 row of the model's width";
                return false;
            }
        }
        const double t0 = now_ms();
        Dev& d = devs[0];
        auto read = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) {
            const Tensor* t = pk.get(n);
            if (t == nullptr || t->dtype != "BF16") return false;
            shape = t->shape;
            return dst == nullptr || pk.read(*t, dst);
        };
        if (!vt.load(vg, d.id, read, err)) return false;
        img_tok = tok;
        CK(cudaSetDevice(d.id));
        CK(cudaMemGetInfo(&d.free0, &d.total));   // (the split sees what the tower left)
        if (o.verbose)
            std::fprintf(stderr, "ds4: vision tower (%d blocks) on CUDA%d: %.0f MiB, up to %d patches, %.1f s\n",
                         vg.layers, d.id, (double) vt.device_bytes() / (1 << 20), vt.max_patches(),
                         (now_ms() - t0) / 1000.0);
        return true;
    }
    bool encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err) {
        if (img_tok < 0) {
            err = "this model has no vision tower (or it is off)";
            return false;
        }
        std::vector<uint16_t> rows;
        int nh = 0, nw = 0;
        if (!vt.encode(rgb, w, h, rows, nh, nw, err)) return false;
        const size_t D = (size_t) g.dim;
        span.resize(((size_t) nh * (nw + 1) + 2) * D);
        uint16_t* o = span.data();
        auto put = [&](const uint16_t* r) {
            std::memcpy(o, r, D * 2);
            o += D;
        };
        put(img_mark[0].data());
        for (int i = 0; i < nh; ++i) {
            for (int j = 0; j < nw; ++j) put(rows.data() + ((size_t) i * nw + j) * D);
            put(img_mark[1].data());
        }
        put(img_mark[2].data());
        return true;
    }
    // the row of a picture's span at position p (null: p is in none)
    const uint16_t* image_row(int64_t p) const {
        if (imgs.empty()) return nullptr;
        auto it = std::upper_bound(imgs.begin(), imgs.end(), p, [](int64_t v, const Image& im) { return v < im.start; });
        if (it == imgs.begin()) return nullptr;
        const Image& im = *(it - 1);
        const int64_t r = p - im.start;
        if (!im.rows || r >= (int64_t) (im.rows->size() / (size_t) g.dim)) return nullptr;
        return im.rows->data() + (size_t) r * g.dim;
    }
    // token tk's embedding row at position p: inside a picture's span, the picture's
    const uint16_t* emb_row(int64_t p, int tk) const {
        const uint16_t* r = tk == img_tok ? image_row(p) : nullptr;
        return r != nullptr ? r : embed_pinned + (size_t) tk * g.dim;
    }
    bool in_image(int64_t p) const { return img_tok >= 0 && toks[(size_t) p] == img_tok && image_row(p) != nullptr; }
    // the Engram's compressed id of token tk at p (-1: a picture's, which no n-gram spans)
    int32_t ctok(int64_t p, int tk) const {
        if (tk == img_tok && image_row(p) != nullptr) return -1;
        return tk < pk.eg_vocab ? pk.eg_map[(size_t) tk] : 0;
    }
    // a chunk's image rows (route()'s bits); false: none
    bool vl_bits(int64_t p0, int T, VlBits& b) const {
        if (imgs.empty()) return false;
        std::memset(&b, 0, sizeof b);
        bool any = false;
        for (int t = 0; t < T && t < kVlRows; ++t)
            if (in_image(p0 + t)) {
                b.w[t >> 5] |= 1u << (t & 31);
                any = true;
            }
        return any;
    }

    // ---- Engram: the window's rows of both tables, read on the host
    // a prompt's (the chunked path): every chunk's rows of each Engram layer read once, ahead of the layers, by a
    // reader thread on an IO pool of its own (deep queues for the NVMe) into kEgSlots shared pinned slots
    // ([layer][chunk rows][per]); a device waits only before its Engram layer, and a slot is read again once every
    // copy from it is done (eg_used: the copies issued, each device's eg_ev[slot] after its last)
    uint8_t* eg_shared[kEgSlots] = {};
    std::unique_ptr<Pool> eg_pool;
    std::atomic<int> eg_read{0};
    std::atomic<int> eg_used[kEgSlots] = {};
    std::atomic<bool> eg_abort{false};   // (the devices' run over: a failed one leaves copies that never come)
    int eg_copies = 0;                 // copies a chunk (an Engram layer's group members, every layer)
    std::vector<int> eg_devs;          // the devices that copy
    void eg_reader(const std::vector<int64_t>& p0, const std::vector<int>& T) {
        const size_t per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        for (size_t k = 0; k < T.size(); ++k) {
            const int sl = (int) (k % kEgSlots);
            if (k >= (size_t) kEgSlots) {   // (chunk k - kEgSlots's copies issued, then done)
                const int want = eg_copies * (int) (k / kEgSlots);
                while (eg_used[sl].load(std::memory_order_acquire) < want) {
                    if (eg_abort.load(std::memory_order_acquire)) return;
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
                for (int di : eg_devs) {
                    cudaSetDevice(devs[(size_t) di].id);
                    cudaEventSynchronize(chunks[(size_t) di]->eg_ev[sl]);
                }
            }
            for (int li = 0; li < pk.eg_layers; ++li)
                engram_rows(p0[k], T[k], li, eg_shared[sl] + (size_t) li * chunk * per, eg_pool.get());
            eg_read.store((int) k + 1, std::memory_order_release);
        }
    }
    std::mutex eg_mu;   // (the decode's windows read Engram rows on the last pool, one at a time)
    // a decode window's (threads mode): its rows all in flight at once (io_uring) from a thread of their own, while the
    // window's first layers run - a device waits only before its Engram layer (eg_win: the window whose rows are in
    // h_eg).  Before, the last CPU-lane pool's threads read them in pread, ~22 deep: ~0.44 ms of the host's a token
    // before every window (and the same reads on an IO pool during the window slowed the CPU lane by a fifth).
    // DS4_EG_URING=1 (opt-in: in the CLI's decode it was slower than the pool's reads, and the serve bench's A/B is
    // still to run)
    std::thread eg_th;
    std::mutex eg_th_mu;
    std::condition_variable eg_th_cv, eg_done_cv;
    struct EgJob {
        int64_t p0 = 0;
        int T = 0;
        unsigned win = 0;
    };
    std::deque<EgJob> eg_jobs;
    std::atomic<unsigned> eg_win[2] = {};   // (a table's: layer 1's are read first)
    Uring eg_ring;
    bool eg_ring_tried = false;
    bool eg_async() const { return par && eg_ring.ok() && pk.eg_layers > 0 && pk.eg_layers <= 2; }
    // the tables' files opened again for these reads: the rows by O_DIRECT (each read once, never again: they
    // stay out of the page cache), the scales buffered with FADV_RANDOM (no readahead) - so the page cache keeps
    // the scales (8 bytes a row, a 3 GB table a layer: half the reads; the NVMe here does ~150K of these a second)
    std::unordered_map<int, std::pair<int, int>> eg_fd;   // file -> (buffered, O_DIRECT; -1: none)
    uint8_t* eg_bounce = nullptr;   // (the O_DIRECT reads' aligned buffers, 1 KB a read)
    size_t eg_bounce_n = 0;
    void eg_open() {
        for (int li = 0; li < pk.eg_layers; ++li) {
            const std::string P = "layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.";
            for (const char* w : {"weight", "scale"}) {
                const Tensor* t = pk.get(P + w);
                if (t == nullptr || eg_fd.count(t->file)) continue;
                const std::string path = pk.model_dir + "/" + pk.files[(size_t) t->file];
                const int b = ::open(path.c_str(), O_RDONLY), dd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
                if (b >= 0) posix_fadvise(b, 0, 0, POSIX_FADV_RANDOM);
                eg_fd[t->file] = {b, dd};
            }
        }
    }
    void eg_loop() {
        const size_t per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        for (;;) {
            EgJob j;
            {
                std::unique_lock<std::mutex> lk(eg_th_mu);
                eg_th_cv.wait(lk, [&] { return quit.load() || !eg_jobs.empty(); });
                if (quit.load()) return;
                j = eg_jobs.front();
                eg_jobs.pop_front();
            }
            for (int li = 0; li < pk.eg_layers; ++li) {
                engram_rows(j.p0, j.T, li, h_eg + (size_t) li * eg_cap * per, nullptr, &eg_ring);
                {
                    std::lock_guard<std::mutex> lk(eg_th_mu);
                    eg_win[li].store(j.win, std::memory_order_release);
                }
                eg_done_cv.notify_all();
            }
        }
    }
    void eg_wait(int li, unsigned win) {
        if (eg_win[li].load(std::memory_order_acquire) == win) return;
        std::unique_lock<std::mutex> lk(eg_th_mu);
        eg_done_cv.wait(lk, [&] { return quit.load() || eg_win[li].load(std::memory_order_acquire) == win; });
    }
    void engram_rows(int64_t p0, int T, int only = -1, uint8_t* dst = nullptr, Pool* pool = nullptr, Uring* ring = nullptr) {
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, hd = pk.eg_hd;
        const size_t per = (size_t) cols * (hd + hd / 32);
        struct Read {
            const Tensor* t;
            uint64_t off;
            uint8_t* dst;
            uint32_t n;
        };
        std::vector<Read> reads;
        reads.reserve((size_t) pk.eg_layers * T * cols * 2);
        for (int li = 0; li < pk.eg_layers; ++li) {
            if (only >= 0 && li != only) continue;
            const std::string P = "layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.";
            const Tensor* tw = pk.get(P + "weight");
            const Tensor* ts = pk.get(P + "scale");
            for (int t = 0; t < T; ++t) {
                const int64_t p = p0 + t;
                uint8_t* rows = dst != nullptr ? dst + (size_t) t * per : h_eg + ((size_t) li * eg_cap + t) * per;
                uint8_t* scales = rows + (size_t) cols * hd;
                if (ctoks[(size_t) p] < 0) {   // a picture's token: no rows (zeros: its gate adds nothing)
                    std::memset(rows, 0, per);
                    continue;
                }
                int64_t prod[8];
                bool blocked = false;
                for (int s = 0; s < pk.eg_ngram; ++s) {   // (the look-back stops at the start and at a picture)
                    blocked = blocked || p < s || ctoks[(size_t) (p - s)] < 0;
                    const int64_t tk = blocked ? pk.eg_pad : ctoks[(size_t) (p - s)];
                    prod[s] = tk * pk.eg_mult[(size_t) li * pk.eg_ngram + s];
                }
                int64_t rolling = prod[0];
                for (int n = 1; n < pk.eg_ngram; ++n) {
                    rolling ^= prod[n];
                    for (int h = 0; h < pk.eg_heads; ++h) {
                        const int c = (n - 1) * pk.eg_heads + h;
                        const int64_t id = rolling % pk.eg_primes[(size_t) li * cols + c] + pk.eg_off[(size_t) li * cols + c];
                        reads.push_back({tw, tw->off + (uint64_t) id * hd, rows + (size_t) c * hd, (uint32_t) hd});
                        reads.push_back({ts, ts->off + (uint64_t) id * (hd / 32), scales + (size_t) c * (hd / 32),
                                         (uint32_t) (hd / 32)});
                    }
                }
            }
        }
        auto one = [&](int i) {
            const Read& r = reads[(size_t) i];
            pk.read_at(r.t->file, r.off, r.n, r.dst);
        };
        if (ring != nullptr) {
            if (eg_bounce_n < reads.size()) {
                std::free(eg_bounce);
                eg_bounce_n = reads.size() * 2;
                eg_bounce = (uint8_t*) std::aligned_alloc(4096, eg_bounce_n * 1024);
            }
            std::vector<UringRead> ur(reads.size());
            std::vector<uint32_t> skip(reads.size(), ~0u);   // (an O_DIRECT read: the row's offset in its sectors)
            for (size_t i = 0; i < reads.size(); ++i) {
                const Read& r = reads[i];
                const auto f = eg_fd.find(r.t->file);
                const int b = f != eg_fd.end() && f->second.first >= 0 ? f->second.first : pk.fds[(size_t) r.t->file];
                const int dd = f != eg_fd.end() ? f->second.second : -1;
                const uint64_t a0 = r.off / 512 * 512, a1 = (r.off + r.n + 511) / 512 * 512;
                if (dd >= 0 && eg_bounce != nullptr && r.n >= 64 && a1 - a0 <= 1024) {
                    ur[i] = {dd, a0, (uint32_t) (a1 - a0), eg_bounce + i * 1024};
                    skip[i] = (uint32_t) (r.off - a0);
                } else
                    ur[i] = {b, r.off, r.n, r.dst};
            }
            if (ring->read_all(ur)) {
                for (size_t i = 0; i < reads.size(); ++i)
                    if (skip[i] != ~0u) std::memcpy(reads[i].dst, eg_bounce + i * 1024 + skip[i], reads[i].n);
                return;
            }
            for (size_t i = 0; i < reads.size(); ++i) one((int) i);   // (the ring failed: pread, here)
            return;
        }
        if (pool != nullptr) {
            pool->run((int) reads.size(), one);
            return;
        }
        std::lock_guard<std::mutex> lk(eg_mu);
        pools.back()->run((int) reads.size(), one);
    }

    // ---- the CPU lane: a window's routed experts that live on the host, as entries (an expert and up to 8 of its
    // tokens, in rows slot0 ..) computed together: each node its column part of every matrix, from memory on that node,
    // in four rounds (the gate / up split-K pieces, each preparing its two input blocks of every token; each band's
    // sums / codebook / Hadamard / swiglu and that block of the down input, which both nodes then read whole; the down
    // pieces; the down bands).  yd[slot] gets each row's expert output.
    struct Ent {
        int e = 0, n = 0, slot0 = 0;
        int tok[kc::kExl3CpuMaxN] = {};
    };
    void run_nodes(int n, const std::function<void(int)>* f, int ph = -1) {
        static const bool stats = std::getenv("DS4_CPU_STATS") != nullptr;
        if (stats && ph >= 0) {   // (each job's own time, summed: the phase's work beside its wall time)
            std::function<void(int)> g[4];
            for (int X = 0; X < NH; ++X)
                g[X] = [this, &f, X, ph](int j) {
                    const auto t0 = std::chrono::steady_clock::now();
                    f[X](j);
                    cpu_job_ns[ph].fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - t0).count(),
                                             std::memory_order_relaxed);
                    cpu_job_n[ph].fetch_add(1, std::memory_order_relaxed);
                };
            for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, g[X]);
            pools[0]->help();
            for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
            return;
        }
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    }

    // DS4_CPU_STATS: the CPU lane's calls, entries, rows and each phase's time, printed with the profile's save
    double cpu_ph_ms[4] = {};
    long cpu_calls = 0, cpu_ents = 0, cpu_rows = 0;
    std::atomic<uint64_t> cpu_job_ns[4] = {}, cpu_job_n[4] = {};
    void cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx = 0) {
        if (ldx == 0) ldx = (size_t) g.dim;
        static const bool stats = std::getenv("DS4_CPU_STATS") != nullptr;
        double tph = stats ? now_ms() : 0.0;
        auto phase = [&](int i) {
            if (!stats) return;
            const double t = now_ms();
            cpu_ph_ms[i] += t - tph;
            tph = t;
        };
        if (stats) {
            ++cpu_calls;
            cpu_ents += (long) ents.size();
            for (const Ent& en : ents) cpu_rows += en.n;
        }
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
        const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
        const int ne = (int) ents.size();
        const float lim = g.lim;
        auto part = [&](int e, int X, int r) {
            const uint8_t* b = l.hh[(size_t) e][(size_t) X];
            const Layer::XCls& C = xc(l, e);
            kc::Exl3Part p;
            p.trellis = (const uint32_t*) (b + C.hl.trellis[r]);
            p.suh = (const uint16_t*) (b + C.hl.suh[r]);
            p.svh = (const uint16_t*) (b + C.hl.svh[r]);
            p.k = C.hl.k[r];
            p.n = C.hl.n[r];
            p.K = C.lay.K[r];
            return p;
        };
        auto xh = [&](int X, int sl, int r) { return c_xh.data() + ((size_t) (X * kSlots + sl) * 2 + r) * E; };
        auto xsb = [&](int X, int sl, int r) { return c_xsb.data() + ((size_t) (X * kSlots + sl) * 2 + r) * EB; };
        auto acc = [&](int X, int sl, int r, int q) {
            return c_acc.data() + (((size_t) (X * kSlots + sl) * 2 + r) * GQ + q) * FFh;
        };
        auto xhd = [&](int sl) { return c_xhd.data() + (size_t) sl * FF; };
        auto xsd = [&](int sl) { return c_xsd.data() + (size_t) sl * FB; };
        auto accd = [&](int X, int sl, int q) { return c_accd.data() + ((size_t) (X * kSlots + sl) * DQ + q) * Eh; };
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / (2 * GQ))];
                const int r = (j / GQ) % 2, q = j % GQ;
                const kc::Exl3Part p = part(en.e, X, r);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    const float* xt = x + (size_t) en.tok[i] * ldx;
                    for (int b = 2 * q; b < 2 * q + 2; ++b)
                        xsb(X, sl, r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, xt + b * 128, xh(X, sl, r) + b * 128);
                    xp[i] = xh(X, sl, r);
                    ap[i] = acc(X, sl, r, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 16, (q + 1) * 16, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 16, (q + 1) * 16, ap);
            };
        run_nodes(ne * 2 * GQ, f, 0);
        phase(0);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / GB)];
                const int b = j % GB;
                const kc::Exl3Part pg = part(en.e, X, 0), pu = part(en.e, X, 1), pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float y[2][128], a[128], hb[128];
                    for (int r = 0; r < 2; ++r) {
                        for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                        for (int q = 0; q < GQ; ++q) {
                            const float* aq = acc(X, sl, r, q) + b * 128;
                            for (int c = 0; c < 128; ++c) a[c] += aq[c];
                        }
                        double xs = 0.0;
                        for (int k = 0; k < EB; ++k) xs += xsb(X, sl, r)[k];
                        kc::exl3_cpu_band_out(r == 0 ? pg : pu, a, (float) xs, b * 128, y[r]);
                    }
                    for (int c = 0; c < 128; ++c) {
                        float gv = y[0][c], uv = y[1][c];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        hb[c] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    const int gb = X * GB + b;
                    xsd(sl)[gb] = kc::exl3_cpu_prepare_block(pd.suh + gb * 128, hb, xhd(sl) + gb * 128);
                }
            };
        run_nodes(ne * GB, f, 1);
        phase(1);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DQ)];
                const int q = j % DQ;
                const kc::Exl3Part p = part(en.e, X, 2);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    xp[i] = xhd(en.slot0 + i);
                    ap[i] = accd(X, en.slot0 + i, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 8, (q + 1) * 8, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 8, (q + 1) * 8, ap);
            };
        run_nodes(ne * DQ, f, 2);
        phase(2);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DB)];
                const int b = j % DB;
                const kc::Exl3Part pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float a[128];
                    for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                    for (int q = 0; q < DQ; ++q) {
                        const float* aq = accd(X, sl, q) + b * 128;
                        for (int c = 0; c < 128; ++c) a[c] += aq[c];
                    }
                    double xs = 0.0;
                    for (int k = 0; k < FB; ++k) xs += xsd(sl)[k];
                    kc::exl3_cpu_band_out(pd, a, (float) xs, b * 128, c_yd.data() + (size_t) sl * E + (size_t) X * Eh + b * 128);
                }
            };
        run_nodes(ne * DB, f, 3);
        phase(3);
    }

    // ---- the expert cache.  A swap moves a CPU-lane expert (`in`) into the layer's spare VRAM slot and the VRAM expert it
    // replaces (`out`) into the spare host parts - both copies of weights that do not change, made while the model runs
    // (the copy stream, the host's repacking) - and cache_tick switches the tables between windows, so the device's
    // table and the host's view of a layer always agree within a window.  In place (file_experts) `out` needs no copy:
    // the CPU lane reads it from the file again.
    void cache_loop(SwapBufs& B) {
        for (;;) {
            Swap sw;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                sw_cv.wait(lk, [&] { return quit.load() || !sw_queue.empty(); });
                if (quit.load()) return;
                sw = sw_queue.front();
                sw_queue.pop_front();
            }
            Layer& l = L[(size_t) sw.layer];
            const Layer::XCls& C = l.cls[0];   // (the cache swaps a layer of one layout alone)
            parts_to_blob(l, sw.in, l.hh[(size_t) sw.in], B.a);
            auto off_turn = [&](int di, cudaStream_t cs) {   // (false: quitting)
                while (!copy_open(di, sw.win)) {
                    if (quit.load()) return false;
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                copy_after_idle(di, cs);
                return true;
            };
            if (l.nm == 1) {
                const Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaStream_t cs = B.st[(size_t) l.dev];
                if (!off_turn(l.dev, cs)) return;
                cudaMemcpyAsync(slot_ptr(l, l.spare_vram), B.a, C.blob, cudaMemcpyHostToDevice, cs);
                if (!file_experts)
                    cudaMemcpyAsync(B.b, slot_ptr(l, l.slot[(size_t) sw.out]), C.blob,
                                    cudaMemcpyDeviceToHost, cs);
                cudaStreamSynchronize(cs);
            } else {   // a group: each member's part in, each member's part out
                for (int m = 0; m < l.nm; ++m) {
                    const Layer& x = member(l, m);
                    cudaSetDevice(devs[(size_t) x.dev].id);
                    cudaStream_t cs = B.st[(size_t) x.dev];
                    blob_to_half(l, sw.in, B.a, m, B.h);
                    if (!off_turn(x.dev, cs)) return;
                    cudaMemcpyAsync(slot_ptr(x, l.spare_vram), B.h, C.hblob, cudaMemcpyHostToDevice, cs);
                    cudaStreamSynchronize(cs);
                    if (file_experts) continue;
                    if (!off_turn(x.dev, cs)) return;
                    cudaMemcpyAsync(B.h, slot_ptr(x, l.slot[(size_t) sw.out]), C.hblob,
                                    cudaMemcpyDeviceToHost, cs);
                    cudaStreamSynchronize(cs);
                    half_to_blob(l, sw.out, B.h, m, B.b);
                }
            }
            if (!file_experts)
                for (int X = 0; X < NH; ++X) blob_to_part(l, sw.out, B.b, X, l.spare_host[(size_t) X]);
            if (check_swaps) {   // DS4_CHECK_SWAPS: both moves against the model file, byte for byte
                std::vector<uint8_t> reg, blob(C.blob), part(C.hl.bytes);
                bool ok = true;
                for (int pass = 0; pass < (file_experts ? 1 : 2); ++pass) {   // (in place: no host copy of `out`)
                    const int e = pass == 0 ? sw.in : sw.out;
                    reg.resize(l.region_bytes[(size_t) e]);
                    pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], reg.size(), reg.data());
                    if (pass == 0) {
                        repack(l, e, reg.data(), blob.data());
                        std::vector<uint8_t> dev(C.blob);
                        if (l.nm == 1)
                            cudaMemcpy(dev.data(), slot_ptr(l, l.spare_vram), C.blob, cudaMemcpyDeviceToHost);
                        else
                            for (int m = 0; m < l.nm; ++m) {   // (the members' parts, put together again)
                                std::vector<uint8_t> h(C.hblob);
                                cudaSetDevice(devs[(size_t) member(l, m).dev].id);
                                cudaMemcpy(h.data(), slot_ptr(member(l, m), l.spare_vram), C.hblob,
                                           cudaMemcpyDeviceToHost);
                                half_to_blob(l, sw.in, h.data(), m, dev.data());
                            }
                        for (int r = 0; r < 3; ++r)
                            ok = ok && std::memcmp(dev.data() + C.lay.trellis[r], blob.data() + C.lay.trellis[r], C.piece_bytes[r][0]) == 0 &&
                                 std::memcmp(dev.data() + C.lay.suh[r], blob.data() + C.lay.suh[r], C.piece_bytes[r][1]) == 0 &&
                                 std::memcmp(dev.data() + C.lay.svh[r], blob.data() + C.lay.svh[r], C.piece_bytes[r][2]) == 0;
                    } else {
                        for (int X = 0; X < NH; ++X) {
                            repack_part(l, e, reg.data(), X, part.data());
                            for (int r = 0; r < 3; ++r) {
                                const size_t tb = (size_t) (C.hl.k[r] / 16) * (C.hl.n[r] / 16) * 32 * C.lay.K[r];
                                ok = ok && std::memcmp(part.data() + C.hl.trellis[r], l.spare_host[(size_t) X] + C.hl.trellis[r], tb) == 0 &&
                                     std::memcmp(part.data() + C.hl.suh[r], l.spare_host[(size_t) X] + C.hl.suh[r], (size_t) C.hl.k[r] * 2) == 0 &&
                                     std::memcmp(part.data() + C.hl.svh[r], l.spare_host[(size_t) X] + C.hl.svh[r], (size_t) C.hl.n[r] * 2) == 0;
                            }
                        }
                    }
                }
                std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out: %s\n", l.il, sw.in, sw.out,
                             ok ? "both match the model file" : "MISMATCH");
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            sw_ready.push_back(sw);
        }
    }

    void rc_loop() {
        std::vector<RcJob> fly;     // issued, not yet done
        std::deque<RcJob> held;     // waiting for their group's turn to end (copy_open)
        std::string err;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(rc_mu);
                if (fly.empty() && held.empty()) rc_cv.wait(lk, [&] { return quit.load() || !rc_q.empty(); });
                if (quit.load()) return;
                for (const RcJob& j : rc_q) held.push_back(j);
                rc_q.clear();
            }
            for (size_t i = 0; i < held.size();) {
                const RcJob j = held[i];
                Layer& l = L[(size_t) j.layer];
                bool open = true;   // (a layer's members are one group's)
                for (int m = 0; m < l.nm && open; ++m) open = copy_open(member(l, m).dev, j.win);
                if (!open) {
                    ++i;
                    continue;
                }
                for (int m = 0; m < l.nm; ++m) {
                    Layer& x = member(l, m);
                    const int di = x.dev;
                    cudaSetDevice(devs[(size_t) di].id);
                    cudaStreamWaitEvent(rc_cs[(size_t) di], rc_go[(size_t) di], 0);   // (after the tick's tables)
                    copy_after_idle(di, rc_cs[(size_t) di]);
                    stream_expert(l, m, j.expert, slot_ptr(x, l.rc_base + j.slot), rc_cs[(size_t) di], err);
                    cudaEventRecord(l.rc_ev[(size_t) j.slot * l.nm + m], rc_cs[(size_t) di]);
                }
                fly.push_back(j);
                held.erase(held.begin() + (std::ptrdiff_t) i);
            }
            for (size_t i = 0; i < fly.size();) {   // the ones whose copies are all done: marked
                const RcJob& j = fly[i];
                Layer& l = L[(size_t) j.layer];
                bool done = true;
                for (int m = 0; m < l.nm && done; ++m) {
                    cudaSetDevice(devs[(size_t) member(l, m).dev].id);
                    done = cudaEventQuery(l.rc_ev[(size_t) j.slot * l.nm + m]) == cudaSuccess;
                }
                if (done) {
                    rc_done[(size_t) (l.rc_off + j.slot)].store(j.gen, std::memory_order_release);
                    fly[i] = fly.back();
                    fly.pop_back();
                } else
                    ++i;
            }
            if (!fly.empty() || !held.empty()) std::this_thread::sleep_for(std::chrono::microseconds(held.empty() ? 200 : 50));
        }
    }

    void rc_tick() {
        if (recent_n <= 0) return;
        const double t_tick = now_ms();
        struct Tm {
            Model::Impl* m;
            double t;
            ~Tm() {
                m->rc_tick_ms += now_ms() - t;
                ++m->rc_ticks;
            }
        } tm{this, t_tick};
        if (rc_cs.empty()) {
            if (!tab_bufs()) return;
            rc_cs.assign(devs.size(), nullptr);
            for (size_t di = 0; di < devs.size(); ++di) {
                cudaSetDevice(devs[di].id);
                cudaStreamCreateWithFlags(&rc_cs[di], cudaStreamNonBlocking);
            }
            int off = 0;
            for (Layer& l : L) {
                l.rc_off = off;
                off += l.rc_n;
            }
            rc_done.reset(new std::atomic<unsigned>[(size_t) std::max(1, off)]);
            for (int i = 0; i < off; ++i) rc_done[(size_t) i].store(0u);
            rc_th = std::thread([this] { rc_loop(); });
        }
        const unsigned now = rc_clock.fetch_add(1, std::memory_order_relaxed) + 1;
        std::vector<std::vector<unsigned long long>>& upd = tick_upd;
        std::vector<RcJob>& jobs = tick_jobs;
        for (Layer& l : L) {
            if (l.rc_n == 0) continue;
            const int nm = l.nm;
            auto set_tab = [&](int e, int i) {   // (i < 0: out of VRAM)
                for (int m = 0; m < nm; ++m) {
                    Layer& x = member(l, m);
                    upd[(size_t) x.dev].push_back((unsigned long long) (x.vram_tab + e));
                    upd[(size_t) x.dev].push_back(i < 0 ? 0ull : l.rc_pin[(size_t) i * nm + m]);
                }
            };
            for (int i = 0; i < l.rc_n; ++i) {   // the fills done: mapped (unless a swap took the expert in meanwhile)
                const int e = l.rc_fill[(size_t) i];
                if (e < 0 || rc_done[(size_t) (l.rc_off + i)].load(std::memory_order_acquire) != l.rc_gen[(size_t) i]) continue;
                l.rc_fill[(size_t) i] = -1;
                if (l.slot[(size_t) e] >= 0) continue;
                l.slot[(size_t) e] = l.rc_base + i;
                l.rc_exp[(size_t) i] = e;
                l.rc_stamp[(size_t) i] = now;
                if (nm == 1) set_vm(l, e, l.rc_base + i);
                set_tab(e, i);
                ++rc_fills_done;
                if (check_swaps) {   // DS4_CHECK_SWAPS: the slot against the model file, byte for byte
                    const Layer::XCls& C = l.cls[0];   // (recent slots: a layer of one layout alone)
                    std::vector<uint8_t> reg(l.region_bytes[(size_t) e]), want(C.blob), dev(C.blob);
                    pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], reg.size(), reg.data());
                    repack(l, e, reg.data(), want.data());
                    if (nm == 1) {
                        cudaSetDevice(devs[(size_t) l.dev].id);
                        cudaMemcpy(dev.data(), slot_ptr(l, l.rc_base + i), C.blob, cudaMemcpyDeviceToHost);
                    } else
                        for (int m = 0; m < nm; ++m) {
                            std::vector<uint8_t> h(member(l, m).cls[0].hblob);
                            cudaSetDevice(devs[(size_t) member(l, m).dev].id);
                            cudaMemcpy(h.data(), slot_ptr(member(l, m), l.rc_base + i),
                                       member(l, m).cls[0].hblob, cudaMemcpyDeviceToHost);
                            half_to_blob(l, e, h.data(), m, dev.data());
                        }
                    bool ok = true;
                    for (int r = 0; r < 3; ++r)
                        ok = ok && std::memcmp(dev.data() + C.lay.trellis[r], want.data() + C.lay.trellis[r], C.piece_bytes[r][0]) == 0 &&
                             std::memcmp(dev.data() + C.lay.suh[r], want.data() + C.lay.suh[r], C.piece_bytes[r][1]) == 0 &&
                             std::memcmp(dev.data() + C.lay.svh[r], want.data() + C.lay.svh[r], C.piece_bytes[r][2]) == 0;
                    std::fprintf(stderr, "ds4 recent: layer %d expert %d in slot %d: %s\n", l.il, e, l.rc_base + i,
                                 ok ? "matches the model file" : "MISMATCH");
                }
            }
            {   // this tick's CPU-lane experts into the history
                std::vector<int> cur(l.cpu_hits.begin(), l.cpu_hits.end());
                std::sort(cur.begin(), cur.end());
                cur.erase(std::unique(cur.begin(), cur.end()), cur.end());
                l.rc_hist.push_back(std::move(cur));
                while ((int) l.rc_hist.size() > recent_win) l.rc_hist.pop_front();
            }
            int started = 0;
            for (size_t h = l.cpu_hits.size(); h-- > 0 && started < recent_fills;) {   // (the latest first)
                const int e = l.cpu_hits[h];
                if (e < 0 || e >= l.n_exp || l.slot[(size_t) e] >= 0 || e == l.sw_in) continue;
                int uses = 0;
                for (const std::vector<int>& hv : l.rc_hist) uses += std::binary_search(hv.begin(), hv.end(), e) ? 1 : 0;
                if (uses < recent_need) continue;
                bool filling = false;
                for (int i = 0; i < l.rc_n; ++i) filling = filling || l.rc_fill[(size_t) i] == e;
                if (filling) continue;
                int v = -1;   // an empty slot, else the least recently used one not filling
                for (int i = 0; i < l.rc_n && v < 0; ++i)
                    if (l.rc_fill[(size_t) i] < 0 && l.rc_exp[(size_t) i] < 0) v = i;
                if (v < 0)
                    for (int i = 0; i < l.rc_n; ++i)
                        if (l.rc_fill[(size_t) i] < 0 && (v < 0 || l.rc_stamp[(size_t) i] < l.rc_stamp[(size_t) v])) v = i;
                if (v < 0) break;
                if (l.rc_exp[(size_t) v] >= 0) {   // (out: its table entry, before the next window)
                    const int old = l.rc_exp[(size_t) v];
                    l.slot[(size_t) old] = -1;
                    set_tab(old, -1);
                    l.rc_exp[(size_t) v] = -1;
                }
                l.rc_fill[(size_t) v] = e;
                l.rc_gen[(size_t) v] = now;
                jobs.push_back({l.il, v, e, now, win_now.load(std::memory_order_relaxed)});
                ++started;
                ++rc_fills_started;
            }
        }
    }

    // a tick's table entries (the recent slots', the swaps'): one launch a device, before the next window; then rc_go
    // there (the fills after it).  (A copy of each entry on its own was ~10 us of the host's, 48 of them 0.5 ms between
    // two windows)
    std::vector<std::vector<unsigned long long>> tick_upd;   // [device]: (entry, value) pairs
    std::vector<RcJob> tick_jobs;
    bool tab_bufs() {
        if (!rc_upd.empty()) return true;
        std::string e0;
        rc_go.assign(devs.size(), nullptr);
        rc_upd.assign(devs.size(), nullptr);
        for (size_t di = 0; di < devs.size(); ++di) {
            cudaSetDevice(devs[di].id);
            cudaEventCreateWithFlags(&rc_go[di], cudaEventDisableTiming);
            if (!hmap(rc_upd[di], (size_t) 2 * kRcUpd, e0)) {
                rc_upd.clear();
                return false;
            }
        }
        return true;
    }
    void tick_flush() {
        const std::vector<RcJob>& jobs = tick_jobs;
        bool any = !jobs.empty();
        for (const auto& u : tick_upd) any = any || !u.empty();
        if (!any || !tab_bufs()) return;
        for (size_t di = 0; di < devs.size(); ++di) {
            const std::vector<unsigned long long>& u = tick_upd[di];
            if (u.empty() && jobs.empty()) continue;
            cudaSetDevice(devs[di].id);
            const int n = (int) std::min<size_t>(u.size() / 2, (size_t) kRcUpd);
            if (n > 0) {
                std::memcpy(rc_upd[di], u.data(), (size_t) n * 16);
                tab_set(rc_upd[di], n, devs[di].st);
            }
            cudaEventRecord(rc_go[di], devs[di].st);
        }
        if (!jobs.empty()) {
            {
                std::lock_guard<std::mutex> lk(rc_mu);
                for (const RcJob& j : jobs) rc_q.push_back(j);
            }
            rc_cv.notify_one();
        }
    }

    void cache_tick(int T) {
        tick_upd.resize(devs.size());
        for (auto& u : tick_upd) u.clear();
        tick_jobs.clear();
        rc_tick();
        swap_tick(T);
        tick_flush();
    }
    void swap_tick(int T) {
        if (!o.expert_cache || cache_ths.empty()) return;
        std::vector<Swap> ready;
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            ready.swap(sw_ready);
        }
        for (const Swap& sw : ready) {
            Layer& l = L[(size_t) sw.layer];
            const int old_slot = l.slot[(size_t) sw.out];
            const std::array<uint8_t*, 4> old_parts = l.hh[(size_t) sw.in];
            l.slot[(size_t) sw.in] = l.spare_vram;
            if (l.nm == 1) set_vm(l, sw.in, l.spare_vram);
            l.hh[(size_t) sw.in] = {nullptr, nullptr, nullptr, nullptr};
            l.host[(size_t) sw.in] = nullptr;
            l.slot[(size_t) sw.out] = -1;
            if (file_experts) {   // (back to its pages in the file)
                l.hh[(size_t) sw.out] = {file_part(l, sw.out), nullptr, nullptr, nullptr};
                l.host[(size_t) sw.out] = l.hh[(size_t) sw.out][0];
            } else {
                l.hh[(size_t) sw.out] = l.spare_host;
                l.host[(size_t) sw.out] = l.spare_host[0];
                l.spare_host = old_parts;
            }
            l.spare_vram = old_slot;
            for (int m = 0; m < l.nm; ++m) {   // (tick_flush: stream order, before the next window)
                Layer& x = member(l, m);
                std::vector<unsigned long long>& u = tick_upd[(size_t) x.dev];
                u.push_back((unsigned long long) (x.vram_tab + sw.in));
                u.push_back((unsigned long long) slot_ptr(x, l.slot[(size_t) sw.in]));
                u.push_back((unsigned long long) (x.vram_tab + sw.out));
                u.push_back(0ull);
            }
            l.swap_pending = false;
            l.sw_in = -1;
            --sw_inflight;
            ++st.swaps;
        }
        decay_tokens += T;
        if (decay_tokens >= 256) {   // older use counts less: x0.9 every 256 tokens
            decay_tokens = 0;
            for (Layer& l : L)
                for (float& h : l.heat) h *= c_decay;
        }
        bool queued = false;
        for (Layer& l : L) {
            if (l.cpu_hits.empty()) continue;
            int best = -1;
            for (int e : l.cpu_hits) {
                bool filling = false;   // (a recent slot's fill brings it: the swaps leave it be)
                for (int i = 0; i < l.rc_n; ++i) filling = filling || l.rc_fill[(size_t) i] == e;
                if (!filling && l.slot[(size_t) e] < 0 && (best < 0 || l.heat[(size_t) e] > l.heat[(size_t) best])) best = e;
            }
            l.cpu_hits.clear();
            if (best < 0 || l.spare_vram < 0 || l.swap_pending || sw_inflight >= c_inflight) continue;
            int cold = -1;
            for (int e = 0; e < l.n_exp; ++e) {
                const int sl = l.slot[(size_t) e];
                if (sl >= 0 && !(l.rc_n > 0 && sl >= l.rc_base && sl < l.rc_base + l.rc_n) &&
                    (cold < 0 || l.heat[(size_t) e] < l.heat[(size_t) cold]))
                    cold = e;
            }
            if (cold < 0 || !(l.heat[(size_t) best] > l.heat[(size_t) cold] * c_margin + c_bonus)) continue;
            l.swap_pending = true;
            l.sw_in = best;
            ++sw_inflight;
            std::lock_guard<std::mutex> lk(sw_mu);
            sw_queue.push_back({l.il, best, cold, win_now.load(std::memory_order_relaxed)});
            queued = true;
        }
        if (queued) sw_cv.notify_all();
    }

    // the decode path's host side: each published routing in turn - its CPU lane experts (if any) into the answer's
    // part, then the answer's seq.  Spins while requests come (a token's 40 layers), naps after 2 ms without one.
    // DS4_ROUTE_TRACE=FILE: the CPU lane's requests (a window's routing, layer by layer) for offline cache studies -
    // first each trunk layer's experts in VRAM ('P' layer n e..), then a record a request ('R' layer T k, T x k picks,
    // T x k in-VRAM flags), all int32 / uint8, little-endian
    std::FILE* rtrace = nullptr;
    void rtrace_open() {
        const char* f = std::getenv("DS4_ROUTE_TRACE");
        if (f == nullptr || rtrace != nullptr) return;
        rtrace = std::fopen(f, "wb");
        if (rtrace == nullptr) return;
        for (const Layer& l : L) {
            if (l.mtp) continue;
            std::vector<int32_t> v;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] >= 0) v.push_back(e);
            const int32_t hdr[3] = {'P', l.il, (int32_t) v.size()};
            std::fwrite(hdr, 4, 3, rtrace);
            std::fwrite(v.data(), 4, v.size(), rtrace);
        }
    }
    void serve_lane() {
        unsigned next = 1;
        rtrace_open();
        auto last = std::chrono::steady_clock::now();
        std::vector<Ent> ents;
        while (!quit.load(std::memory_order_relaxed)) {
            Req& r = reqs[next % kRing];
            if (*(volatile unsigned*) &r.seq != next) {
                cpu_relax();
                if (std::chrono::steady_clock::now() - last > std::chrono::milliseconds(2))
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const int li = layer_of[next % kRing].load(std::memory_order_acquire);
            Resp& rp = resps[next % kRing];
            const int T = std::max(1, std::min(kWin, r.T));
            for (int t = 0; t < T; ++t) rp.cpu[t] = 0;
            if (li >= 0 && li < (int) L.size()) {
                Layer& l = L[(size_t) li];
                uint32_t* pr = prof.data() + (size_t) li * g.n_exp;
                ents.clear();
                // the window's CPU-lane picks by expert (up to 8 tokens an entry), their weights by entry row
                float wv[kWin * kPicks];
                int tok_of[kWin * kPicks];
                int nslots = 0;
                std::vector<std::pair<int, int>> picks;   // (expert, t * kPicks + i), CPU-lane only
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < l.k_exp; ++i) {
                        const int e = r.ids[t][i];
                        if (e < 0 || e >= l.n_exp) continue;
                        ++pr[e];
                        l.heat[(size_t) e] += 1.0f;
                        if (l.slot[(size_t) e] < 0) {
                            l.cpu_hits.push_back(e);
                            picks.push_back({e, t * kPicks + i});
                        } else if (l.rc_n > 0 && l.slot[(size_t) e] >= l.rc_base && l.slot[(size_t) e] < l.rc_base + l.rc_n)
                            l.rc_stamp[(size_t) (l.slot[(size_t) e] - l.rc_base)] = rc_clock.load(std::memory_order_relaxed);
                    }
                if (rtrace != nullptr) {
                    const int32_t hdr[4] = {'R', li, T, l.k_exp};
                    std::fwrite(hdr, 4, 4, rtrace);
                    std::vector<int32_t> ids;
                    std::vector<uint8_t> in;
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < l.k_exp; ++i) {
                            const int e = r.ids[t][i];
                            ids.push_back(e);
                            in.push_back(e >= 0 && e < l.n_exp && l.slot[(size_t) e] >= 0 ? 1 : 0);
                        }
                    std::fwrite(ids.data(), 4, ids.size(), rtrace);
                    std::fwrite(in.data(), 1, in.size(), rtrace);
                }
                std::sort(picks.begin(), picks.end());
                for (size_t a = 0; a < picks.size();) {
                    Ent en;
                    en.e = picks[a].first;
                    en.slot0 = nslots;
                    en.n = 0;
                    while (a < picks.size() && picks[a].first == en.e && en.n < kc::kExl3CpuMaxN) {
                        const int t = picks[a].second / kPicks, i = picks[a].second % kPicks;
                        en.tok[en.n++] = t;
                        tok_of[nslots] = t;
                        wv[nslots] = r.w[t][i];
                        ++nslots;
                        ++a;
                    }
                    ents.push_back(en);
                }
                if (!ents.empty()) {
                    const double tc = now_ms();
                    cpu_experts(l, &r.x[0][0], ents, (size_t) kMaxDim);
                    const int D = g.dim;
                    for (int t = 0; t < T; ++t) {
                        bool any = false;
                        for (int sl = 0; sl < nslots; ++sl) any = any || tok_of[sl] == t;
                        if (!any) continue;
                        float* o = rp.part[t];
                        for (int e = 0; e < D; ++e) o[e] = 0.0f;
                        for (int sl = 0; sl < nslots; ++sl) {
                            if (tok_of[sl] != t) continue;
                            const float* y = c_yd.data() + (size_t) sl * D;
                            for (int e = 0; e < D; ++e) o[e] += wv[sl] * y[e];
                        }
                        rp.cpu[t] = 1;
                    }
                    st.cpu_ms += now_ms() - tc;
                    st.cpu_experts += nslots;
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile unsigned*) &rp.seq = next;
            ++next;
            last = std::chrono::steady_clock::now();
        }
    }

    bool mv(Dev& d, std::initializer_list<x3::MvJob> jobs, std::string& err) {
        x3::MvJob j[x3::kMaxMvJobs];
        int n = 0;
        for (const x3::MvJob& x : jobs) j[n++] = x;
        if (!x3::mv(j, n, d.st)) {
            err = "exl3 mv failed";
            return false;
        }
        return true;
    }
    bool mv_n(Dev& d, x3::MvJob* j, int n, std::string& err) {
        for (int a = 0; a < n; a += x3::kMaxMvJobs)
            if (!x3::mv(j + a, std::min(x3::kMaxMvJobs, n - a), d.st)) {
                err = "exl3 mv failed";
                return false;
            }
        return true;
    }
    // y[c] = x[c] W for the T activations x + c * x_cs (floats), into y + c * y_cs
    static x3::MvJob job(const x3::Mat& m, const float* x, float* y, int T = 1, size_t x_cs = 0, size_t y_cs = 0,
                         float alpha = 1.0f) {
        x3::MvJob j;
        j.m = &m;
        j.xf = x;
        j.y = y;
        j.alpha = alpha;
        j.ncol = T;
        j.xf_cs = x_cs;
        j.y_cs = y_cs;
        return j;
    }

    // ---- the DSpark drafter (DeepSeek's DSparkBlock / forward_spec).  Its blocks' rings hold the trunk's main KV:
    // seed() turns a window's taps (the streams' mean entering layers 37..39) into it, through main_proj / main_norm
    // and each block's wkv / kv_norm / rope / FP8, for the positions the trunk kept.  draft() runs the block [y, noise x
    // (block - 1)] at positions pos .. through the blocks (each query attending to the ring and to the whole block),
    // the trunk's head, the markov bias token by token (greedy) and the confidence head.
    bool seed_drafter(int64_t p0, int n, int row0, std::string& err) {
        if (g.n_mtp == 0 || n <= 0) return true;
        if (n > kWin * kPicks) {
            err = "seeding the drafter with " + std::to_string(n) + " rows";
            return false;
        }
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        const int nt = (int) g.d_targets.size(), D = g.dim, HD = g.hd;
        for (int a = 0; a < n; a += kWin) {   // (a chunk's last 128 rows: 16 a GEMV)
            const int na = std::min(kWin, n - a);
            CK(cudaMemcpyAsync(d.xe, h_taps + (size_t) (row0 + a) * nt * D, (size_t) na * nt * D * 4,
                               cudaMemcpyHostToDevice, d.st));
            if (!mv(d, {job(d_main_proj, d.xe, d.ed + (size_t) a * D, na, (size_t) nt * D, D)}, err)) return false;
        }
        CK(cudaEventRecord(ev_seed, d.st));
        Finish fm;
        fm.w = d_main_norm;
        fm.eps = g.eps;
        finish(d.ed, n, D, D, fm, d.st);
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (l.dev != d_dev) {
                err = "the DSpark blocks are split across devices";
                return false;
            }
            for (int a = 0; a < n; a += kWin) {
                const int na = std::min(kWin, n - a);
                if (!mv(d, {job(l.wkv, d.ed + (size_t) a * D, d.kvr, na, D, HD)}, err)) return false;
                Finish fk;
                fk.w = l.kv_norm;
                fk.eps = g.eps;
                fk.rd = g.rd;
                fk.pos = (int) (p0 + a);
                fk.freqs = d.freqs_win;
                fk.quant = 1;
                fk.dst = d.ext + (size_t) g.win * HD;
                fk.dst_stride = HD;
                finish(d.kvr, na, HD, HD, fk, d.st);
                ext_to_ring(d.ext, l.win, p0 + a, g.win, na, HD, d.st);
            }
        }
        return true;
    }

    struct Samp {   // a request's sampling, and the draw index of the token after the drafted one
        float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
        int top_k = 64;
        uint64_t seed = 0, counter = 0;
        strata::kernels::SamplerParams at(uint64_t k) const {   // (the chain for draw counter + k)
            strata::kernels::SamplerParams sp;
            sp.temperature = temperature;
            sp.greedy = temperature <= 0.0f;
            sp.top_p = top_p;
            sp.top_k = top_k;
            sp.min_p = min_p;
            sp.seed = seed;
            sp.counter = counter + k;
            return sp;
        }
    };
    bool draft(int y, const Samp& sm, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
        const double t0 = now_ms();
        const bool ok = draft_(y, sm, out, conf, err);
        st.draft_ms += now_ms() - t0;
        ++st.drafts_rounds;
        return ok;
    }
    bool draft_(int y, const Samp& sm, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
        out.clear();
        conf.clear();
        spec_lists = false;
        const bool sampled = sm.temperature > 0.0f && draft_sample;
        const int T = g.d_block, D = g.dim, H = g.hc, V = g.vocab, R = g.d_rank;
        const int64_t p0 = pos;
        if (g.n_mtp == 0 || T < 1 || T > kWin || p0 + T > o.max_context) return true;
        int tk[kWin];
        tk[0] = y;
        for (int t = 1; t < T; ++t) tk[t] = g.d_noise;
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        for (int t = 0; t < T; ++t) std::memcpy(h_emb + (size_t) t * D, embed_pinned + (size_t) tk[t] * D, (size_t) D * 2);
        CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
        embed(d.emb_rows, T, d.h, D, H, d.st);
        init_pre(d.pre, T, d.st);
        WinPlan dp;   // (window attention alone: the mailbox seqs)
        dp.async = !sync_decode && T <= async_max_t;
        for (int i = 0; i < g.n_mtp && dp.async; ++i) {
            const int il = g.nl + i;
            dp.seq[il] = ++seq_issued;
            layer_of[dp.seq[il] % kRing].store(il, std::memory_order_release);
        }
        const double tq0 = now_ms();
        for (int i = 0; i < g.n_mtp; ++i)
            if (!layer(L[(size_t) (g.nl + i)], 0, p0, T, dp, err)) return false;
        const double tq1 = now_ms();
        cudaSetDevice(d.id);
        hc_pre_norm(d.h, d.pre, nullptr, D, H, T, g.eps, d.x, d.st);   // (the confidence head reads it before the norm)
        Dev& hd = devs[(size_t) head_dev];
        // (through the host to the head's device, and to the head group's other members - the drafter's device can be
        // the head group's lead itself, a group of every GPU)
        if (d_dev != head_dev || groups[(size_t) hd.grp].mem.size() > 1) {
            CK(cudaMemcpyAsync(d.bounce, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, d.st));
            CK(cudaEventRecord(d.ev_out, d.st));
        }
        if (d_dev != head_dev) {
            cudaSetDevice(hd.id);
            CK(cudaStreamWaitEvent(hd.st, d.ev_out, 0));
            CK(cudaMemcpyAsync(hd.x, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, hd.st));
        }
        // the drafter's final norm and the trunk's head, on every member of the head group (the others' shares first)
        const std::vector<int>& hg = groups[(size_t) hd.grp].mem;
        for (int m = (int) hg.size() - 1; m >= 0; --m) {
            Dev& e = devs[(size_t) hg[(size_t) m]];
            cudaSetDevice(e.id);
            if (m > 0) {
                CK(cudaStreamWaitEvent(e.st, d.ev_out, 0));
                CK(cudaMemcpyAsync(e.ao, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, e.st));
            } else
                CK(cudaMemcpyAsync(e.ao, e.x, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, e.st));
            Finish fn;
            fn.w = e.d_norm;
            fn.eps = g.eps;
            finish(e.ao, T, D, D, fn, e.st);
            if (!head_share(e, m, e.ao, T, err, !mk_split)) return false;
        }
        const double tq2 = now_ms();
        h_idx[0] = y;
        if (mk_split) {
            // the markov steps over the group: each member its rows of the bias and its share's argmax, every member
            // then the global one (the sampler's tie rule) and its E row - the same drafts as the lead's whole rows,
            // each step ~1/nm of their 66 MB and no gather of the shares
            const int nm = (int) hg.size();
            const unsigned seq = ++mk_seq;
            for (int m = 0; m < nm; ++m) {
                Dev& e = devs[(size_t) hg[(size_t) m]];
                cudaSetDevice(e.id);
                CK(cudaMemcpyAsync(e.mk_ints, h_idx, 4, cudaMemcpyHostToDevice, e.st));
                CK(cudaMemcpyAsync(e.mk_erow, h_mk_embed + (size_t) y * R, (size_t) R * 2, cudaMemcpyHostToDevice, e.st));
            }
            for (int i = 0; i < T; ++i)
                for (int m = 0; m < nm; ++m) {
                    Dev& e = devs[(size_t) hg[(size_t) m]];
                    cudaSetDevice(e.id);
                    float* row = m == 0 ? hd.logits + (size_t) i * V : e.logits_part + (size_t) i * e.mk_vn;
                    markov_share(row, e.mk_share, e.mk_erow, e.mk_vn, R, e.st);
                    if (sampled) {   // (each member its share's top k, every member the same draw of the merged row)
                        SpecChain ch;
                        ch.k = sm.top_k > 0 && sm.top_k < 64 ? sm.top_k : 64;
                        ch.top_p = sm.top_p;
                        ch.min_p = sm.min_p;
                        ch.temperature = sm.temperature;
                        ch.seed = sm.seed;
                        ch.counter = sm.counter + (uint64_t) i;
                        unsigned* lists = mk_lists + (size_t) i * nm * kSpecList;
                        topk_share(row, e.mk_vn, e.mk_v0, ch.k, lists + (size_t) m * kSpecList, seq, e.st);
                        spec_pick(lists, nm, kSpecList, seq, ch, e.mk_ints + i + 1, h_mk_embed, e.mk_erow, R,
                                  m == 0 ? d_spec + i : nullptr, e.st);
                        continue;
                    }
                    argmax_share(row, e.mk_vn, e.mk_v0, mk_slots + ((size_t) i * nm + m) * 4, seq, e.st);
                    markov_pick(mk_slots + (size_t) i * nm * 4, nm, seq, e.mk_ints + i + 1, h_mk_embed, e.mk_erow, R,
                                e.st);
                }
            cudaSetDevice(hd.id);
            spec_lists = sampled;
        } else {
            cudaSetDevice(hd.id);
            CK(cudaMemcpyAsync(d_ints, h_idx, 4, cudaMemcpyHostToDevice, hd.st));
            strata::kernels::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            // the markov bias of the token before, then that position's draft: greedy, or (sampled) drawn through the
            // request's chain with its kept list - the token draw counter + i will be
            for (int i = 0; i < T; ++i) {
                markov_add(hd.logits + (size_t) i * V, d_mk_embed, d_mk_head, d_ints + i, V, R, hd.st);
                if (sampled)
                    strata::kernels::sample_draft(hd.logits + (size_t) i * V, 1, V, sm.at((uint64_t) i), d_spec + i,
                                                  d_ints + i + 1, hd.st);
                else
                    strata::kernels::sample_tokens(hd.logits + (size_t) i * V, 1, V, nullptr, 0, sp, d_ints + i + 1, hd.st);
            }
            spec_lists = sampled;
        }
        confidence(hd.x, d_mk_embed, d_ints, d_conf, T, D, R, d_confv, hd.st);
        int tok[kWin + 1];
        float cf[kWin];
        CK(cudaMemcpyAsync(h_idx, d_ints, (size_t) (T + 1) * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaMemcpyAsync(h_w, d_confv, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
        const double tq3 = now_ms();
        CK(cudaStreamSynchronize(hd.st));
        if (draft_times) {
            const double tq4 = now_ms();
            dt_acc[0] += tq1 - tq0, dt_acc[1] += tq2 - tq1, dt_acc[2] += tq3 - tq2, dt_acc[3] += tq4 - tq3;
            if (++dt_n % 64 == 0)
                std::fprintf(stderr, "draft host ms (%lld drafts): blocks %.3f, head %.3f, markov steps %.3f, wait %.3f\n", dt_n,
                             dt_acc[0] / dt_n, dt_acc[1] / dt_n, dt_acc[2] / dt_n, dt_acc[3] / dt_n);
        }
        for (int i = 0; i <= T; ++i) tok[i] = h_idx[i];
        for (int i = 0; i < T; ++i) cf[i] = h_w[i];
        for (int i = 0; i < T; ++i) {
            out.push_back(tok[i + 1]);
            conf.push_back(cf[i]);
        }
        return true;
    }

    int sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                   const float* bias) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::kernels::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        sp.logit_bias = bias;
        strata::kernels::sample_tokens(d.logits + (size_t) row * g.vocab, 1, g.vocab, nullptr, 0, sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    // [y, drafts] at positions pos ..: the trunk's verdict, its next token, and the state back to what it kept.
    // Greedy: the drafts the trunk's argmax agrees with.  Sampled: speculative sampling over the request's chain - each
    // draft kept with probability min(1, p/q) (q its kept list: the drafter's, or a point at a greedy draft), the first
    // one rejected replaced by a draw from max(0, p - q), all kept: the next row's plain draw (draw counter + i for row
    // i) - so each token is distributed as the trunk's own draw.
    bool verify(int y, const std::vector<int>& drafts, const Samp& sm, int& accepted, int& next, std::string& err) {
        const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
        const int T = n + 1, V = g.vocab;
        const int64_t p0 = pos;
        int tk[kWin];
        tk[0] = y;
        for (int i = 0; i < n; ++i) tk[i + 1] = drafts[(size_t) i];
        spec_win = true;
        const double tv = now_ms();
        const bool ok = forward(tk, T, T, err, 0);
        st.verify_ms += now_ms() - tv;
        spec_win = false;
        if (!ok) return false;
        Dev& hd = devs[(size_t) head_dev];
        cudaSetDevice(hd.id);
        int m = 0;
        if (sm.temperature <= 0.0f) {
            strata::kernels::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            strata::kernels::sample_tokens(hd.logits, T, V, nullptr, 0, sp, d_ints + 16, hd.st);   // (the T rows at once)
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = h_idx[m];
        } else {
            for (int i = 0; i < n; ++i) h_idx[i] = drafts[(size_t) i];
            CK(cudaMemcpyAsync(d_ints + 32, h_idx, (size_t) n * 4, cudaMemcpyHostToDevice, hd.st));
            if (!spec_lists) {   // greedy drafts: q a point at each
                for (int i = 0; i < n; ++i) {
                    h_spec[i].n = 1;
                    h_spec[i].ids[0] = drafts[(size_t) i];
                    h_spec[i].p[0] = 1.0;
                }
                CK(cudaMemcpyAsync(d_spec, h_spec, (size_t) n * sizeof(*h_spec), cudaMemcpyHostToDevice, hd.st));
            }
            if (spec_diag && spec_lists && n <= 8) {   // (p's kept lists beside q's: the trunk's rows through the chain)
                strata::kernels::sample_draft(hd.logits, n, V, sm.at(0), d_spec + 8, d_ints + 16, hd.st);
                CK(cudaMemcpyAsync(h_spec, d_spec, (size_t) (8 + n) * sizeof(*h_spec), cudaMemcpyDeviceToHost, hd.st));
                CK(cudaStreamSynchronize(hd.st));
                for (int i = 0; i < n; ++i) {
                    const strata::kernels::SpecList &q = h_spec[i], &pp = h_spec[8 + i];
                    double sg = 0.0, ss = 0.0;
                    for (int a = 0; a < pp.n; ++a) {
                        if (pp.ids[a] == q.ids[0]) sg = pp.p[a];
                        for (int b = 0; b < q.n; ++b)
                            if (q.ids[b] == pp.ids[a]) ss += std::min(pp.p[a], q.p[b]);
                    }
                    diag_g[i] += sg;
                    diag_s[i] += ss;
                    ++diag_n[i];
                }
                if (++diag_rounds % 64 == 0) {
                    std::fprintf(stderr, "spec diag after %lld rounds (position: rounds, greedy p(argmax q), drawn sum min(p,q)):",
                                 diag_rounds);
                    for (int i = 0; i < kWin && diag_n[i] > 0; ++i)
                        std::fprintf(stderr, " %d: %lld %.3f %.3f", i, diag_n[i], diag_g[i] / diag_n[i], diag_s[i] / diag_n[i]);
                    std::fprintf(stderr, "\n");
                }
            }
            strata::kernels::sample_verify(hd.logits, n, V, sm.at(0), d_spec, d_ints + 32, d_ints + 48, hd.st);
            CK(cudaMemcpyAsync(h_idx, d_ints + 48, (size_t) n * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            if (m < n) next = h_idx[m];
            else next = sample_row(n, sm.temperature, sm.top_p, sm.top_k, sm.min_p, sm.seed, sm.counter + (uint64_t) n,
                                   nullptr);
        }
        spec_lists = false;
        rollback(p0, T, m + 1);
        if (!seed_drafter(p0, m + 1, 0, err)) return false;
        accepted = m;
        st.drafts_offered += n;
        st.drafts_accepted += m;
        return true;
    }

    // a verified window keeps its first `keep` positions: the ring rows and the compressors' state of the rest undone
    // (the compressed KV and the index keys past them are written again when those positions come)
    void rollback(int64_t p0, int T, int keep) {
        if (keep >= T) return;
        // the ring rows: one launch a device for all its layers (a launch a layer and member was ~300 on 8-way tensor
        // parallel, a millisecond of the host's before the next draft)
        std::vector<RingBatch> rb(devs.size());
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                RingBatch& b = rb[(size_t) l.dev];
                if (b.n == kMaxRingBatch) {   // (more layers on a device than a batch takes: this one alone)
                    cudaSetDevice(devs[(size_t) l.dev].id);
                    ring_restore(l.win, l.ring_bak, p0, g.win, keep, T, g.hd, devs[(size_t) l.dev].st);
                    continue;
                }
                b.ring[b.n] = l.win;
                b.bak[b.n++] = l.ring_bak;
            }
        for (size_t di = 0; di < devs.size(); ++di)
            if (rb[di].n > 0) {
                cudaSetDevice(devs[di].id);
                ring_restore_batch(rb[di], p0, g.win, keep, T, g.hd, devs[di].st);
            }
        // the compressors' state: the same, one launch a device (two copies and a launch a layer and member were
        // ~0.3 ms of the host's before the next draft)
        std::vector<CompBatch> cb(devs.size());
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                Dev& d = devs[(size_t) l.dev];
                if (l.c_snap == nullptr) continue;
                CompBatch& b = cb[(size_t) l.dev];
                if (b.n == kMaxRingBatch) {   // (more than a batch takes: this one alone)
                    cudaSetDevice(d.id);
                    const size_t ns = (size_t) l.ratio * g.hd * 4;
                    cudaMemcpyAsync(l.c_kv, l.c_snap, ns, cudaMemcpyDeviceToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, l.c_snap + (size_t) l.ratio * g.hd, ns, cudaMemcpyDeviceToDevice, d.st);
                    compress_win(l.c_rows, l.c_rows + (size_t) kWin * g.hd, l.c_kv, l.c_score, l.ratio, (int) p0, keep,
                                 g.hd, l.c_norm, g.eps, d.latent, d.st);
                    continue;
                }
                b.kv_state[b.n] = l.c_kv;
                b.score_state[b.n] = l.c_score;
                b.snap[b.n] = l.c_snap;
                b.kv[b.n] = l.c_rows;
                b.score[b.n] = l.c_rows + (size_t) kWin * g.hd;
                b.ratio[b.n++] = l.ratio;
            }
        for (size_t di = 0; di < devs.size(); ++di)
            if (cb[di].n > 0) {
                cudaSetDevice(devs[di].id);
                comp_restore_batch(cb[di], p0, keep, g.hd, devs[di].st);
            }
        pos = p0 + keep;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
    }

    // ---- a window: T tokens at positions pos .. pos + T - 1 through every layer; the logits of its last `logits` rows
    bool step(int token, bool want_logits, std::string& err) { return forward(&token, 1, want_logits ? 1 : 0, err); }


    // layer L0 for a chunk of T tokens at p0 as group member mi: layer()'s arithmetic over the chunk's rows (the
    // projections as GEMMs, the attention and the indexer in slabs, the routed experts as sorted rows: the VRAM ones in
    // place, the CPU lane's streamed in batches).  src_only: the KV source's part alone - its compressed KV and index
    // keys from the attention's input (YOCO's first consumer layer, a chunk before a truncated run's tail); lo: the
    // window's floor (a truncated run's consumer layers: its tail's start)
    // a chunk's sum over its group: three members or more through the copy engines (allreduce_dma, c.y its scratch:
    // free between the MoE's sets), unless DS4_AR_DMA=0
    void chunk_allreduce(Dev& d, Chunk& c, const ArGroup& ab, int mi, int n, cudaStream_t s) {
        chunk_allreduce_on(d, c, ab, mi, c.acc, n, s, c.ar_aux);
    }
    void chunk_allreduce_on(Dev& d, Chunk& c, const ArGroup& ab, int mi, float* buf, int n, cudaStream_t s,
                            ArAux& aux) {
        static const bool dma = [] {
            const char* e = std::getenv("DS4_AR_DMA");
            return e == nullptr || std::atoi(e) != 0;
        }();
        HostReducer* hr = groups[(size_t) d.grp].hred.get();
        if (dma && hr != nullptr) {   // (the lead submits the sum before its members' copies come)
            const unsigned seq = ++c.arb_seq;
            if (mi == 0) hr->submit(seq, n);
            allreduce_host(buf, n, ab, *hr, mi, seq, c.y, s, aux);
            return;
        }
        if (dma && ab.nm > 2 && ab.sums != nullptr && ab.bflags != nullptr) allreduce_dma(buf, n, ab, mi, ++c.arb_seq, c.y, s, &aux);
        else allreduce(buf, n, ab, mi, ++c.arb_seq, d.xcnt, s);
    }

    // the MoE's sum owed by the layer before (Chunk::pend): range r's rows [r0, r1) once its sum is in - the residual
    // and pre, from that layer's f_post / f_comb / f_pre
    void pend_range(Chunk& c, int r, int& r0, int& r1) const {
        r0 = (int) ((int64_t) c.pend_T * r / Chunk::kPendRanges);
        r1 = (int) ((int64_t) c.pend_T * (r + 1) / Chunk::kPendRanges);
    }
    void pend_take(Chunk& c, int r, cudaStream_t s) {
        const int D = g.dim, H = g.hc;
        int r0, r1;
        pend_range(c, r, r0, r1);
        cudaStreamWaitEvent(s, c.pend_ev[r], 0);
        if (r1 <= r0) return;
        hc_post_inplace(c.acc + (size_t) r0 * D, c.h + (size_t) r0 * H * D, c.f_post + (size_t) r0 * 8,
                        c.f_comb + (size_t) r0 * 64, D, H, r1 - r0, s);
        cudaMemcpyAsync(c.pre + (size_t) r0 * 8, c.f_pre + (size_t) r0 * 8, (size_t) (r1 - r0) * 8 * 4,
                        cudaMemcpyDeviceToDevice, s);
    }
    void pend_finish(Chunk& c, cudaStream_t s) {
        if (!c.pend) return;
        for (int r = 0; r < Chunk::kPendRanges; ++r) pend_take(c, r, s);
        c.pend = false;
    }
    bool layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                     bool defer = false, bool src_only = false, int64_t lo = 0) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        Chunk& c = *chunks[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ab = groups[(size_t) L0.grp].arb;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        // the residual the layer before owes: by ranges beside the attention's first, row-wise steps below, or (an
        // Engram layer: its gate reads the whole residual first) whole now
        const bool by_range = c.pend && l.eg < 0 && c.pend_T == T && !src_only;
        if (!by_range) pend_finish(c, s);
        // Engram, in slabs of 256 tokens
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            const int slot = wp.k % kEgSlots;   // (read ahead by eg_reader)
            if (eg_read.load(std::memory_order_acquire) <= wp.k) {
                const double te = now_ms();
                while (eg_read.load(std::memory_order_acquire) <= wp.k)
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                if (mi == 0) st.engram_ms += now_ms() - te;
            }
            uint8_t* host = eg_shared[slot] + (size_t) l.eg * chunk * per;
            CK(cudaMemcpyAsync(c.eg_rows, host, (size_t) T * per, cudaMemcpyHostToDevice, s));
            CK(cudaEventRecord(c.eg_ev[slot], s));
            eg_used[slot].fetch_add(1, std::memory_order_acq_rel);
            for (int s0 = 0; s0 < T; s0 += 256) {
                const int ns = std::min(256, T - s0);
                engram_dequant_tok(c.eg_rows + (size_t) s0 * per, per, cols, ns, ehd, c.eg_x, s);
                x3::rows_f16(c.eg_x, cols * ehd, nullptr, ns, cols * ehd, c.eg_x16, s);
                if (!gemm(d, c, l.eg_wkv, c.eg_x16, cols * ehd, ns, c.eg_kv, (H + 1) * D, 0.0f, err)) return false;
                engram_gate(c.h + (size_t) s0 * H * D, c.eg_kv, l.eg_w, H, D, ns, g.eps, s);
            }
        }
        // attention: the row-wise first steps (rows [r0, r0 + n))
        auto first_steps = [&](int r0, int n) -> bool {
            const size_t a = (size_t) r0;
            if (!src_only) {
                if (!hc_mix_chunk(d, c, c.h + a * H * D, l.hca_fn, n, c.mix + a * 32, err)) return false;
                hc_split(c.mix + a * 32, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, n, c.a_pre + a * 8,
                         c.a_post + a * 8, c.a_comb + a * 64, s);
            }
            hc_pre_norm(c.h + a * H * D, c.pre + a * 8, l.attn_norm, D, H, n, g.eps, c.x + a * D, s);
            x3::rows_f16(c.x + a * D, D, nullptr, n, D, c.x16 + a * D, s);
            if (!src_only && (!gemm(d, c, l.wq_a, c.x16 + a * D, D, n, c.qr + a * g.q_lora, g.q_lora, 0.0f, err) ||
                              !gemm(d, c, l.wkv, c.x16 + a * D, D, n, c.kvr + a * HD, HD, 0.0f, err)))
                return false;
            if (l.kv_src && !gemm(d, c, l.c_wkv, c.x16 + a * D, D, n, c.ckr + a * HD, HD, 0.0f, err)) return false;
            if (l.kv_src && l.ratio > 1 && !gemm(d, c, l.c_wgate, c.x16 + a * D, D, n, c.cgr + a * HD, HD, 0.0f, err))
                return false;
            if (src_only) return true;
            Finish fq;
            fq.w = l.q_norm;
            fq.eps = g.eps;
            finish(c.qr + a * g.q_lora, n, g.q_lora, g.q_lora, fq, s);
            x3::rows_f16(c.qr + a * g.q_lora, g.q_lora, nullptr, n, g.q_lora, c.qr16 + a * g.q_lora, s);
            return true;
        };
        if (by_range) {   // each range once its sum is in (the later ones' sums beside these steps)
            static const bool serial = std::getenv("DS4_DEFER_SERIAL") != nullptr;   // (a test: every sum first)
            if (serial)
                for (int r = 0; r < Chunk::kPendRanges; ++r) cudaStreamWaitEvent(s, c.pend_ev[r], 0);
            for (int r = 0; r < Chunk::kPendRanges; ++r) {
                int r0, r1;
                pend_range(c, r, r0, r1);
                pend_take(c, r, s);
                if (r1 > r0 && !first_steps(r0, r1 - r0)) return false;
            }
            c.pend = false;
        } else if (!first_steps(0, T))
            return false;
        // the window rows: the ring's (positions p0 - win ..) then the chunk's
        if (!src_only) {
            ring_to_ext(l.win, c.ext, p0, g.win, HD, s);
            Finish fk;
            fk.w = l.kv_norm;
            fk.eps = g.eps;
            fk.rd = g.rd;
            fk.pos = (int) p0;
            fk.freqs = freqs;
            fk.quant = 1;
            fk.dst = c.ext + (size_t) g.win * HD;
            fk.dst_stride = HD;
            finish(c.kvr, T, HD, HD, fk, s);
            ext_to_ring(c.ext, l.win, p0, g.win, T, HD, s, nullptr);
        }
        const uint8_t* ckv = nullptr;   // (the source's packed rows)
        const long long ss = (long long) o.max_context + 64;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            const int64_t g_first = p0 / R, g_end = (p0 + T) / R;
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            const float* latent = c.latent;
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(c.ckr, T, HD, HD, fc, s);
                    latent = c.ckr;
                } else
                    compress_win(c.ckr, c.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, c.latent, s);
            }
            const int clen_max = (int) ((p0 + T) / R);
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    x3::rows_f16(latent, HD, nullptr, ng, HD, c.lat16, s);
                    if (!gemm(d, c, l.i_wk, c.lat16, HD, ng, c.ikr, g.ix_hd, 0.0f, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_row();
                    fi.pdst_stride = (long long) ik_row();
                    finish(c.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                }
                if (clen_max > 0 && !src_only) {
                    if (!gemm_f16w(c, l.i_wproj, g.ix_heads, D, c.x16, D, T,
                                   1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), c.iw, 64, err))
                        return false;
                    for (int a0 = 0; a0 < T; a0 += c.AI) {
                        const int na = std::min(c.AI, T - a0);
                        const int64_t pa = p0 + a0;
                        if (!gemm(d, c, l.i_wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.iq,
                                  g.ix_heads * g.ix_hd, 0.0f, err))
                            return false;
                        Finish fi;
                        fi.rd = g.rd;
                        fi.pos = (int) pa;
                        fi.vpp = g.ix_heads;
                        fi.freqs = freqs;
                        fi.quant = 3;
                        finish(c.iq, na * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                        Rows lim;
                        lim.gen = 1;
                        lim.p0 = pa;
                        lim.R = R;
                        lim.kcap = g.ix_topk;
                        lim.ccap = g.cand_topk;
                        lim.blk = std::max(1, g.cand_blk);
                        const int nlast = (int) ((pa + na) / R);   // the slab's widest query
                        int* tk = c.topk + (size_t) a0 * topk_stride();
                        int* cd = c.cand + (size_t) a0 * cand_stride();
                        const float* iw = c.iw + (size_t) a0 * 64;
                        if (nlast <= 0) continue;
                        if (l.uses_cand && (nlast + g.cand_blk - 1) / g.cand_blk > g.cand_topk) {
                            Rows cl = lim;
                            cl.gen = 2;
                            index_score(c.iq, iw, nullptr, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, g.cand_topk * g.cand_blk,
                                        cd, (int) cand_stride(), g.cand_blk, cl, c.scores, ss, s);
                            Rows tr = lim;
                            tr.gen = 3;
                            topk(c.scores, ss, na, tr, cd, (int) cand_stride(), g.cand_blk, tk, (int) topk_stride(), s);
                        } else {
                            index_score(c.iq, iw, nullptr, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, nlast, nullptr, 0, 1,
                                        lim, c.scores, ss, s);
                            if (l.cand_src) {
                                Rows cr = lim;
                                cr.kcap = g.cand_topk;
                                cand_blocks(c.scores, ss, na, cr, g.cand_blk, c.bm, (long long) bm_stride(), cd,
                                            (int) cand_stride(), s);
                            }
                            topk(c.scores, ss, na, lim, nullptr, 0, 1, tk, (int) topk_stride(), s);
                        }
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * kv_row();
                fc.pdst_stride = (long long) kv_row();
                finish(const_cast<float*>(latent), ng, HD, HD, fc, s);
                if (mi == 0 && sr.remote) {   // the new rows for the other groups' readers (slot k % 2, once
                    const int slot = wp.k % 2;  // they took the chunk before last's)
                    if (wp.k >= 2)
                        for (int r : sr.readers) wait_flags(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, 1, 32, wp.win - 2, s);
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.stage(slot);
                    x.bytes[0] = (size_t) ng * kv_row();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_row();
                        x.dst[1] = sr.stage(slot) + x_cap() * HD * 2;
                        x.bytes[1] = (size_t) ng * ik_row();
                        x.n = 2;
                    }
                    xfer_put(x, sr.flag(slot), wp.win, d.xcnt, s);
                }
            }
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (src_only) return true;
        // the attention in slabs: q, the sparse attention, wo_a, wo_b (this member's share) into acc; with three
        // members or more each slab's sum on ar_s while the next slab computes (every member the same slabs: c.A
        // is the group's, a member without heads its zeros)
        static const bool overlap = [] {
            const char* e = std::getenv("DS4_AR_OVERLAP");
            return e == nullptr || std::atoi(e) != 0;
        }();
        const bool slab_ar = overlap && nm > 2 && c.ar_s != nullptr;
        if (NHL == 0 && !slab_ar) CK(cudaMemsetAsync(c.acc, 0, (size_t) T * D * sizeof(*c.acc), s));   // (no heads)
        for (int a0 = 0; a0 < T && (NHL > 0 || slab_ar); a0 += c.A) {
            const int na = std::min(c.A, T - a0);
            if (NHL == 0) {
                CK(cudaMemsetAsync(c.acc + (size_t) a0 * D, 0, (size_t) na * D * sizeof(*c.acc), s));
                CK(cudaEventRecord(c.ar_go, s));
                CK(cudaStreamWaitEvent(c.ar_s, c.ar_go, 0));
                chunk_allreduce_on(d, c, ab, mi, c.acc + (size_t) a0 * D, na * D, c.ar_s, c.ar_aux2);
                continue;
            }
            if (!gemm(d, c, l.wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.q, HQ, 0.0f, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) (p0 + a0);
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(c.q, na * NHL, HD, HD, fr, s);
            AttnRows ar;
            ar.gen = 1;
            ar.p0 = p0;
            ar.lo = lo;
            ar.t0 = a0;
            ar.win = g.win;
            ar.R = l.ratio;
            ar.kcap = g.ix_topk;
            sparse_attn(c.q, c.ext, nullptr, ckv, c.topk + (size_t) a0 * topk_stride(), (int) topk_stride(), ar, l.sink, NHL, HD,
                        na, 1.0f / std::sqrt((float) HD), c.o, s);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) (p0 + a0);
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(c.o, na * NHL, HD, HD, fo, s);
            x3::rows_f16(c.o, HQ, nullptr, na, HQ, c.o16, s);
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                if (!gemm(d, c, l.wo_a[gi], c.o16 + (size_t) gi * per, HQ, na, c.oa + (size_t) gi * g.o_lora,
                          OG * g.o_lora, 0.0f, err))
                    return false;
            x3::rows_f16(c.oa, OG * g.o_lora, nullptr, na, OG * g.o_lora, c.oa16, s);
            if (!gemm(d, c, l.wo_b, c.oa16, OG * g.o_lora, na, c.acc + (size_t) a0 * D, D, 0.0f, err)) return false;
            if (slab_ar) {
                CK(cudaEventRecord(c.ar_go, s));
                CK(cudaStreamWaitEvent(c.ar_s, c.ar_go, 0));
                chunk_allreduce_on(d, c, ab, mi, c.acc + (size_t) a0 * D, na * D, c.ar_s, c.ar_aux2);
            }
        }
        if (slab_ar) {   // (the residual and the MoE after every slab's sum)
            CK(cudaEventRecord(c.ar_done, c.ar_s));
            CK(cudaStreamWaitEvent(s, c.ar_done, 0));
        } else if (nm > 1)
            chunk_allreduce(d, c, ab, mi, T * D, s);
        hc_post_inplace(c.acc, c.h, c.a_post, c.a_comb, D, H, T, s);
        // the MoE
        if (!hc_mix_chunk(d, c, c.h, l.hcf_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, c.f_pre, c.f_post, c.f_comb, s);
        hc_pre_norm(c.h, c.a_pre, l.ffn_norm, D, H, T, g.eps, c.x, s);
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        if (!gemm_f16w(c, l.gate_w, l.n_exp, D, c.x16, D, T, 1.0f, c.glog, 1024, err)) return false;
        {
            VlBits vb;
            const bool vl = l.gate_bias_vl != nullptr && vl_bits(p0, T, vb);
            route(c.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, c.ridx, c.rw, s, l.gate_bias_vl,
                  vl ? &vb : nullptr);
        }
        const int n_vs = L0.n_slots + (int) L0.host_list.size();
        CK(cudaMemcpyAsync(l.vs_dev, L0.vs_pin, (size_t) l.n_exp * 4, cudaMemcpyHostToDevice, s));
        moe_sort(c.ridx, T, l.k_exp, l.vs_dev, n_vs, c.bounds, c.row_tok, c.pos, s);
        CK(cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (n_vs + 1) * 4, cudaMemcpyDeviceToHost, s));
        // the shared expert into acc while the host waits for the bounds
        if (!gemm(d, c, l.s1, c.x16, D, T, c.sg, FF, 0.0f, err) || !gemm(d, c, l.s3, c.x16, D, T, c.su, FF, 0.0f, err))
            return false;
        swiglu(c.sg, c.su, T * FF, g.lim, c.sg, s);
        x3::rows_f16(c.sg, FF, nullptr, T, FF, c.sh16, s);
        if (!gemm(d, c, l.s2, c.sh16, FF, T, c.acc, D, 0.0f, err)) return false;
        CK(cudaStreamSynchronize(s));
        std::vector<int> b(c.bounds_pin, c.bounds_pin + n_vs + 1);
        if (mi == 0) {   // the profile (the CPU lane's experts: by the plan's order)
            uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) {
                const int v = L0.vs_pin[e];
                pr[e] += (uint32_t) (b[(size_t) v + 1] - b[(size_t) v]);
            }
            st.experts += (int64_t) T * l.k_exp;
        }
        // the streamed experts as their batches arrive (each batch's slots freed for the copies at once), the VRAM ones
        // in shares between them (the device works while the copies run)
        const int di = l.dev, P = Chunk::kParts;
        int nb = 0;
        for (size_t q = S.kp; q < S.b.size() && S.b[q].il == L0.il && S.b[q].k == wp.k; ++q) ++nb;
        int v_done = 0;
        for (int bi = 0; bi < nb; ++bi) {
            const Batch& bt = S.b[S.kp];
            const int half = (int) (S.kp % P);
            CK(cudaStreamWaitEvent(s, c.filled[half], 0));
            const int v0 = L0.n_slots + bt.j0;
            if (!run_set(d, c, L0, l, true, c.ring + (size_t) half * (c.ring_n / P) * c.ring_stride, v0, c.ring_stride,
                         c.bounds, b.data(), v0, v0 + bt.n, T, err))
                return false;
            CK(cudaEventRecord(c.freed[half], s));
            ++S.kp;
            if (!copy_ahead(d, c, S, di, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (bi + 1) / nb);
            if (!run_set(d, c, L0, l, false, nullptr, 0, 0, c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, L0, l, false, nullptr, 0, 0, c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        if (defer && slab_ar) {   // the sum by ranges on ar_s; the next layer (here) takes each as it comes
            CK(cudaEventRecord(c.ar_go, s));
            CK(cudaStreamWaitEvent(c.ar_s, c.ar_go, 0));
            c.pend_T = T;
            for (int r = 0; r < Chunk::kPendRanges; ++r) {
                int r0, r1;
                pend_range(c, r, r0, r1);
                if (r1 > r0) chunk_allreduce_on(d, c, ab, mi, c.acc + (size_t) r0 * D, (r1 - r0) * D, c.ar_s, c.ar_aux2);
                CK(cudaEventRecord(c.pend_ev[r], c.ar_s));
            }
            c.pend = true;
            return true;
        }
        if (nm > 1) chunk_allreduce(d, c, ab, mi, T * D, s);
        hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
        CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        return true;
    }

    // the CPU lane's experts of every layer, in index order, and each expert's virtual slot (before a chunk)
    void plan_streams() {
        for (Layer& l : L) {
            if (l.mtp || l.vs_pin == nullptr) continue;
            l.host_list.clear();
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0) l.host_list.push_back(e);
            if (mixed(l))   // (class by class: a streamed batch's experts mostly of one layout, run_set's runs)
                std::stable_sort(l.host_list.begin(), l.host_list.end(),
                                 [&](int a, int b) { return l.ecls[(size_t) a] < l.ecls[(size_t) b]; });
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] >= 0) l.vs_pin[e] = l.slot[(size_t) e];
            for (size_t j = 0; j < l.host_list.size(); ++j) l.vs_pin[l.host_list[j]] = l.n_slots + (int) j;
        }
    }

    // a chunk's hand-off between groups: h | pre | topk | cand of a chunk of T rows
    Xfer chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {c.h, c.pre, c.topk, c.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    // a run of chunks (start positions p0[k], sizes T[k], seqs seq[k]) on device di: its layers chunk after chunk, the
    // groups pipelined (a group takes chunk k + 1 while the next takes chunk k: the hand-offs and the sources' rows
    // two deep, each slot reused once its readers acknowledged it), the streamed experts' copies running ahead across
    // layers and chunks
    struct Run {
        std::vector<int64_t> p0;
        std::vector<int> T;
        std::vector<unsigned> seq;
        // a truncated run (tail_rows): skip[k] - chunk k is before the tail (layer n_pre its KV source's part alone, the
        // layers after it nothing); lo - the tail's start, the consumer layers' window floor (0: not truncated)
        std::vector<char> skip;
        int64_t lo = 0;
        // chunk k at layer il: the whole layer, or any of it (the source's part)
        bool full(int k, int il, int n_pre) const { return !skip[(size_t) k] || il < n_pre; }
        bool any(int k, int il, int n_pre) const { return !skip[(size_t) k] || il <= n_pre; }
    };
    // topk's segments (TopkScratch): for a verify window's rows (a drafter's block and its token: 6; wider windows, a
    // prompt's 16, take topk_kernel), as many a row as the longest row needs - 0.5 MB a device at 256K
    static constexpr size_t kTksRows = 8;
    size_t tks_p() const {
        return (size_t) std::max(2, std::min(kTopkSegMax, (int) ((o.max_context + kTopkSeg - 1) / kTopkSeg)));
    }
    // the rows of a run of n that go through the consumer layers (Options::prefill_keep; n: all of them)
    int tail_rows(int n) const {
        if (n_pre < 0 || o.prefill_keep < 0 || logits_all || dump_l != nullptr) return n;
        const int R = o.prefill_keep > 0 ? std::max(o.prefill_keep, g.win) : (g.win - 1) * (g.nl - n_pre) + g.win;
        return std::min(n, R);
    }
    bool run_dev_chunks(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        Streamer S;
        const int B = std::max(1, c.ring_n / Chunk::kParts);
        for (int k = 0; k < nk; ++k)
            for (int il = 0; il < g.nl; ++il) {
                const Layer& L0 = L[(size_t) il];
                bool here = false;
                for (int m = 0; m < L0.nm; ++m) here = here || member(L0, m).dev == di;
                if (!here || !run.full(k, il, n_pre)) continue;   // (a skipped chunk's layer streams nothing)
                for (int j0 = 0; j0 < (int) L0.host_list.size(); j0 += B)
                    S.b.push_back({k, il, j0, std::min(B, (int) L0.host_list.size() - j0)});
            }
        if (!S.b.empty() && c.ring == nullptr) {
            err = "no slots to stream experts into on CUDA" + std::to_string(d.id);
            return false;
        }
        std::atomic<bool> ra_stop{false};
        std::thread ra;
        if (file_experts && !S.b.empty()) ra = std::thread([this, &S, di, &ra_stop] { read_ahead(S, di, ra_stop); });
        struct Join {   // (every way out of the run: the read-ahead ends before S does)
            std::atomic<bool>& stop;
            std::thread& th;
            ~Join() {
                stop.store(true);
                if (th.joinable()) th.join();
            }
        } ra_join{ra_stop, ra};
        if (!copy_ahead(d, c, S, di, err)) return false;
        for (int k = 0; k < nk; ++k) {
            const int64_t p0 = run.p0[(size_t) k];
            const int T = run.T[(size_t) k], slot = k % 2;
            const bool last_k = k + 1 == nk;
            WinPlan wp;
            wp.par = true;
            wp.async = true;
            wp.win = run.seq[(size_t) k];
            wp.k = k;
            bool started = false;
            std::vector<char> got(srcs.size(), 0);
            for (int il = 0; il < g.nl; ++il) {
                Layer& L0 = L[(size_t) il];
                int mi = -1;
                for (int m = 0; m < L0.nm; ++m)
                    if (member(L0, m).dev == di) mi = m;
                if (mi < 0) continue;
                if (!started && run.any(k, il, n_pre)) {   // the chunk's input: its embedding rows, or the previous group's hand-off
                    started = true;
                    if (il == 0) {
                        if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                        for (int t = 0; t < T; ++t)
                            std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                        (size_t) D * 2);
                        CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                        CK(cudaEventRecord(c.emb_ev[slot], s));
                        embed(c.emb, T, c.h, D, H, s);
                        init_pre(c.pre, T, s);
                    } else {
                        Chunk& a = *chunks[(size_t) L[(size_t) il - 1].dev];
                        xfer_get(chunk_xfer(a.hand[slot], c, T, false), a.hand_flag[slot], wp.win, s);
                        set_flag(a.hand_ack + ((size_t) slot * 8 + mi) * 32, wp.win, s);
                    }
                }
                for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                    if (sidx < 0 || got[(size_t) sidx]) continue;
                    got[(size_t) sidx] = 1;
                    Src& sr = srcs[(size_t) sidx];
                    if (L[(size_t) sr.layer].grp == L0.grp) continue;
                    const int64_t g_first = p0 / sr.ratio;
                    const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                    Xfer x;
                    if (ng > 0 && sr.ckv[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot);
                        x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * kv_row();
                        x.bytes[x.n++] = (size_t) ng * kv_row();
                    }
                    if (ng > 0 && sr.ik[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot) + x_cap() * g.hd * 2;
                        x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_row();
                        x.bytes[x.n++] = (size_t) ng * ik_row();
                    }
                    if (x.n > 0) xfer_get(x, sr.flag(slot), wp.win, s);
                    set_flag(sr.x_ack + ((size_t) slot * devs.size() + di) * 32, wp.win, s);   // (taken, or none)
                }
                if (last_k && mi == 0 && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        pend_finish(c, s);   // (the residual the layer before owes, first)
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                if (!run.any(k, il, n_pre)) continue;   // (a chunk before a truncated run's tail: its sources' rows only)
                const bool src_only = !run.full(k, il, n_pre);
                const int64_t lo = il >= n_pre ? run.lo : 0;
                const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
                // DS4_AR_DEFER=1: the MoE's sum by ranges beside the next layer's first steps (off: +0-2% on the
                // 16K-token prompt, and those steps' GEMMs on a range's rows round their sums otherwise - perplexity
                // on 4 x 1024 tokens at --tp 4 7.5141 with it, 7.4993 without)
                static const bool defer_ok = [] {
                    const char* e = std::getenv("DS4_AR_DEFER");
                    return e != nullptr && std::atoi(e) != 0;
                }();
                if (!layer_chunk(L0, mi, p0, T, wp, S, err, defer_ok && !last && dump_l == nullptr && !src_only, src_only,
                                 lo))
                    return false;
                if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, il, p0, T);
                // the hand-off, once the next group took the chunk before last (the chunks it takes: a truncated run's
                // consumer layers take only its tail)
                if (last && mi == 0 && il + 1 < g.nl && run.any(k, il + 1, n_pre)) {
                    const int nnext = L[(size_t) il + 1].nm;
                    if (k >= 2 && run.any(k - 2, il + 1, n_pre))
                        wait_flags(c.hand_ack + (size_t) slot * 8 * 32, nnext, 32, wp.win - 2, s);
                    xfer_put(chunk_xfer(c.hand[slot], c, T, true), c.hand_flag[slot], wp.win, d.xcnt, s);
                }
            }
            if (di == head_dev && (size_t) k < chunk_done.size()) {
                CK(cudaEventRecord(chunk_done[(size_t) k], s));
                chunk_rec[(size_t) k].store(1, std::memory_order_release);
            }
            if (d.grp == devs[(size_t) head_dev].grp) {   // the head group: each member its share of the logits
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        if (di != head_dev) continue;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
        }
        return true;
    }

    // n tokens at positions pos .. in chunks of equal size (at most `chunk`), pipelined over the groups; the logits of
    // the last min(logits_rows, 16) rows (logits_rows > 16: a run of one chunk, every row's into chunk_logits)
    bool forward_chunks(const int* tk, int n, int logits_rows, std::string& err) {
        if (n < 1) return true;
        if (pos + n > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < n; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        // equal chunks of at most `chunk`; DS4_CHUNK_FIRST=N: the first one N tokens, the rest equal - the first
        // group hands the next one its first chunk sooner (the pipeline's start), the rest stream the experts as few
        // times as before.  A truncated run (tail_rows): the rows before its tail and the tail each so
        static const int first_env = std::getenv("DS4_CHUNK_FIRST") ? std::atoi(std::getenv("DS4_CHUNK_FIRST")) : 0;
        logits_all = logits_rows > kWin;
        const int tail = tail_rows(n);
        Run run;
        run.lo = tail < n ? pos + n - tail : 0;
        auto part = [&](int a0, int a1, bool skip) {
            if (a1 <= a0) return;
            const int first = a0 == 0 && first_env > 0 && a1 - a0 > first_env + 16
                                  ? std::min(chunk, std::max(16, first_env / 16 * 16)) : 0;
            auto push = [&](int a, int t) {
                run.p0.push_back(pos + a);
                run.T.push_back(t);
                run.seq.push_back(++win_seq);
                run.skip.push_back(skip ? 1 : 0);
            };
            if (first > 0) push(a0, first);
            const int b0 = a0 + first, rest = a1 - b0, nc = (rest + chunk - 1) / chunk, per = (rest + nc - 1) / nc;
            for (int a = b0; a < a1; a += per) push(a, std::min(per, a1 - a));
        };
        part(0, n - tail, true);
        part(n - tail, n, false);
        if (std::getenv("DS4_TRUNC_SPLIT_ONLY") != nullptr) {   // (a test: the truncated run's chunks, every row everywhere)
            std::fill(run.skip.begin(), run.skip.end(), 0);
            run.lo = 0;
        }
        if (logits_all) chunk_logits.resize((size_t) n * g.vocab);
        for (int t = 0; t < n; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        plan_streams();
        const int T_last = run.T.back();
        n_logits = logits_all ? 0 : std::min(logits_rows, T_last);
        const double tl = now_ms();
        {   // each chunk's end on the head device, for the progress reports
            cudaSetDevice(devs[(size_t) head_dev].id);
            while (chunk_done.size() < run.T.size()) {
                cudaEvent_t e = nullptr;
                CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                chunk_done.push_back(e);
            }
            chunk_rec.reset(new std::atomic<int>[chunk_done.size()]);
            for (size_t i = 0; i < chunk_done.size(); ++i) chunk_rec[i].store(0);
        }
        std::vector<std::string> errs(devs.size());
        eg_read.store(0);
        eg_abort.store(false);
        for (auto& u : eg_used) u.store(0);
        std::thread egr;
        if (eg_copies > 0) egr = std::thread([this, &run] { eg_reader(run.p0, run.T); });
        struct JoinEg {   // (every way out: the reader done first - it waits only for copies the devices issue)
            std::thread& th;
            ~JoinEg() {
                if (th.joinable()) th.join();
            }
        } eg_join{egr};
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, &run, &errs] {
                std::string e;
                if (!run_dev_chunks((int) di, run, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        size_t reported = 0;
        for (;;) {
            bool all = true;
            for (auto& w : workers) all = all && w->idle();
            if (progress) {   // chunks through every layer (the last layer's device past their event)
                while (reported < run.T.size() && chunk_rec[reported].load(std::memory_order_acquire) &&
                       cudaEventQuery(chunk_done[reported]) == cudaSuccess)
                    ++reported;
                if (reported > 0 && reported < run.T.size())
                    progress(run.p0[reported - 1] + run.T[reported - 1] - run.p0[0]);
            }
            if (all) {
                eg_abort.store(true, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        for (const std::string& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
        st.layer_ms += now_ms() - tl;
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        if (logits_all) n_logits = (T_last - 1) % kWin + 1;   // (sample() takes the last row of d.logits)
        pos += n;
        st.tokens += n;
        cache_tick(n);
        const int nr = std::min(T_last, tap_cap);
        if (!seed_drafter(pos - nr, nr, 0, err)) return false;
        return true;
    }

    // the window's plan, in window order (the indexer's and the candidate source's sizes carry from window to window)
    void plan_window(int64_t p0, int T, WinPlan& wp) {
        wp.par = par;
        wp.async = par || (!sync_decode && T <= async_max_t);
        wp.vl = 0;
        wp.skip = win_skip;
        wp.lo = win_lo;
        for (int t = 0; t < T && !imgs.empty(); ++t)
            if (in_image(p0 + t)) wp.vl |= 1u << t;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.ratio > 0) {
                const int R = l.ratio;
                int n[kWin], k[kWin], clen_max = 0;
                for (int t = 0; t < T; ++t) {
                    n[t] = (int) ((p0 + t + 1) / R);
                    k[t] = std::min(g.ix_topk, n[t]);
                    clen_max = std::max(clen_max, n[t]);
                }
                if (l.ix_src) {
                    for (int t = 0; t < T; ++t) wp.cand[il][t] = cur_cand_n[t];
                    if (clen_max > 0 && l.cand_src)
                        for (int t = 0; t < T; ++t) cur_cand_n[t] = std::min(g.cand_topk, (n[t] + g.cand_blk - 1) / g.cand_blk);
                    for (int t = 0; t < T; ++t) cur_topk_n[t] = k[t];
                }
                for (int t = 0; t < T; ++t) wp.topk[il][t] = cur_topk_n[t];
            }
            if (wp.async && (!wp.skip || il < n_pre)) {   // (the layers whose MoE runs: the service thread's order)
                wp.seq[il] = ++seq_issued;
                layer_of[wp.seq[il] % kRing].store(il, std::memory_order_release);
            }
        }
    }

    int head_nm() const { return (int) groups[(size_t) devs[(size_t) head_dev].grp].mem.size(); }
    // the head over `rows` (<= 16) normed rows xn on member mi of the head group (device d): its share of each row's
    // logits (columns [v0, v0 + vn), split128 of V); the lead gathers the others' into d.logits (rows V apart) - two
    // slots by seq (Vmax apart, the widest share's), a member reusing one once the lead acknowledged it.  gather
    // false: each member keeps its share (the drafter's markov steps over the group, mk_split)
    bool head_share(Dev& d, int mi, const float* xn, int rows, std::string& err, bool gather = true) {
        const int V = g.vocab, D = g.dim, nm = head_nm();
        int v0 = 0, vn = 0, Vmax = 0, z = 0;
        split128(V, nm, mi, v0, vn);
        split128(V, nm, 0, z, Vmax);
        Group& G = groups[(size_t) d.grp];
        float* out = mi == 0 ? d.logits : d.logits_part;
        const size_t ld = mi == 0 ? (size_t) V : (size_t) vn;
        cudaSetDevice(d.id);
        // (as many activations at a time as the head's split-K partials fit the scratch: 8 for a shared head, 4 for a
        // whole one - each pass reads the whole share)
        const int per = std::max(1, x3::mv_cols(d.head));
        for (int c = 0; c < rows; c += per)
            if (!mv(d, {job(d.head, xn + (size_t) c * D, out + (size_t) c * ld, std::min(per, rows - c), D, ld)}, err))
                return false;
        if (nm == 1 || !gather) return true;
        const unsigned seq = ++d.head_seq;
        const int slot = (int) (seq % 2);
        auto stage = [&](int m) { return G.lg_stage + ((size_t) slot * (nm - 1) + (m - 1)) * kWin * Vmax; };
        auto flag = [&](int m) { return G.lg_flag + ((size_t) slot * (nm - 1) + (m - 1)) * 32; };
        if (mi > 0) {
            if (seq > 2) wait_flags(G.lg_ack + (size_t) slot * 32, 1, 32, seq - 2, d.st);
            Xfer x;
            x.n = 1;
            x.src[0] = d.logits_part;
            x.dst[0] = stage(mi);
            x.bytes[0] = (size_t) rows * vn * 4;
            xfer_put(x, flag(mi), seq, d.xcnt, d.st);
            return true;
        }
        for (int m = 1; m < nm; ++m) {
            int m0 = 0, mn = 0;
            split128(V, nm, m, m0, mn);
            Xfer x;
            x.n = 1;
            x.src[0] = stage(m);
            x.dst[0] = d.logits_part;
            x.bytes[0] = (size_t) rows * mn * 4;
            xfer_get(x, flag(m), seq, d.st);
            CK(cudaMemcpy2DAsync(d.logits + (size_t) m0, (size_t) V * 4, d.logits_part, (size_t) mn * 4,
                                 (size_t) mn * 4, (size_t) rows, cudaMemcpyDeviceToDevice, d.st));
        }
        set_flag(G.lg_ack + (size_t) slot * 32, seq, d.st);
        return true;
    }

    // the head's rows (the last n_logits of the window) on head group member di (its h / pre: the group's)
    bool head_rows_dev(int di, int T, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const int D = g.dim, H = g.hc;
        if (n_logits <= 0) return true;
        const int t0 = T - n_logits;
        hc_pre_norm(d.h + (size_t) t0 * H * D, d.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, d.st);
        return head_share(d, d.mi, d.x, n_logits, err);
    }

    // the DSpark drafter's taps: the streams' mean entering target layer il, on device d (after the last seeding read
    // h_taps)
    bool tap(Dev& d, int il, int T, std::string& err) {
        const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
        if (g.n_mtp == 0 || it == g.d_targets.end()) return true;
        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
        const int D = g.dim;
        cudaSetDevice(d.id);
        hc_mean(d.h, T, g.hc, D, d.tap, D, d.st);
        CK(cudaStreamWaitEvent(d.st, ev_seed, 0));
        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4, (size_t) T,
                             cudaMemcpyDeviceToHost, d.st));
        return true;
    }

    bool forward(const int* tk, int T, int logits_rows, std::string& err, int seed_rows = -1) {
        if (T < 1 || T > kWin) {
            err = "a window of " + std::to_string(T) + " tokens";
            return false;
        }
        if (pos + T > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < T; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        const int64_t p0 = pos;
        for (int t = 0; t < T; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        if (!eg_ring_tried && par && pk.eg_layers > 0) {
            eg_ring_tried = true;
            if (std::getenv("DS4_EG_URING") && std::atoi(std::getenv("DS4_EG_URING")) != 0) {
                eg_ring.init(512);
                if (eg_ring.ok()) eg_open();
            }
        }
        if (eg_async()) {   // (forward_par's window: win_seq + 1)
            if (!eg_th.joinable()) eg_th = std::thread([this] { eg_loop(); });
            {
                std::lock_guard<std::mutex> lk(eg_th_mu);
                eg_jobs.push_back({p0, T, win_seq + 1});
            }
            eg_th_cv.notify_one();
        } else {
            const double te = now_ms();
            if (pk.eg_layers > 0) engram_rows(p0, T);
            st.engram_ms += now_ms() - te;
        }
        const int D = g.dim;
        for (int t = 0; t < T; ++t)   // (h_emb is written again by the next window, after this one's last sync)
            std::memcpy(h_emb + (size_t) t * D, emb_row(p0 + t, tk[t]), (size_t) D * 2);
        plan_window(p0, T, wplan);
        n_logits = std::min(logits_rows, T);
        const double tl = now_ms();
        if (par) {
            if (!forward_par(p0, T, err)) return false;
        } else if (!forward_serial(p0, T, err)) return false;
        st.layer_ms += now_ms() - tl;
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        pos += T;
        st.tokens += T;
        cache_tick(T);
        if (seed_rows != 0 && !seed_drafter(p0, seed_rows < 0 ? T : seed_rows, 0, err)) return false;
        return true;
    }

    // one thread: the layers device after device, the hand-offs through pinned memory and events
    bool forward_serial(int64_t p0, int T, std::string& err) {
        const int D = g.dim, H = g.hc;
        int cur = L[0].dev;
        {
            Dev& d = devs[(size_t) cur];
            cudaSetDevice(d.id);
            CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, d.st));
            embed(d.emb_rows, T, d.h, D, H, d.st);
            init_pre(d.pre, T, d.st);
        }
        std::vector<float> dump;
        if (dump_h != nullptr) dump.resize((size_t) T * g.nl * H * D);
        for (int il = 0; il < g.nl; ++il) {
            Layer& l = L[(size_t) il];
            if (l.dev != cur) {
                Dev& a = devs[(size_t) cur];
                Dev& b = devs[(size_t) l.dev];
                cudaSetDevice(a.id);
                const size_t nh = (size_t) T * H * D * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                             nc = (size_t) T * cand_stride() * 4;
                uint8_t* bb = a.bounce;
                CK(cudaMemcpyAsync(bb, a.h, nh, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh, a.pre, np, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np, a.topk, nk, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np + nk, a.cand, nc, cudaMemcpyDeviceToHost, a.st));
                CK(cudaEventRecord(a.ev_out, a.st));
                cudaSetDevice(b.id);
                CK(cudaStreamWaitEvent(b.st, a.ev_out, 0));
                CK(cudaMemcpyAsync(b.h, bb, nh, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.pre, bb + nh, np, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.topk, bb + nh + np, nk, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.cand, bb + nh + np + nk, nc, cudaMemcpyHostToDevice, b.st));
                cur = l.dev;
            }
            if (!tap(devs[(size_t) l.dev], il, T, err)) return false;
            if (!layer(l, 0, p0, T, wplan, err)) return false;
            if (dump_h != nullptr) {   // DS4_DUMP_H: every layer's residual stream, for tools/ds4/compare.py --layers
                Dev& dd = devs[(size_t) l.dev];
                std::vector<float> hv((size_t) T * H * D);
                cudaSetDevice(dd.id);
                CK(cudaMemcpyAsync(hv.data(), dd.h, hv.size() * 4, cudaMemcpyDeviceToHost, dd.st));
                CK(cudaStreamSynchronize(dd.st));
                for (int t = 0; t < T; ++t)
                    std::memcpy(dump.data() + ((size_t) t * g.nl + l.il) * H * D, hv.data() + (size_t) t * H * D,
                                (size_t) H * D * 4);
            }
        }
        if (dump_h != nullptr) std::fwrite(dump.data(), 4, dump.size(), dump_h);
        if (cur != head_dev) {
            err = "the head is not on the last layer's GPU";
            return false;
        }
        return head_rows_dev(head_dev, T, err);
    }

    // threads: each device's thread enqueues its layers (a group's members each their share) at once, the
    // devices waiting on each other through host-mapped flags (the window's seq)
    bool forward_par(int64_t p0, int T, std::string& err) {
        wplan.win = ++win_seq;
        win_now.store(wplan.win, std::memory_order_release);
        std::vector<std::string> errs(devs.size());
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, p0, T, &errs] {
                std::string e;
                if (!run_dev((int) di, p0, T, wplan, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        for (auto& w : workers) w->wait();
        for (const std::string& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
        return true;
    }

    // the hand-off between groups: h | pre | topk | cand of device d <-> the host-mapped buffer
    Xfer hand_xfer(uint8_t* host, Dev& d, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {d.h, d.pre, d.topk, d.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    bool run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc;
        bool started = false;
        std::vector<char> got(srcs.size(), 0);
        for (int il = 0; il < g.nl; ++il) {
            Layer& L0 = L[(size_t) il];
            int mi = -1;
            for (int m = 0; m < L0.nm; ++m)
                if (member(L0, m).dev == di) mi = m;
            if (mi < 0) continue;
            const bool any = !wp.skip || il <= n_pre;   // (a window before a truncated prompt's tail: to n_pre only)
            if (!started && any) {   // the window's input: the embedding rows, or the previous group's hand-off
                started = true;
                if (il == 0) {
                    CK(cudaMemcpyAsync(d.emb_rows, h_emb, (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    embed(d.emb_rows, T, d.h, D, H, s);
                    init_pre(d.pre, T, s);
                } else {
                    const Dev& a = devs[(size_t) L[(size_t) il - 1].dev];
                    xfer_get(hand_xfer(a.hand, d, T, false), a.hand_flag, wp.win, s);
                }
            }
            // the compressed KV / index keys another group's source layer wrote this window
            for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                if (sidx < 0 || got[(size_t) sidx]) continue;
                got[(size_t) sidx] = 1;
                Src& sr = srcs[(size_t) sidx];
                if (L[(size_t) sr.layer].grp == L0.grp) continue;
                const int64_t g_first = p0 / sr.ratio;
                const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                if (ng == 0) continue;
                Xfer x;
                if (sr.ckv[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage;
                    x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * kv_row();
                    x.bytes[x.n++] = (size_t) ng * kv_row();
                }
                if (sr.ik[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage + x_cap() * g.hd * 2;
                    x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_row();
                    x.bytes[x.n++] = (size_t) ng * ik_row();
                }
                if (x.n > 0) xfer_get(x, sr.x_flag, wp.win, s);
            }
            if (!any) continue;
            if (mi == 0 && !tap(d, il, T, err)) return false;
            if (!layer(L0, mi, p0, T, wp, err)) return false;
            if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, il, p0, T);
            const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
            if (last && mi == 0 && il + 1 < g.nl && (!wp.skip || il + 1 <= n_pre))
                xfer_put(hand_xfer(d.hand, d, T, true), d.hand_flag, wp.win, d.xcnt, s);
        }
        if (d.grp == devs[(size_t) head_dev].grp && !head_rows_dev(di, T, err)) return false;
        if (!idle_ev.empty()) {   // (its turn over: the expert cache's copies from here, copy_open)
            CK(cudaEventRecord(idle_ev[(size_t) di], s));
            idle_win[(size_t) di].store(wp.win, std::memory_order_release);
        }
        return true;
    }

    // a source's new cache rows (src on device `from`, bytes of them) to the same place of its other devices' copies,
    // through pinned memory: the copy out on the source's stream, each copy in on its device's stream after it
    bool replicate(std::vector<uint8_t*>& copies, uint8_t* stage, cudaEvent_t ev, const uint8_t* src, size_t at,
                   size_t bytes, int from, std::string& err) {   // (at: bytes into each copy)
        bool any = false;
        for (size_t od = 0; od < devs.size(); ++od) any = any || (copies[od] != nullptr && (int) od != from);
        if (!any) return true;
        Dev& d = devs[(size_t) from];
        CK(cudaMemcpyAsync(stage, src, bytes, cudaMemcpyDeviceToHost, d.st));
        CK(cudaEventRecord(ev, d.st));
        for (size_t od = 0; od < devs.size(); ++od) {
            if (copies[od] == nullptr || (int) od == from) continue;
            Dev& o2 = devs[od];
            cudaSetDevice(o2.id);
            CK(cudaStreamWaitEvent(o2.st, ev, 0));
            CK(cudaMemcpyAsync(copies[od] + at, stage, bytes, cudaMemcpyHostToDevice, o2.st));
        }
        cudaSetDevice(d.id);
        return true;
    }

    // layer L0 for the window, as group member mi (its share of the heads, of the output projection and of the FFNs;
    // the partial sums all-reduced across the group).  A window before a truncated prompt's tail (wp.skip): layer n_pre
    // its KV source's part alone (layer_chunk's src_only); the tail's: the consumer layers' windows from wp.lo on
    bool layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ag = groups[(size_t) L0.grp].ar;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        const bool src_only = wp.skip && !l.mtp && l.il == n_pre;
        const int64_t lo = !wp.skip && !l.mtp && l.il >= n_pre ? wp.lo : 0;
        // Engram
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            uint8_t* src = h_eg + (size_t) l.eg * eg_cap * per;
            if (wp.par && eg_async() && eg_win[l.eg].load(std::memory_order_acquire) != wp.win) {
                const double te = now_ms();
                eg_wait(l.eg, wp.win);
                if (mi == 0) st.engram_ms += now_ms() - te;
            }
            for (int t = 0; t < T; ++t)   // [token][rows | scales] -> [all rows][all scales]
                CK(cudaMemcpyAsync(d.eg_rows + (size_t) t * cols * ehd, src + (size_t) t * per, (size_t) cols * ehd,
                                   cudaMemcpyHostToDevice, s));
            uint8_t* sc = d.eg_rows + (size_t) T * cols * ehd;
            for (int t = 0; t < T; ++t)
                CK(cudaMemcpyAsync(sc + (size_t) t * cols * (ehd / 32), src + (size_t) t * per + (size_t) cols * ehd,
                                   (size_t) cols * (ehd / 32), cudaMemcpyHostToDevice, s));
            engram_dequant(d.eg_rows, sc, T * cols, ehd, d.eg_x, s);
            if (!mv(d, {job(l.eg_wkv, d.eg_x, d.eg_kv, T, (size_t) cols * ehd, (size_t) (H + 1) * D)}, err)) return false;
            engram_gate(d.h, d.eg_kv, l.eg_w, H, D, T, g.eps, s);
        }
        // attention
        // (hc_mix + hc_split on st2 beside the attention: the pre norm takes the lagged pre, a_post / a_comb / a_pre are
        // read from the attention's hc_post on - st waits for them there)
        if (!src_only) {
            CK(cudaEventRecord(d.hc_go, s));
            CK(cudaStreamWaitEvent(d.st2, d.hc_go, 0));
            hc_mix(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.st2);
            hc_split(d.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, d.a_pre, d.a_post, d.a_comb, d.st2);
            CK(cudaEventRecord(d.hc_done, d.st2));
        }
        hc_pre_norm(d.h, d.pre, l.attn_norm, D, H, T, g.eps, d.x, s);
        {
            x3::MvJob j[4];
            int nj = 0;
            if (!src_only) {
                j[nj++] = job(l.wq_a, d.x, d.qr, T, D, g.q_lora);
                j[nj++] = job(l.wkv, d.x, d.kvr, T, D, HD);
            }
            if (l.kv_src) j[nj++] = job(l.c_wkv, d.x, d.ckr, T, D, HD);
            if (l.kv_src && l.ratio > 1) j[nj++] = job(l.c_wgate, d.x, d.cgr, T, D, HD);
            if (!mv_n(d, j, nj, err)) return false;
        }
        Finish fq;
        fq.w = l.q_norm;
        fq.eps = g.eps;
        if (!src_only) finish(d.qr, T, g.q_lora, g.q_lora, fq, s);
        if (NHL > 0 && !src_only) {
            if (!mv(d, {job(l.wq_b, d.qr, d.q, T, g.q_lora, HQ)}, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) p0;
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(d.q, T * NHL, HD, HD, fr, s);
        }
        // the window rows: the ring's (positions p0 - win ..) then this window's, each query its own 128.  A DSpark
        // block's ring holds the trunk's main KV (positions <= p0 - 1, seeded); its block of drafts attends to all of
        // that ring and to the whole block, and leaves the ring as it was
        if (!src_only) {
            if (l.mtp) CK(cudaMemcpyAsync(d.ext, l.win, (size_t) g.win * HD * 2, cudaMemcpyDeviceToDevice, s));
            else ring_to_ext(l.win, d.ext, p0, g.win, HD, s);
            Finish fk;
            fk.w = l.kv_norm;
            fk.eps = g.eps;
            fk.rd = g.rd;
            fk.pos = (int) p0;
            fk.freqs = freqs;
            fk.quant = 1;
            fk.dst = d.ext + (size_t) g.win * HD;
            fk.dst_stride = HD;
            finish(d.kvr, T, HD, HD, fk, s);
            if (!l.mtp) ext_to_ring(d.ext, l.win, p0, g.win, T, HD, s, spec_win ? l.ring_bak : nullptr);
        }
        AttnRows ar;
        for (int t = 0; t < T; ++t) {
            if (l.mtp) {
                ar.win0[t] = 0;
                ar.nwin[t] = (int) std::min<int64_t>(g.win, p0);
                ar.nsel[t] = T;
                continue;
            }
            // ext row of position max(0, lo, p - win + 1)
            const int first = (int) std::max<int64_t>(g.win - p0 + lo, t + 1);
            ar.win0[t] = first;
            ar.nwin[t] = g.win + t - first + 1;
            ar.nsel[t] = 0;
        }
        const uint16_t* ckv = nullptr;   // (a DSpark block: its BF16 rows in ext)
        const uint8_t* ckvp = nullptr;   // (a trunk layer: the source's packed rows)
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            // the groups this window completes: g_first .. g_first + ng - 1
            const int64_t g_first = p0 / R;              // the first group whose last position is >= p0
            const int64_t g_end = (p0 + T) / R;          // groups completed by p0 + T - 1
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(d.ckr, T, HD, HD, fc, s);
                    CK(cudaMemcpyAsync(d.latent, d.ckr, (size_t) T * HD * 4, cudaMemcpyDeviceToDevice, s));
                } else {
                    if (spec_win && l.c_snap != nullptr) {   // to replay the accepted rows if drafts are rejected
                        const size_t ns = (size_t) R * HD * 4, nr = (size_t) T * HD * 4;
                        CK(cudaMemcpyAsync(l.c_snap, l.c_kv, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_snap + (size_t) R * HD, l.c_score, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows, d.ckr, nr, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows + (size_t) kWin * HD, d.cgr, nr, cudaMemcpyDeviceToDevice, s));
                    }
                    compress_win(d.ckr, d.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, d.latent, s);
                }
            }
            Rows lim;
            int clen_max = 0;
            for (int t = 0; t < T; ++t) {
                lim.n[t] = (int) ((p0 + t + 1) / R);
                lim.k[t] = std::min(g.ix_topk, lim.n[t]);
                clen_max = std::max(clen_max, lim.n[t]);
            }
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    if (!mv(d, {job(l.i_wk, d.latent, d.ikr, ng, HD, g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_row();
                    fi.pdst_stride = (long long) ik_row();
                    finish(d.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                    if (!wp.par && !replicate(ks.ik, ks.h_ik, ks.ev_ik, fi.pdst, (size_t) g_first * ik_row(),
                                              (size_t) ng * ik_row(), l.dev, err))
                        return false;
                }
                if (clen_max > 0 && !src_only) {
                    if (!mv(d, {job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, (size_t) g.ix_heads * g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.rd = g.rd;
                    fi.pos = (int) p0;
                    fi.vpp = g.ix_heads;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                    // (iw rows are 64 apart)
                    gemv_f16_rows(l.i_wproj, d.x, (size_t) D, g.ix_heads, D, T,
                                  1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), d.iw, 64, s);
                    const long long ss = (long long) o.max_context + 64;
                    bool restrict_ = false;
                    for (int t = 0; t < T; ++t)
                        restrict_ = restrict_ || (lim.n[t] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                    if (l.uses_cand && restrict_) {
                        Rows cl;   // the candidate positions of each query: its blocks x blk
                        int nmax = 0;
                        for (int t = 0; t < T; ++t) {
                            cl.n[t] = lim.n[t];
                            cl.k[t] = wp.cand[l.il][t];
                            nmax = std::max(nmax, wp.cand[l.il][t] * g.cand_blk);
                        }
                        index_score(d.iq, d.iw, nullptr, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, nmax, d.cand,
                                    (int) cand_stride(), g.cand_blk, cl, d.scores, ss, s);
                        if (index_check)
                            index_check_rows(d, l.il, d.iq, d.iw, ks.ik[(size_t) l.dev], T, nmax, d.cand, g.cand_blk, cl, ss, s);
                        Rows tr;
                        for (int t = 0; t < T; ++t) {
                            tr.n[t] = wp.cand[l.il][t] * g.cand_blk;
                            tr.k[t] = lim.k[t];
                        }
                        topk(d.scores, ss, T, tr, d.cand, (int) cand_stride(), g.cand_blk, d.topk, (int) topk_stride(), s,
                             &d.tks);
                    } else {
                        index_score(d.iq, d.iw, nullptr, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, clen_max, nullptr, 0, 1,
                                    lim, d.scores, ss, s);
                        if (index_check)
                            index_check_rows(d, l.il, d.iq, d.iw, ks.ik[(size_t) l.dev], T, clen_max, nullptr, 1, lim, ss, s);
                        if (l.cand_src) {
                            Rows cr;
                            for (int t = 0; t < T; ++t) {
                                cr.n[t] = lim.n[t];
                                cr.k[t] = g.cand_topk;
                            }
                            cand_blocks(d.scores, ss, T, cr, g.cand_blk, d.bm, (long long) bm_stride(), d.cand,
                                        (int) cand_stride(), s);
                        }
                        topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s, &d.tks);
                        if (topk_check) {   // (into d.cand's rows past the window's: scratch here)
                            const size_t nw = (size_t) T * topk_stride();
                            std::vector<int> tkc_a(nw), tkc_b(nw);
                            CK(cudaMemcpyAsync(tkc_a.data(), d.topk, nw * 4, cudaMemcpyDeviceToHost, s));
                            topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s);
                            CK(cudaMemcpyAsync(tkc_b.data(), d.topk, nw * 4, cudaMemcpyDeviceToHost, s));
                            CK(cudaStreamSynchronize(s));
                            bool bad = false;
                            for (int t = 0; t < T; ++t)
                                bad = bad || std::memcmp(tkc_a.data() + (size_t) t * topk_stride(),
                                                         tkc_b.data() + (size_t) t * topk_stride(), (size_t) lim.k[t] * 4) != 0;
                            if (bad) ++tkc_bad;
                            const long long nc = ++tkc_calls;
                            if (nc % 256 == 0 || bad)
                                std::fprintf(stderr, "topk check: %lld calls, %lld differed (layer %d, n %d)\n", nc,
                                             tkc_bad.load(), l.il, lim.n[0]);
                        }
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * kv_row();
                fc.pdst_stride = (long long) kv_row();
                finish(d.latent, ng, HD, HD, fc, s);
                if (!wp.par &&
                    !replicate(sr.ckv, sr.h_ckv, sr.ev_ckv, fc.pdst, (size_t) g_first * kv_row(), (size_t) ng * kv_row(), l.dev,
                               err))
                    return false;
                if (wp.par && mi == 0 && sr.remote) {   // the new rows for the other groups' readers
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.x_stage;
                    x.bytes[0] = (size_t) ng * kv_row();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_row();
                        x.dst[1] = sr.x_stage + x_cap() * HD * 2;
                        x.bytes[1] = (size_t) ng * ik_row();
                        x.n = 2;
                    }
                    xfer_put(x, sr.x_flag, wp.win, d.xcnt, s);
                }
            }
            for (int t = 0; t < T; ++t) ar.nsel[t] = lim.n[t] > 0 ? wp.topk[l.il][t] : 0;
            ckvp = sr.ckv[(size_t) l.dev];
        }
        if (src_only) return true;
        if (l.mtp) {   // the block's rows, after the ring's in ext, as every query's selection
            ckv = d.ext + (size_t) g.win * HD;
            iota_rows(d.topk, T, T, (int) topk_stride(), s);
        }
        if (NHL > 0) {
            sparse_attn(d.q, d.ext, ckv, ckvp, d.topk, (int) topk_stride(), ar, l.sink, NHL, HD, T,
                        1.0f / std::sqrt((float) HD), d.o, s);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) p0;
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(d.o, T * NHL, HD, HD, fo, s);
            x3::MvJob j[16];
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                j[gi] = job(l.wo_a[gi], d.o + (size_t) gi * per, d.oa + (size_t) gi * g.o_lora, T, HQ, (size_t) OG * g.o_lora);
            if (!mv_n(d, j, OG, err)) return false;
            if (!mv(d, {job(l.wo_b, d.oa, d.ao, T, (size_t) OG * g.o_lora, D)}, err)) return false;
        } else CK(cudaMemsetAsync(d.ao, 0, (size_t) T * D * sizeof(*d.ao), s));   // (no heads: nothing to add)
        if (nm > 1) allreduce(d.ao, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
        CK(cudaStreamWaitEvent(s, d.hc_done, 0));
        hc_post(d.ao, d.h, d.a_post, d.a_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        // the MoE
        // (the same beside the MoE: f_post / f_comb / f_pre are read from its hc_post on)
        CK(cudaEventRecord(d.hc_go, s));
        CK(cudaStreamWaitEvent(d.st2, d.hc_go, 0));
        hc_mix(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.st2);
        hc_split(d.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, d.f_pre, d.f_post, d.f_comb, d.st2);
        CK(cudaEventRecord(d.hc_done, d.st2));
        hc_pre_norm(d.h, d.a_pre, l.ffn_norm, D, H, T, g.eps, d.x, s);
        if (wp.async) {
            // the routing goes to the service thread through the mailbox (a group's lead publishes it, the members
            // make the same plan) while the device runs the shared expert and the VRAM experts (from the plan the
            // router made), and waits for the answer only before the experts' sum
            gemv_f16_rows(l.gate_w, d.x, (size_t) D, l.n_exp, D, T, 1.0f, d.glog, 1024, s);
            const unsigned sq = wp.seq[l.il];
            const int slot = (int) (sq % kRing);
            const bool lead = mi == 0;
            route_publish(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, l.vram_tab, d.x, D, d.ridx, d.rw,
                          d.plan_ptr, lead ? reqs + slot : nullptr, sq, s, l.gate_bias_vl, l.mtp ? 0u : wp.vl);
            if (!mv(d, {job(l.s1, d.x, d.sg, T, D, FF), job(l.s3, d.x, d.su, T, D, FF)}, err)) return false;
            swiglu(d.sg, d.su, T * FF, g.lim, d.sh, s);
            if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, FF, D)}, err)) return false;
            // (a mixed layer's: lay.hdr - the kernels read each blob's own layout)
            const x3::ExpertLayout& lay = nm > 1 ? l.cls[0].hlay : l.cls[0].lay;
            if (T == 1 && moe_token_path) {
                x3::MoeArgs ma;
                ma.plan_ptr = d.plan_ptr;
                ma.plan_w = d.rw;
                ma.cpu_part = lead ? resps[slot].part[0] : nullptr;
                ma.cpu_flag = lead ? &resps[slot].cpu[0] : nullptr;
                ma.lay = lay;
                ma.k = l.k_exp;
                ma.n_embd = D;
                ma.n_ff = FF;
                ma.limit = g.lim;
                x3::moe_gate_up(ma, d.x, d.xg, s);
                if (lead) wait_seq(&resps[slot].seq, sq, s);
                x3::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
            } else {   // each distinct expert read once for the window's tokens
                gf::MoeWin mw;
                mw.plan_ptr = d.plan_ptr;
                mw.T = T;
                mw.k = l.k_exp;
                mw.grp_ptr = d.grp_ptr;
                mw.grp_start = d.grp_start;
                mw.grp_ent = d.grp_ent;
                mw.n_groups = d.n_groups;
                gf::moe_win_group(mw, s);
                x3::WinArgs wa;
                wa.grp_ptr = d.grp_ptr;
                wa.grp_start = d.grp_start;
                wa.grp_ent = d.grp_ent;
                wa.n_groups = d.n_groups;
                wa.plan_ptr = d.plan_ptr;
                wa.plan_w = d.rw;
                wa.cpu_part = lead ? &resps[slot].part[0][0] : nullptr;
                wa.cpu_stride = kMaxDim;
                wa.cpu_flag = lead ? resps[slot].cpu : nullptr;
                wa.wait_ptr = lead ? &resps[slot].seq : nullptr;
                wa.wait_val = sq;
                wa.x = d.x;
                wa.x_stride = (size_t) D;
                wa.h = d.xg;
                wa.part = d.ed;
                wa.sh_out = d.sh_out;
                wa.out = d.moe_out;
                wa.out_stride = (size_t) D;
                wa.lay = lay;
                wa.T = T;
                wa.k = l.k_exp;
                wa.n_embd = D;
                wa.n_ff = FF;
                wa.limit = g.lim;
                if (!x3::win_experts(wa, s)) {
                    err = "exl3 win_experts failed";
                    return false;
                }
            }
            if (lead) st.experts += (int64_t) T * l.k_exp;
            if (nm > 1) allreduce(d.moe_out, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
            CK(cudaStreamWaitEvent(s, d.hc_done, 0));
            hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
            std::swap(d.h, d.h2);
            CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            return true;
        }
        if (nm > 1) {
            err = "a tensor-parallel layer on the host-routed path";
            return false;
        }
        // (glog rows are 1024 apart: route reads them so)
        gemv_f16_rows(l.gate_w, d.x, (size_t) D, l.n_exp, D, T, 1.0f, d.glog, 1024, s);
        VlBits vb = {};
        vb.w[0] = l.mtp ? 0u : wp.vl;
        route(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, d.ridx, d.rw, s, l.gate_bias_vl,
              vb.w[0] != 0 ? &vb : nullptr);
        CK(cudaMemcpyAsync(h_idx, d.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_w, d.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_x, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaEventRecord(d.ev, s));
        // the shared expert while the host looks at the picks
        if (!mv(d, {job(l.s1, d.x, d.sg, T, D, g.ff), job(l.s3, d.x, d.su, T, D, g.ff)}, err)) return false;
        swiglu(d.sg, d.su, T * g.ff, g.lim, d.sh, s);
        if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, g.ff, D)}, err)) return false;
        const double tw = now_ms();
        CK(cudaEventSynchronize(d.ev));
        st.gpu_wait_ms += now_ms() - tw;
        // the picks by expert: the VRAM ones' rows (gathered inputs, one GEMV per matrix for all their tokens), the
        // CPU lane's entries
        Combine cm;
        Gather ga;
        uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        int vex[kWin * kPicks], vrow0[kWin * kPicks], vn[kWin * kPicks];
        int nvex = 0, nrows = 0;
        std::vector<Ent> ents;
        std::vector<std::pair<int, float>> cpu_rows;   // slot -> (token, weight)
        std::vector<std::vector<std::pair<int, int>>> per((size_t) l.n_exp);   // expert -> (token, pick)
        for (int t = 0; t < T; ++t) {
            cm.n[t] = 0;
            cm.cpu[t] = 0;
            for (int i = 0; i < l.k_exp; ++i) {
                const int e = h_idx[t * kPicks + i];
                if (e < 0 || e >= l.n_exp) {
                    err = "router picked expert " + std::to_string(e);
                    return false;
                }
                ++pr[e];
                l.heat[(size_t) e] += 1.0f;
                if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                per[(size_t) e].push_back({t, i});
            }
        }
        int nslots = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            const auto& pe = per[(size_t) e];
            if (pe.empty()) continue;
            if (l.slot[(size_t) e] >= 0) {
                vex[nvex] = e;
                vrow0[nvex] = nrows;
                vn[nvex] = (int) pe.size();
                for (const auto& tp : pe) {
                    ga.src[nrows] = tp.first;
                    const int t = tp.first;
                    cm.row[t][cm.n[t]] = nrows;
                    cm.w[t][cm.n[t]] = h_w[t * kPicks + tp.second];
                    ++cm.n[t];
                    ++nrows;
                }
                ++nvex;
            } else {
                for (size_t a = 0; a < pe.size(); a += kc::kExl3CpuMaxN) {
                    Ent en;
                    en.e = e;
                    en.slot0 = nslots;
                    en.n = (int) std::min<size_t>(kc::kExl3CpuMaxN, pe.size() - a);
                    for (int i = 0; i < en.n; ++i) {
                        en.tok[i] = pe[a + (size_t) i].first;
                        cpu_rows.push_back({pe[a + (size_t) i].first, h_w[pe[a + (size_t) i].first * kPicks + pe[a + (size_t) i].second]});
                        cm.cpu[en.tok[i]] = 1;
                    }
                    nslots += en.n;
                    ents.push_back(en);
                }
            }
        }
        st.experts += (int64_t) T * l.k_exp;
        st.cpu_experts += nslots;
        if (nrows > 0) {
            gather_rows(d.x, ga, nrows, D, d.xe, s);
            x3::MvJob j[2 * kWin * kPicks];
            int nj = 0;
            for (int i = 0; i < nvex; ++i) {
                const auto& vm = l.vm[(size_t) vex[i]];
                j[nj++] = job(vm[0], d.xe + (size_t) vrow0[i] * D, d.xg + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
                j[nj++] = job(vm[1], d.xe + (size_t) vrow0[i] * D, d.xu + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
            }
            if (!mv_n(d, j, nj, err)) return false;
            swiglu(d.xg, d.xu, nrows * g.ff, g.lim, d.xhh, s);
            nj = 0;
            for (int i = 0; i < nvex; ++i)
                j[nj++] = job(l.vm[(size_t) vex[i]][2], d.xhh + (size_t) vrow0[i] * g.ff, d.ed + (size_t) vrow0[i] * D,
                              vn[i], g.ff, D);
            if (!mv_n(d, j, nj, err)) return false;
        }
        if (!ents.empty()) {
            const double tc = now_ms();
            cpu_experts(l, h_x, ents);
            std::fill(h_part, h_part + (size_t) T * D, 0.0f);
            for (int sl = 0; sl < nslots; ++sl) {
                float* o = h_part + (size_t) cpu_rows[(size_t) sl].first * D;
                const float wv = cpu_rows[(size_t) sl].second;
                const float* y = c_yd.data() + (size_t) sl * D;
                for (int e = 0; e < D; ++e) o[e] += wv * y[e];
            }
            st.cpu_ms += now_ms() - tc;
            CK(cudaMemcpyAsync(d.cpu_part, h_part, (size_t) T * D * 4, cudaMemcpyHostToDevice, s));
        }
        combine(d.ed, ents.empty() ? nullptr : d.cpu_part, d.sh_out, cm, T, D, d.moe_out, s);
        CK(cudaStreamWaitEvent(s, d.hc_done, 0));
        hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        // (h_part / h_x are written again only after the next layer's route has synchronized, which follows these)
        return true;
    }

    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::kernels::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        strata::kernels::sample_tokens(d.logits + (size_t) std::max(0, n_logits - 1) * g.vocab, 1, g.vocab, nullptr, 0,
                                       sp, d.tok, d.st);
        int t = -1;
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    bool logits(std::vector<float>& out) {
        if (logits_all) {
            out = chunk_logits;
            return true;
        }
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        out.resize((size_t) std::max(1, n_logits) * g.vocab);
        return cudaMemcpy(out.data(), d.logits, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void save_profile() const {
        if (std::getenv("DS4_CPU_STATS") != nullptr && cpu_calls > 0)
            std::fprintf(stderr, "CPU lane: %ld calls, %.2f experts and %.2f rows a call; per call: gate/up %.3f ms, "
                                 "bands %.3f, down %.3f, down bands %.3f\n", cpu_calls, (double) cpu_ents / cpu_calls,
                         (double) cpu_rows / cpu_calls, cpu_ph_ms[0] / cpu_calls, cpu_ph_ms[1] / cpu_calls,
                         cpu_ph_ms[2] / cpu_calls, cpu_ph_ms[3] / cpu_calls);
        if (std::getenv("DS4_CPU_STATS") != nullptr && rc_ticks > 0)
            std::fprintf(stderr, "recent slots: %lld ticks, %.3f ms a tick, %.2f fills started and %.2f mapped a tick\n",
                         (long long) rc_ticks, rc_tick_ms / rc_ticks, (double) rc_fills_started / rc_ticks,
                         (double) rc_fills_done / rc_ticks);
        if (std::getenv("DS4_CPU_STATS") != nullptr && cpu_calls > 0) {
            const int thr = pools.empty() ? 1 : (int) (pools[0]->size() + (pools.size() > 1 ? pools[1]->size() : 0));
            for (int i = 0; i < 4; ++i)
                std::fprintf(stderr, "  phase %d: %.1f jobs a call, %.1f us a job; the jobs' time %.0f%% of %d threads x the "
                                     "phase's wall\n", i, (double) cpu_job_n[i] / cpu_calls,
                             cpu_job_n[i] ? cpu_job_ns[i] / 1e3 / cpu_job_n[i] : 0.0,
                             100.0 * cpu_job_ns[i] / 1e6 / (cpu_ph_ms[i] * thr), thr);
        }
        if (o.profile.empty()) return;
        const std::string tmp = o.profile + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) return;
        const bool ok = std::fwrite(prof.data(), 4, prof.size(), f) == prof.size();
        std::fclose(f);
        if (ok) std::rename(tmp.c_str(), o.profile.c_str());
    }

    Tiers tiers() const {
        Tiers t;
        for (const Layer& l : L) {
            if (l.mtp) continue;
            t.vram += l.n_vram;
            t.slots += l.n_slots;
            t.cpu += l.n_exp - l.n_vram;
            for (int e = 0; e < l.n_exp; ++e) (l.slot[(size_t) e] >= 0 ? t.vram_gb : t.cpu_gb) += xc(l, e).blob / 1e9;
        }
        return t;
    }

    std::vector<std::string> info_lines() const {
        std::vector<std::string> v;
        const Tiers t = tiers();
        char b[512];
        std::snprintf(b, sizeof b, "INFO context=%lld kv=fp4 kv_resident=%lld expert_slots=%lld engine_kind=ds4 "
                                   "cpu_experts=%lld experts=%d ram_slots=%lld vram_gb=%.1f ram_gb=%.1f mtp=%d cpu_threads=%d "
                                   "engine=ds4-0.1",
                      (long long) o.max_context, (long long) o.max_context, (long long) t.slots, (long long) t.cpu,
                      g.nl * g.n_exp, (long long) t.cpu, t.vram_gb, t.cpu_gb, g.n_mtp > 0 ? 1 : 0, cpu_threads());
        v.push_back(b);
        // the Monitor's map of each GPU (serve/gpu_map.py, the GLM engine's VRAM lines): what each part holds on the
        // card, in bytes - the dense weights (the drafter's blocks apart), the KV caches, the window's buffers, the
        // drafter's blocks (their weights, a share estimated, and experts), the routed experts' slots, the vision
        // tower (in this process: vision_own); the prompt chunks' buffers (prompt); and tensor parallel's copies (tp,
        // out of weights and kv): of a group of tp_n, each member's (tp_n - 1) / tp_n of the tensors every member holds
        // whole and of its KV caches (every member has its layers' whole KV) - the drafter's few caches on the first
        // GPU at --tp 9 counted with them
        for (const Dev& d : devs) {
            size_t fr = 0, tot = 0;
            cudaSetDevice(d.id);
            cudaMemGetInfo(&fr, &tot);
            int a = -1, z = -1;
            long long slots = 0;
            size_t mtp_exp = 0, mtp_dense = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m) {
                    const Layer& x = member(l, m);
                    if (x.dev != (int) (&d - devs.data())) continue;
                    if (l.mtp) {
                        mtp_exp += slots_bytes(x);
                        mtp_dense += l.dense_bytes;
                        continue;
                    }
                    if (a < 0) a = l.il;
                    z = l.il;
                    slots += l.n_slots;
                }
            mtp_dense = std::min(mtp_dense, d.weights);
            mtp_exp = std::min(mtp_exp, d.experts);
            const size_t tn = groups[(size_t) d.grp].mem.size();
            const size_t tp_w = std::min(d.tp_w, d.weights - mtp_dense), tp_kv = tn > 1 ? d.caches / tn * (tn - 1) : 0;
            const size_t vis = img_tok >= 0 && vt.device() == d.id ? vt.device_bytes() : 0;
            char pci[32] = "-";
            if (cudaDeviceGetPCIBusId(pci, sizeof pci, d.id) != cudaSuccess) std::snprintf(pci, sizeof pci, "-");
            std::snprintf(b, sizeof b,
                          "VRAM dev=%d pci=%s total=%zu free=%zu weights=%zu kv=%zu kv_ram=0 act=%zu drafter=%zu window=0 "
                          "pool=%zu tail=0 borrow=0 vision=0 vision_own=%zu prompt=%zu tp=%zu tp_n=%zu slots=%lld ctx=%lld "
                          "layers=%d-%d",
                          d.id, pci, tot, fr, d.weights - mtp_dense - tp_w, d.caches - tp_kv, d.scratch, mtp_dense + mtp_exp,
                          d.experts - mtp_exp, vis, d.prompt, tp_w + tp_kv, tn, slots, (long long) o.max_context, a, z);
            v.push_back(b);
        }
        if (img_tok >= 0) {   // the server's image resize (DeepSeek's image_processor.py)
            std::snprintf(b, sizeof b,
                          "INFO vision=ds4 vision_patch=%d vision_merge=%d vision_min_pixels=%lld vision_max_tokens=%d "
                          "vision_max_wh_ratio=%lld image_token=%d vision_gpu=%d vision_mib=%lld",
                          vg.patch, vg.merge, (long long) pk.num("vision.min_pixels", 0), vg.max_tokens,
                          (long long) pk.num("vision.max_wh_ratio", 0), img_tok, vt.device(),
                          (long long) (vt.device_bytes() >> 20));
            v.push_back(b);
        }
        return v;
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;
bool Model::load(const Options& o, std::string& err) { return p_->load(o, err); }
bool Model::step(int token, bool logits, std::string& err) { return p_->step(token, logits, err); }
bool Model::feed(const int* tokens, int n, int logits_rows, std::string& err) {
    Impl& p = *p_;
    if (p.chunk > 0 && n >= p.chunk_min) return p.forward_chunks(tokens, n, logits_rows, err);   // a long run
    p.logits_all = false;
    // (threads mode, Options::prefill_keep's approximate tail: the windows before it end at its start and run the
    // consumer layers not at all - no drafter seeding from them either; the tail's windows see only the tail there)
    const int tail = p.par ? p.tail_rows(n) : n;
    const int64_t lo = tail < n ? p.pos + n - tail : 0;
    bool ok = true;
    for (int a = 0; a < n && ok;) {
        p.win_skip = a < n - tail;
        p.win_lo = lo;
        const int T = std::min(kWin, (p.win_skip ? n - tail : n) - a);
        ok = p.forward(tokens + a, T, a + T == n ? logits_rows : 0, err, p.win_skip ? 0 : -1);
        a += T;
    }
    p.win_skip = false;
    p.win_lo = 0;
    return ok;
}
int Model::feed_unit() const { return p_->chunk > 0 ? std::max(p_->chunk, (int) p_->o.max_context) : kWin; }
void Model::set_progress(std::function<void(int64_t)> f) { p_->progress = std::move(f); }
int Model::sample(float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter) {
    return p_->sample(t, tp, tk, mp, seed, counter);
}
bool Model::logits(std::vector<float>& out) { return p_->logits(out); }
int Model::image_token() const { return p_->img_tok; }
int Model::dim() const { return p_->g.dim; }
bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err) {
    return p_->encode_image(rgb, width, height, span, err);
}
void Model::set_images(std::vector<Image> images) {
    std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) { return a.start < b.start; });
    p_->imgs = std::move(images);
}
int Model::draft_block() const { return p_->g.n_mtp > 0 ? p_->g.d_block : 0; }
bool Model::draft(int y, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter, std::vector<int>& d,
                  std::vector<float>& c, std::string& err) {
    Impl::Samp sm;
    sm.temperature = t;
    sm.top_p = tp;
    sm.top_k = tk;
    sm.min_p = mp;
    sm.seed = seed;
    sm.counter = counter;
    return p_->draft(y, sm, d, c, err);
}
bool Model::verify(int y, const std::vector<int>& d, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter,
                   int& accepted, int& next, std::string& err) {
    Impl::Samp sm;
    sm.temperature = t;
    sm.top_p = tp;
    sm.top_k = tk;
    sm.min_p = mp;
    sm.seed = seed;
    sm.counter = counter;
    return p_->verify(y, d, sm, accepted, next, err);
}
int64_t Model::pos() const { return p_->pos; }
int64_t Model::max_context() const { return p_->o.max_context; }
int Model::vocab() const { return p_->g.vocab; }
const std::vector<int32_t>& Model::tokens() const { return p_->toks; }
void Model::reset() { p_->reset(); }
void Model::checkpoint(int slot) { p_->checkpoint(slot); }
bool Model::restore(int64_t limit) { return p_->restore(limit); }
int64_t Model::restorable(int64_t limit) const { return p_->restorable(limit); }
void Model::set_prefill_keep(int keep) { p_->o.prefill_keep = keep; }
void Model::set_draft_sample(int on) { p_->draft_sample = on < 0 ? p_->draft_sample_env : on != 0; }
void Model::save_profile() const { p_->save_profile(); }
const Stats& Model::stats() const { return p_->st; }
Tiers Model::tiers() const { return p_->tiers(); }
std::string Model::describe() const { return p_->describe(); }
std::vector<std::string> Model::info_lines() const { return p_->info_lines(); }

}  // namespace strata::ds4
