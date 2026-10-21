// zc_probe: what the PCIe link gives a kernel reading host memory directly (zero-copy), versus cudaMemcpy, and
// whether a read-only mmap of a model file can be registered with CUDA (the GPU then reads page-cache pages in place).
//   nvcc -O2 -arch=sm_86 -o zc_probe zc_probe.cu && ./zc_probe <model shard>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>

__global__ void sum_kernel(const int4* p, size_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int4 v = p[i];
        acc += (unsigned) (v.x ^ v.y ^ v.z ^ v.w);
    }
    if (acc == 0x123456789ull) atomicAdd(out, acc);   // (keeps the reads)
}

static double read_bw(const void* dptr, size_t bytes, int blocks, int reps) {
    unsigned long long* o;
    cudaMalloc(&o, 8);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    sum_kernel<<<blocks, 256>>>((const int4*) dptr, bytes / 16, o);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) sum_kernel<<<blocks, 256>>>((const int4*) dptr, bytes / 16, o);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaFree(o);
    return (double) bytes * reps / (ms / 1e3) / 1e9;
}

int main(int argc, char** argv) {
    const size_t MB = 1 << 20, bytes = 512 * MB;
    struct rlimit rl;
    getrlimit(RLIMIT_MEMLOCK, &rl);
    std::printf("RLIMIT_MEMLOCK soft %llu hard %llu\n", (unsigned long long) rl.rlim_cur, (unsigned long long) rl.rlim_max);
    void *h, *d_h, *d;
    cudaHostAlloc(&h, bytes, cudaHostAllocMapped);
    std::memset(h, 1, bytes);
    cudaHostGetDevicePointer(&d_h, h, 0);
    cudaMalloc(&d, bytes);
    for (int blocks : {82, 164, 328, 656, 1312})
        std::printf("zero-copy read of cudaHostAlloc memory, %4d blocks: %.1f GB/s\n", blocks, read_bw(d_h, bytes, blocks, 5));
    std::printf("VRAM read: %.1f GB/s\n", read_bw(d, bytes, 656, 20));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < 5; ++r) cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    std::printf("cudaMemcpy H2D from pinned: %.1f GB/s\n", 5.0 * bytes / (ms / 1e3) / 1e9);
    // small transfers: one 7.3 MB expert
    const size_t ex = 7300 * 1024;
    cudaEventRecord(a);
    for (int r = 0; r < 50; ++r) cudaMemcpyAsync(d, h, ex, cudaMemcpyHostToDevice);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms, a, b);
    std::printf("cudaMemcpy H2D one 7.3 MB expert: %.3f ms (%.1f GB/s)\n", ms / 50, 50.0 * ex / (ms / 1e3) / 1e9);
    std::printf("zero-copy read of a 7.3 MB region, 656 blocks: %.1f GB/s\n", read_bw(d_h, ex, 656, 50));
    if (argc > 1) {
        int fd = open(argv[1], O_RDONLY);
        struct stat st;
        fstat(fd, &st);
        const size_t len = st.st_size & ~(size_t) (4096 - 1);
        void* m = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
        const size_t off = 1024 * MB, reg = 512 * MB;
        char* p = (char*) m + off;
        volatile char sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < reg; i += 4096) sink += p[i];   // (fault it in)
        auto t1 = std::chrono::steady_clock::now();
        std::printf("faulted in %zu MB of %s in %.0f ms\n", reg / MB, argv[1],
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
        cudaError_t e = cudaHostRegister(p, reg, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        std::printf("cudaHostRegister(read-only mmap of the file, %zu MB): %s\n", reg / MB, cudaGetErrorString(e));
        if (e == cudaSuccess) {
            void* dp;
            cudaHostGetDevicePointer(&dp, p, 0);
            std::printf("zero-copy read of the registered file pages: %.1f GB/s\n", read_bw(dp, reg, 656, 5));
            auto t2 = std::chrono::steady_clock::now();
            cudaHostUnregister(p);
            std::printf("unregister: %.1f ms\n",
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count());
        } else {
            cudaGetLastError();
        }
        munmap(m, len);
        close(fd);
    }
    return 0;
}
