// src/ds4/ds4_uring.hpp - a batch of file reads in flight at once (io_uring, the raw system calls: no liburing), for
// the decode's Engram rows - a window's ~250 random reads of a few hundred bytes from tables far larger than the page
// cache, which a pool of threads each blocking in pread kept ~22 deep on the NVMe (~1 ms a window).  Linux only; where
// the kernel has no io_uring (or forbids it) init() fails and the caller reads as before.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__linux__) && __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#define DS4_URING 1
#endif

namespace strata::ds4 {

struct UringRead {
    int fd = -1;
    uint64_t off = 0;
    uint32_t n = 0;
    void* dst = nullptr;
};

class Uring {
public:
    Uring() = default;
    Uring(const Uring&) = delete;
    Uring& operator=(const Uring&) = delete;
    ~Uring() {
#if defined(DS4_URING)
        if (sqes_ != nullptr) munmap(sqes_, sqes_sz_);
        if (cq_ptr_ != nullptr && cq_ptr_ != sq_ptr_) munmap(cq_ptr_, cq_sz_);
        if (sq_ptr_ != nullptr) munmap(sq_ptr_, sq_sz_);
        if (fd_ >= 0) close(fd_);
#endif
    }

    bool init(unsigned entries) {
#if defined(DS4_URING)
        io_uring_params p;
        std::memset(&p, 0, sizeof(p));
        fd_ = (int) syscall(__NR_io_uring_setup, entries, &p);
        if (fd_ < 0) return false;
        n_ = p.sq_entries;
        sq_sz_ = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_sz_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) sq_sz_ = cq_sz_ = sq_sz_ > cq_sz_ ? sq_sz_ : cq_sz_;
        sq_ptr_ = mmap(nullptr, sq_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
        if (sq_ptr_ == MAP_FAILED) {
            sq_ptr_ = nullptr;
            return false;
        }
        cq_ptr_ = single ? sq_ptr_
                         : mmap(nullptr, cq_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_CQ_RING);
        if (cq_ptr_ == MAP_FAILED) {
            cq_ptr_ = nullptr;
            return false;
        }
        sqes_sz_ = p.sq_entries * sizeof(io_uring_sqe);
        void* s = mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES);
        if (s == MAP_FAILED) return false;
        sqes_ = (io_uring_sqe*) s;
        char* sq = (char*) sq_ptr_;
        char* cq = (char*) cq_ptr_;
        sq_tail_ = (unsigned*) (sq + p.sq_off.tail);
        sq_mask_ = *(unsigned*) (sq + p.sq_off.ring_mask);
        sq_array_ = (unsigned*) (sq + p.sq_off.array);
        cq_head_ = (unsigned*) (cq + p.cq_off.head);
        cq_tail_ = (unsigned*) (cq + p.cq_off.tail);
        cq_mask_ = *(unsigned*) (cq + p.cq_off.ring_mask);
        cqes_ = (io_uring_cqe*) (cq + p.cq_off.cqes);
        ok_ = true;
        return true;
#else
        (void) entries;
        return false;
#endif
    }
    bool ok() const { return ok_; }

    // every read in full: up to the ring's entries in flight at a time; a short or failed one again by pread
    bool read_all(const std::vector<UringRead>& rs) {
#if defined(DS4_URING)
        if (!ok_) return false;
        bool good = true;
        for (size_t a = 0; a < rs.size();) {
            const unsigned m = (unsigned) std::min<size_t>(n_, rs.size() - a);
            unsigned tail = __atomic_load_n(sq_tail_, __ATOMIC_RELAXED);
            for (unsigned i = 0; i < m; ++i) {
                const UringRead& r = rs[a + i];
                const unsigned idx = tail & sq_mask_;
                io_uring_sqe& e = sqes_[idx];
                std::memset(&e, 0, sizeof(e));
                e.opcode = IORING_OP_READ;
                e.fd = r.fd;
                e.off = r.off;
                e.addr = (uint64_t) (uintptr_t) r.dst;
                e.len = r.n;
                e.user_data = a + i;
                sq_array_[idx] = idx;
                ++tail;
            }
            __atomic_store_n(sq_tail_, tail, __ATOMIC_RELEASE);
            unsigned done = 0, submitted = 0;
            while (done < m) {
                const unsigned to_submit = m - submitted;
                const long r = syscall(__NR_io_uring_enter, fd_, to_submit, m - done, IORING_ENTER_GETEVENTS, nullptr, 0);
                if (r < 0) return false;
                submitted += (unsigned) r;
                unsigned head = __atomic_load_n(cq_head_, __ATOMIC_RELAXED);
                const unsigned ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
                for (; head != ctail; ++head) {
                    const io_uring_cqe& c = cqes_[head & cq_mask_];
                    const UringRead& rr = rs[(size_t) c.user_data];
                    if (c.res != (int) rr.n) good = good && repread(rr, c.res);
                    ++done;
                }
                __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
            }
            a += m;
        }
        return good;
#else
        (void) rs;
        return false;
#endif
    }

private:
#if defined(DS4_URING)
    static bool repread(const UringRead& r, int got) {   // (the rest of a short read, or all of a failed one)
        uint64_t k = got > 0 ? (uint64_t) got : 0;
        while (k < r.n) {
            const ssize_t q = ::pread(r.fd, (uint8_t*) r.dst + k, r.n - k, (off_t) (r.off + k));
            if (q <= 0) return false;
            k += (uint64_t) q;
        }
        return true;
    }
    io_uring_sqe* sqes_ = nullptr;
    io_uring_cqe* cqes_ = nullptr;
    unsigned *sq_tail_ = nullptr, *sq_array_ = nullptr, *cq_head_ = nullptr, *cq_tail_ = nullptr;
    unsigned sq_mask_ = 0, cq_mask_ = 0, n_ = 0;
    void *sq_ptr_ = nullptr, *cq_ptr_ = nullptr;
    size_t sq_sz_ = 0, cq_sz_ = 0, sqes_sz_ = 0;
#endif
    int fd_ = -1;
    bool ok_ = false;
};

}  // namespace strata::ds4
