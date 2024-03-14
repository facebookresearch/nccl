// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include "nccl.h"

template <typename T>
static inline __device__ uint32_t
vecElementAdd(const uint32_t& a, const uint32_t& b) {
  if (std::is_same<T, half>::value) {
    const __half* x = reinterpret_cast<const __half*>(&a);
    const __half* y = reinterpret_cast<const __half*>(&b);

#if (__CUDA_ARCH__ >= 700)
    __half2 p = __halves2half2(x[0], x[1]);
    __half2 q = __halves2half2(y[0], y[1]);

    __half2 z = __hadd2(p, q);
    return (reinterpret_cast<uint32_t*>(&z))[0];
#else
    half z[2] = { __hadd(x[0], y[0]), __hadd(x[1], y[1]) };
    return (reinterpret_cast<uint32_t*>(z))[0];
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* y = reinterpret_cast<const __nv_bfloat16*>(&b);

#if (__CUDA_ARCH__ >= 800)
    __nv_bfloat162 p = {x[0], x[1]};
    __nv_bfloat162 q = {y[0], y[1]};

    __nv_bfloat162 z = __hadd2(p, q);
    return (reinterpret_cast<uint32_t*>(&z))[0];
#else
    __nv_bfloat16 z[2] = {x[0] + y[0], x[1] + y[1]};
    return (reinterpret_cast<uint32_t*>(z))[0];
#endif
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && (__CUDA_ARCH__ >= 800) && defined(NCCL_ENABLE_FP8)
  } else if (std::is_same<T, __nv_fp8_e4m3>::value) {
    const __nv_fp8_e4m3* x = reinterpret_cast<const __nv_fp8_e4m3*>(&a);
    const __nv_fp8_e4m3* y = reinterpret_cast<const __nv_fp8_e4m3*>(&b);
    __half2 r[2] = {
      __hadd2(__halves2half2(__half(x[0]), __half(x[1])),
          __halves2half2(__half(y[0]), __half(y[1]))),
      __hadd2(__halves2half2(__half(x[2]), __half(x[3])),
          __halves2half2(__half(y[2]), __half(y[3])))
    };
    __nv_fp8x4_e4m3 z(r[0], r[1]);
    return (reinterpret_cast<uint32_t*>(&z))[0];
  } else if (std::is_same<T, __nv_fp8_e5m2>::value) {
    const __nv_fp8_e5m2* x = reinterpret_cast<const __nv_fp8_e5m2*>(&a);
    const __nv_fp8_e5m2* y = reinterpret_cast<const __nv_fp8_e5m2*>(&b);
    __half2 r[2] = {
      __hadd2(__halves2half2(__half(x[0]), __half(x[1])),
          __halves2half2(__half(y[0]), __half(y[1]))),
      __hadd2(__halves2half2(__half(x[2]), __half(x[3])),
          __halves2half2(__half(y[2]), __half(y[3])))
    };
    __nv_fp8x4_e5m2 z(r[0], r[1]);
    return (reinterpret_cast<uint32_t*>(&z))[0];
#endif // #if defined(__CUDA_FP8_TYPES_EXIST__) && (__CUDA_ARCH__ >= 800) && defined(NCCL_ENABLE_FP8)
  }

  return 0;
}

/* create a special version of seqAdd that can be disabled at
 * compile-time for bfloat16 (using enable_if).  This is because the
 * base version of seqAdd does not compile for bfloat16, so we are
 * essentially tricking the compiler.  We never call this version for
 * bfloat16, so it doesn't matter that it does not compile, but the
 * compiler unfortunately does not know that. */
template <typename T, uint32_t NRANKS>
static inline __device__
typename std::enable_if<!std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
    && !std::is_same<T, __nv_bfloat16>::value
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    && !std::is_same<T, __nv_fp8_e4m3>::value
    && !std::is_same<T, __nv_fp8_e5m2>::value
#endif
    , uint4>::type
seqAdd(const T** src, size_t offset) {
  T dst[16 / sizeof(T)] = {0};
  for (int i = 0; i < NRANKS; i++) {
    uint4 vals = reinterpret_cast<const uint4*>(&src[i][offset])[0];
    const T* src_d = reinterpret_cast<const T*>(&vals);
    for (int j = 0; j < 16 / sizeof(T); j++) {
      dst[j] += src_d[j];
    }
  }
  return reinterpret_cast<uint4*>(&dst)[0];
}

template <typename T, uint32_t NRANKS>
static inline __device__
typename std::enable_if<std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
    || std::is_same<T, __nv_bfloat16>::value
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    || std::is_same<T, __nv_fp8_e4m3>::value
    || std::is_same<T, __nv_fp8_e5m2>::value
#endif
    , uint4>::type
seqAdd(const T** src, size_t offset) {
  uint4 x = {0, 0, 0, 0};

  return x;
}

template <typename T, uint32_t NRANKS>
static inline __device__ uint4 vecAdd(const T** src, size_t offset) {
  if (std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
      || std::is_same<T, __nv_bfloat16>::value
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
      || std::is_same<T, __nv_fp8_e4m3>::value
      || std::is_same<T, __nv_fp8_e5m2>::value
#endif
  ) {
    uint4 dst = {0, 0, 0, 0};
    for (int i = 0; i < NRANKS; i++) {
      /* 16-byte load */
      uint4 vals = reinterpret_cast<const uint4*>(&src[i][offset])[0];

      /* vector additions */
      dst.x = vecElementAdd<T>(dst.x, vals.x);
      dst.y = vecElementAdd<T>(dst.y, vals.y);
      dst.z = vecElementAdd<T>(dst.z, vals.z);
      dst.w = vecElementAdd<T>(dst.w, vals.w);
    }
    return dst;
  } else {
    return seqAdd<T, NRANKS>(src, offset);
  }
}

template <typename T>
static inline __device__
typename std::enable_if<std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
    || std::is_same<T, __nv_bfloat16>::value
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    || std::is_same<T, __nv_fp8_e4m3>::value
    || std::is_same<T, __nv_fp8_e5m2>::value
#endif
    , uint4>::type
vecAdd(const T* src_a, const T* src_b) {
  /* 16-byte loads */
  uint4 vals_a = reinterpret_cast<const uint4*>(src_a)[0];
  uint4 vals_b = reinterpret_cast<const uint4*>(src_b)[0];

  /* vector additions */
  uint4 dst;
  dst.x = vecElementAdd<T>(vals_a.x, vals_b.x);
  dst.y = vecElementAdd<T>(vals_a.y, vals_b.y);
  dst.z = vecElementAdd<T>(vals_a.z, vals_b.z);
  dst.w = vecElementAdd<T>(vals_a.w, vals_b.w);
  return dst;
}

template <typename T>
static inline __device__
typename std::enable_if<!std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
    && !std::is_same<T, __nv_bfloat16>::value
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    && !std::is_same<T, __nv_fp8_e4m3>::value
    && !std::is_same<T, __nv_fp8_e5m2>::value
#endif
    , uint4>::type
vecAdd(const T* src_a, const T* src_b) {
  /* 16-byte loads */
  uint4 vals_a = reinterpret_cast<const uint4*>(src_a)[0];
  uint4 vals_b = reinterpret_cast<const uint4*>(src_b)[0];

  /* cast back to original type and do sequential additions */
  T dst[16 / sizeof(T)];
  const T* src_a_loaded = reinterpret_cast<const T*>(&vals_a);
  const T* src_b_loaded = reinterpret_cast<const T*>(&vals_b);
  for (int j = 0; j < 16 / sizeof(T); j++) {
    dst[j] = src_a_loaded[j] + src_b_loaded[j];
  }
  return reinterpret_cast<uint4*>(&dst)[0];
}

// barrier_uponKernelLaunch synchronizes with other ranks to ensure data buffers
// are ready to be consumed. it only needs to wait for a single thread (one
// from each rank) to set the ready bit.
//
// this barrier is expected to be firstly called before performing collective
// algorithm
template <uint32_t NRANKS>
static inline __device__ void
barrier_uponKernelLaunch(uintptr_t* barrierMbox, uintptr_t barrierFlag, int rank) {
  volatile uintptr_t* barrier_d = barrierMbox;

  if (threadIdx.x < NRANKS) {
    // 1st block notify other ranks that kernel got launched (data
    // is ready to be consumed)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      barrier_d[rank] = barrierFlag;
    }

    // wait until all other ranks are ready
    while ((barrier_d[threadIdx.x] & 1UL) != (barrierFlag & 1UL)) {
    }
  }

  // sync remaining threads in this block
  __syncthreads();
}

// IPC version of the barrier just wants to make sure that we have
// crossed the barrier flag
template <uint32_t NRANKS>
static inline __device__ void
barrier_uponKernelLaunch_ipc(uintptr_t* barrierMbox, uintptr_t barrierFlag, int rank) {
  volatile uintptr_t* barrier_d = barrierMbox;

  if (threadIdx.x < NRANKS) {
    // 1st block notify other ranks that kernel got launched (data
    // is ready to be consumed)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      barrier_d[rank] = barrierFlag;
    }

    // wait until all other ranks are ready
    while (barrier_d[threadIdx.x] < barrierFlag) {
    }
  }

  // sync remaining threads in this block
  __syncthreads();
}

// release and acquire pattern
// ensure prior operations (e.g reduce-scatter) from the current thread visible
// to operations (e.g all-gather) from other threads.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=release#release-and-acquire-patterns
static inline __device__ void st_flag_release(uintptr_t& flag, uintptr_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b64 [%1], %0;" ::"l"(flag), "l"(flag_addr));
#else
    __threadfence_system();
    asm volatile("st.global.volatile.b64 [%1], %0;" ::"l"(flag), "l"(flag_addr));
#endif
}

static inline __device__ void ld_flag_acquire(uintptr_t& flag, uintptr_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.global.volatile.b64 %0, [%1];" : "=l"(flag) : "l"(flag_addr));
#endif
}

// barrier_onSameBlockIdx_releaseAcquire
// this barrier does inter-rank syncrhonization on the same block-idx. Each block
// is synchronizing with same block-id from all other ranks under the assumption
// that this block(X) only depends on data the other block(X) from remote ranks,
// which is true because the same chunk of data is always processed by the same block-id.
//
// It also does a release/acquire semantics to ensure operations prior to this
// barrier become visible to all other threads
//
// this barrier is expected to be called between Reduce-Scatter and All-Gather to
// ensure results from RS are visible to AG.
template <uint32_t NRANKS>
static inline __device__ void
barrier_onSameBlockIdx_releaseAcquire(
    uintptr_t* barrierMbox, uintptr_t barrierFlag, int rank) {
  __syncthreads();

  if (threadIdx.x < NRANKS) {
    // mark this block on this rank as ready
    if (threadIdx.x == 0) {
      st_flag_release(barrierFlag, barrierMbox + blockIdx.x * NRANKS + rank);
    }

    // wait until all other ranks with the same blockId are ready
    uintptr_t otherRankFlag = 0;
    do {
      ld_flag_acquire(otherRankFlag, barrierMbox + blockIdx.x * NRANKS + threadIdx.x);
    } while ((otherRankFlag & 1UL) != (barrierFlag & 1UL));
  }

  // sync remaining threads in this block
  __syncthreads();
}

// IPC version of the barrier just wants to make sure that we have
// crossed the barrier flag
template <uint32_t NRANKS>
static inline __device__ void
barrier_onSameBlockIdx_releaseAcquire_ipc(
    uintptr_t* barrierMbox, uintptr_t barrierFlag, int rank) {
  __syncthreads();

  if (threadIdx.x < NRANKS) {
    // mark this block on this rank as ready
    if (threadIdx.x == 0) {
      st_flag_release(barrierFlag, barrierMbox + blockIdx.x * NRANKS + rank);
    }

    // wait until all other ranks with the same blockId are ready
    uintptr_t otherRankFlag = 0;
    do {
      ld_flag_acquire(otherRankFlag, barrierMbox + blockIdx.x * NRANKS + threadIdx.x);
    } while (otherRankFlag < barrierFlag);
  }

  // sync remaining threads in this block
  __syncthreads();
}

// barrier_onSameBlockIdx
// similar as barrier_onSameBlockIdx_releaseAcquire except that it doesn't add
// release/acquire constraint on memory ordering.
//
// this barrier is expected to be called at the very last stage
template <uint32_t NRANKS>
static inline __device__ void
barrier_onSameBlockIdx(uintptr_t* barrierMbox, uintptr_t barrierFlag, int rank) {
  volatile uintptr_t* barrier_d = barrierMbox;

  __syncthreads();

  if (threadIdx.x < NRANKS) {
    // mark this block on this rank as ready
    if (threadIdx.x == 0) {
      barrier_d[blockIdx.x * NRANKS + rank] = barrierFlag;
    }

    // wait until all other ranks with the same blockId are ready
    while ((barrier_d[blockIdx.x * NRANKS + threadIdx.x] & 1UL) != (barrierFlag & 1UL)) {
    }
  }

  // sync remaining threads in this block
  __syncthreads();
}

// IPC version of the barrier just wants to make sure that we have
// crossed the barrier flag
template <uint32_t NRANKS>
static inline __device__ void
barrier_onSameBlockIdx_ipc(uintptr_t* barrierMbox, uintptr_t barrierFlag, int rank) {
  volatile uintptr_t* barrier_d = barrierMbox;

  __syncthreads();

  if (threadIdx.x < NRANKS) {
    // mark this block on this rank as ready
    if (threadIdx.x == 0) {
      barrier_d[blockIdx.x * NRANKS + rank] = barrierFlag;
    }

    // wait until all other ranks with the same blockId are ready
    while (barrier_d[blockIdx.x * NRANKS + threadIdx.x] < barrierFlag) {
    }
  }

  // sync remaining threads in this block
  __syncthreads();
}
