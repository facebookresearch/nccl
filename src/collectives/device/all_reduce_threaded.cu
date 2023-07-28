// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "all_reduce.h"
#include "all_reduce_threaded.h"
#include "common.h"
#include "collectives.h"

#define idx(nranks, i, j) ((i) * (nranks) + (j))

template <typename T>
static inline __device__ uint32_t vecElementAdd(const uint32_t& a, const uint32_t& b)
{
    if (std::is_same<T, half>::value) {
        const __half *x = reinterpret_cast<const __half*>(&a);
        const __half *y = reinterpret_cast<const __half*>(&b);

        __half2 p = __halves2half2(x[0], x[1]);
        __half2 q = __halves2half2(y[0], y[1]);

        __half2 z = __hadd2(p, q);
        return (reinterpret_cast<uint32_t *>(&z))[0];
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (std::is_same<T, __nv_bfloat16>::value) {
        const __nv_bfloat16 *x = reinterpret_cast<const __nv_bfloat16*>(&a);
        const __nv_bfloat16 *y = reinterpret_cast<const __nv_bfloat16*>(&b);

#if (__CUDA_ARCH__ >= 800)
        __nv_bfloat162 p = { x[0], x[1] };
        __nv_bfloat162 q = { y[0], y[1] };

        __nv_bfloat162 z = __hadd2(p, q);
        return (reinterpret_cast<uint32_t *>(&z))[0];
#else
        __nv_bfloat16 z[2] = { x[0] + y[0], x[1] + y[1] };
        return (reinterpret_cast<uint32_t *>(z))[0];
#endif
    }
#endif

    return 0;
}

/* create a special version of seqAdd that can be disabled at
 * compile-time for bfloat16 (using enable_if).  This is because the
 * base version of seqAdd does not compile for bfloat16, so we are
 * essentially tricking the compiler.  We never call this version for
 * bfloat16, so it doesn't matter that it does not compile, but the
 * compiler unfortunately does not know that. */
#if defined(__CUDA_BF16_TYPES_EXIST__)
template <typename T, uint32_t NRANKS>
static inline __device__
typename std::enable_if<!std::is_same<T,__nv_bfloat16>::value, uint4>::type
seqAdd(const T **src, size_t offset)
{
    T dst[16 / sizeof(T)] = { 0 };
    for (int i = 0; i < NRANKS; i++) {
        uint4 vals = reinterpret_cast<const uint4 *>(&src[i][offset])[0];
        const T *src_d = reinterpret_cast<const T *>(&vals);
        for (int j = 0; j < 16 / sizeof(T); j++) {
            dst[j] += src_d[j];
        }
    }
    return reinterpret_cast<uint4 *>(&dst)[0];
}

template <typename T, uint32_t NRANKS>
static inline __device__
typename std::enable_if<std::is_same<T,__nv_bfloat16>::value, uint4>::type
seqAdd(const T **src, size_t offset)
{
    uint4 x = { 0, 0, 0, 0 };

    return x;
}

#else

template <typename T, uint32_t NRANKS>
static inline __device__ uint4 seqAdd(const T **src, size_t offset)
{
    T dst[16 / sizeof(T)] = { 0 };
    for (int i = 0; i < NRANKS; i++) {
        /* 16-byte load */
        uint4 vals = reinterpret_cast<const uint4 *>(&src[i][offset])[0];

        /* sequential additions */
        const T *src_d = reinterpret_cast<const T *>(&vals);
        for (int j = 0; j < 16 / sizeof(T); j++) {
            dst[j] += src_d[j];
        }
    }
    return reinterpret_cast<uint4 *>(&dst)[0];
}

#endif

template <typename T, uint32_t NRANKS>
static inline __device__ uint4 vecAdd(const T **src, size_t offset)
{
    if (std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
        || std::is_same<T, __nv_bfloat16>::value
#endif
    ) {
        uint4 dst = { 0, 0, 0, 0 };
        for (int i = 0; i < NRANKS; i++) {
            /* 16-byte load */
            uint4 vals = reinterpret_cast<const uint4 *>(&src[i][offset])[0];

            /* vector additions */
            dst.x = vecElementAdd<T>(dst.x, vals.x);
            dst.y = vecElementAdd<T>(dst.y, vals.y);
            dst.z = vecElementAdd<T>(dst.z, vals.z);
            dst.w = vecElementAdd<T>(dst.w, vals.w);
        }
        return dst;
    } else {
        return seqAdd<T,NRANKS>(src, offset);
    }
}

template <typename T>
static inline __device__ uint4 vecAdd(const T *src_a, const T *src_b)
{
    /* 16-byte loads */
    uint4 vals_a = reinterpret_cast<const uint4 *>(src_a)[0];
    uint4 vals_b = reinterpret_cast<const uint4 *>(src_b)[0];

    if (std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
        || std::is_same<T, __nv_bfloat16>::value
#endif
    ) {
        /* vector additions */
        uint4 dst;
        dst.x = vecElementAdd<T>(vals_a.x, vals_b.x);
        dst.y = vecElementAdd<T>(vals_a.y, vals_b.y);
        dst.z = vecElementAdd<T>(vals_a.z, vals_b.z);
        dst.w = vecElementAdd<T>(vals_a.w, vals_b.w);
        return dst;
    } else {
        /* cast back to original type and do sequential additions */
        T dst[16 / sizeof(T)];
        const T *src_a_loaded = reinterpret_cast<const T *>(&vals_a);
        const T *src_b_loaded = reinterpret_cast<const T *>(&vals_b);
        for (int j = 0; j < 16 / sizeof(T); j++) {
            dst[j] = src_a_loaded[j] + src_b_loaded[j];
        }
        return reinterpret_cast<uint4 *>(&dst)[0];
    }
}

/*
 * Barrier Algorithm --
 * Consider the barrier mailbox as a 2D array (numranks x numranks).
 * A group of threads in each rank (global thread ID < numranks), set
 * their local source buffer address in the column corresponding to
 * their rank.  Then a group of threads in each block (block local
 * thread ID < numranks), check to see if the row corresponding to
 * their rank is set.  Finally, all threads in the block synchronize.
 *
 * Memory consistency --
 * Because the barrier mailbox is volatile, we do not need to worry
 * about register caching and the barrier data will always be
 * propagated through cache consistency.  Typically, we would store
 * the source buffer address in a buffer and then perform a barrier
 * synchronization to let the remaining ranks know that the source
 * buffer is ready.  However, because the GPU only maintains weak
 * ordering of store operations, the compiler or hardware could
 * reorder the store of the source buffer from the barrier store,
 * which can lead to incorrect results.  We would need to separate
 * these operations with a __threadfence_system() call to ensure store
 * ordering, which is expensive.  We workaround that by merging the
 * source buffer distribution with the barrier operation by using a
 * bit OR'ed combination of the source buffer address and the barrier
 * flag simultaneously.  Per the CUDA memory consistency semantics
 * defined in
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions,
 * this should ensure that the barrier flag and the source buffer
 * should both be visible to the other ranks simultaneously.
 *
 * Mailbox reuse across collective operations --
 * The barrier mailbox might be reused across collective operations,
 * so we need to ensure that we are not reading the buffer address
 * from the previous iteration.  Because our algorithm is only
 * applicable for 16-byte aligned addresses, the last four bits of the
 * buffer address are always zero.  We use the last bit of the address
 * buffer to indicate whether the data is valid or not, by swapping
 * between 1 and 0 between each iteration.  To retrieve the actual
 * source buffer address, we simply mask that last bit.
 */
template <uint32_t NRANKS>
static inline __device__ void barrier(uintptr_t *barrierPtrs, uintptr_t barrierFlag, int rank)
{
    volatile uintptr_t *barrier_d = barrierPtrs;

    if (threadIdx.x < NRANKS) {
        /* first block sets barrier values */
        if (blockIdx.x == 0) {
            barrier_d[idx(NRANKS, threadIdx.x, rank)] = barrierFlag;
        }

        /* all blocks check for values to be set */
        while ((barrier_d[idx(NRANKS, rank, threadIdx.x)] & 1UL) != (barrierFlag & 1UL)) {}
    }

    /* remaining threads in the block wait */
    __syncthreads();
}

/* We use a simple Allgather + local reduce algorithm here.  For small
 * messages, we are mostly latency bound on fast networks such as
 * NVLink.  So fetching data from all the GPUs simultaneously should
 * basically take the same amount of time as fetching data from one
 * GPU.  This algorithm directly reads data from the other GPUs and
 * reduces it into the local destination buffer. */
template <typename T, uint32_t NRANKS>
static inline __device__ void allreduceFlat(uintptr_t *barrierMbox, uintptr_t barrierFlag, int rank,
                                            const T *sendbuff, T *recvbuff, size_t count)
{
    const int gtidx = threadIdx.x + blockDim.x * blockIdx.x;

    /* global barrier */
    barrier<NRANKS>(barrierMbox, (reinterpret_cast<uintptr_t>(sendbuff)) | barrierFlag, rank);

    /* fetch remote source buffer addresses from the barrier mailbox */
    const T *src[NRANKS];
    for (int i = 0; i < NRANKS; i++) {
        src[i] = reinterpret_cast<const T *>
            (barrierMbox[idx(NRANKS, rank, (rank + i) & (NRANKS - 1))] & ~1UL);
    }

    for (size_t offset = gtidx * 16 / sizeof(T); offset < count; offset += gridDim.x * blockDim.x * 16 / sizeof(T)) {
        reinterpret_cast<uint4 *>(&recvbuff[offset])[0] = vecAdd<T,NRANKS>(src, offset);
    }
}

/* Hierarchical algorithm for slightly larger (but still
 * latency-sensitive) messages.  In this algorithm, we avoid every
 * rank fetching all of the data from every other rank that the flat
 * algorithm above does.  Instead, each rank fetches only a subset of
 * data from all other ranks and reduces locally.  Then we do a second
 * step where the reduced data is Allgathered (by direct copy by each
 * rank). */
template <typename T, uint32_t NRANKS>
static inline __device__ void allreduceTree(uintptr_t *barrierMbox, uintptr_t barrierFlag, int rank,
                                             const T *sendbuff, T *tmpbuff, T *recvbuff, size_t count)
{
    const int gtidx = threadIdx.x + blockDim.x * blockIdx.x;

    /* global barrier */
    barrier<NRANKS>(barrierMbox, (reinterpret_cast<uintptr_t>(sendbuff)) | barrierFlag, rank);

    const T *src[NRANKS];
    for (int i = 0; i < NRANKS; i++) {
        int r = (rank + i) & (NRANKS - 1);
        src[i] = reinterpret_cast<const T *> (barrierMbox[idx(NRANKS, rank, r)] & ~1UL);
    }

    size_t offsetStart = gtidx * 16 / sizeof(T);
    size_t offsetMax = count / NRANKS;
    size_t offsetStride = NRANKS * gridDim.x * blockDim.x * 16 / sizeof(T);

    for (size_t offset = offsetStart; offset < offsetMax; offset += offsetStride) {
        reinterpret_cast<uint4 *>(&recvbuff[offset])[0] = vecAdd<T,NRANKS>(src, offset + rank * count / NRANKS);
    }

    /* we cannot avoid a __threadfence_system() here because the next
     * step requires us to access the data that just got reduced by
     * the other ranks.  So we need to tell the compiler/hardware to
     * not reorder the above reduction to happen after the below
     * Allgather. */
    __threadfence_system();

    /* global barrier */
    barrier<NRANKS>(barrierMbox + NRANKS * NRANKS,
                    (reinterpret_cast<uintptr_t>(tmpbuff)) | barrierFlag, rank);

    int rankOffset[NRANKS];
    for (int i = 0; i < NRANKS; i++) {
        int r = (rank + i) & (NRANKS - 1);
        src[i] = reinterpret_cast<const T *> (barrierMbox[NRANKS * NRANKS + idx(NRANKS, rank, r)] & ~1UL);
        rankOffset[i] = r * count / NRANKS;
    }

    /* simple direct-access Allgather in 16-byte loads */
    for (size_t offset = offsetStart; offset < offsetMax; offset += offsetStride) {
        for (int i = 0; i < NRANKS; i++) {
            reinterpret_cast<uint4 *>(&recvbuff[offset + rankOffset[i]])[0] =
                reinterpret_cast<const uint4 *>(&src[i][offset])[0];
        }
    }
}

template <typename T, uint32_t NRANKS>
static inline __device__ void peerReduce(uintptr_t *localMbox, uintptr_t *peerMbox, T *tmpbuff, T *recvbuff,
                                         size_t count)
{
    volatile uintptr_t *peerMboxV = peerMbox;
    volatile uintptr_t *localMboxV = localMbox;

    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            *peerMboxV = reinterpret_cast<uintptr_t>(tmpbuff);
        }
        while (*localMboxV == 0) {}
    }
    __syncthreads();

    const T *src = reinterpret_cast<const T *>(*localMboxV);
    const int gtidx = threadIdx.x + blockDim.x * blockIdx.x;

    /* simple reduction with one peer rank */
    for (size_t offset = gtidx * 16 / sizeof(T); offset < count; offset += gridDim.x * blockDim.x * 16 / sizeof(T)) {
        reinterpret_cast<uint4 *>(&recvbuff[offset])[0] = vecAdd<T>(&src[offset], (const T *) &tmpbuff[offset]);
    }

    *localMbox = 0;
}

template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_Threaded_Flat(uintptr_t *barrierMbox,
                                                   uintptr_t barrierFlag, int rank,
                                                   const T *sendbuff, T *recvbuff, size_t count)
{
    allreduceFlat<T,NRANKS>(barrierMbox, barrierFlag, rank, sendbuff, recvbuff, count);
}

template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_Threaded_Tree(uintptr_t *barrierMbox, uintptr_t barrierFlag, int rank,
                                                   const T *sendbuff, T *tmpbuff, T *recvbuff, size_t count)
{
    allreduceTree<T,NRANKS>(barrierMbox, barrierFlag, rank, sendbuff, tmpbuff, recvbuff, count);
}

template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_Threaded_HCM_Flat(uintptr_t *cliqueBarrierMbox, uintptr_t *localMbox,
                                                       uintptr_t *peerMbox, uintptr_t barrierFlag, int cliqueRank,
                                                       const T *sendbuff, T *tmpbuff, T *recvbuff, size_t count)
{
    /* For HCM systems, we break the Allreduce into two parts.  In the
     * first part, we perform the Allreduce within the clique (the set
     * of ranks that are topologically all-to-all connected with
     * direct NVLink connections), i.e., within the "mesh".  Then each
     * rank reduces data with its peer on the other mesh, i.e., across
     * the "cube".  We only support two meshes currently: this should
     * be sufficient for ZionEx and Zion4S.  It is unclear if NVIDIA
     * has other platforms that have a more generalized version of
     * HCM, so this code does not support the fully general case of
     * multidimensional cubes. */
    allreduceFlat<T,NRANKS/2>(cliqueBarrierMbox, barrierFlag, cliqueRank, sendbuff, tmpbuff, count);
    __threadfence_system();
    peerReduce<T,NRANKS>(localMbox, peerMbox, tmpbuff, recvbuff, count);
}

template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_Threaded_HCM_Tree(uintptr_t *cliqueBarrierMbox, uintptr_t *localMbox,
                                                       uintptr_t *peerMbox, uintptr_t barrierFlag, int cliqueRank,
                                                       const T *sendbuff, T *tmpbuff, T *recvbuff, size_t count)
{
    /* using the recvbuff as a temporary buffer, so the output of
     * allreduce_tree goes into tmpbuff */
    allreduceTree<T,NRANKS/2>(cliqueBarrierMbox, barrierFlag, cliqueRank, sendbuff, recvbuff, tmpbuff, count);
    __threadfence_system();
    peerReduce<T,NRANKS>(localMbox, peerMbox, tmpbuff, recvbuff, count);
}

DECL_THREADED_FUNC(char);
DECL_THREADED_FUNC(uint8_t);
DECL_THREADED_FUNC(int32_t);
DECL_THREADED_FUNC(uint32_t);
DECL_THREADED_FUNC(int64_t);
DECL_THREADED_FUNC(uint64_t);
DECL_THREADED_FUNC(half);
DECL_THREADED_FUNC(float);
DECL_THREADED_FUNC(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_THREADED_FUNC(__nv_bfloat16);
#endif
