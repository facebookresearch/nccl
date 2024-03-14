// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "all_reduce_dda.h"
#include "dda_kernel.cuh"

/*
 * We use a simple Allgather + local reduce algorithm here.  For small
 * messages, we are mostly latency bound on fast networks such as
 * NVLink.  So fetching data from all the GPUs simultaneously should
 * basically take the same amount of time as fetching data from one
 * GPU.  This algorithm directly reads data from the other GPUs and
 * reduces it into the local destination buffer.
 */
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Flat(
    uintptr_t barrierFlag,
    DdaDeviceState* devStates,
    int rank,
    const T* sendbuff,
    T* recvbuff,
    size_t count) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  // always use rank0's barrierMbox as the shared barrier
  uintptr_t* mbox = devStates[0].threadedBarrierMbox;
  barrier_uponKernelLaunch<NRANKS>(
      mbox,
      (reinterpret_cast<uintptr_t>(sendbuff)) | barrierFlag,
      rank);

  const T* srcs[NRANKS];
  for (int i = 0; i < NRANKS; i++) {
    int nbrRank = (rank + i) & (NRANKS - 1);
    srcs[i] = reinterpret_cast<const T*>(mbox[nbrRank] & ~1UL);
  }

  const size_t countPerThread = 16 / sizeof(T);
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = count;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    reinterpret_cast<uint4*>(&recvbuff[idx])[0] =
      vecAdd<T, NRANKS>(srcs, idx);
  }

  barrier_onSameBlockIdx<NRANKS>(
      mbox + NRANKS,
      barrierFlag,
      rank);
}

template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Flat_ipc(
    uintptr_t barrierFlag,
    DdaDeviceState* devStates,
    int rank,
    T* recvbuff,
    size_t count) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  // always use rank0's barrierMbox as the shared barrier
  uintptr_t* mbox = devStates[0].ipcBarrierMbox;
  uintptr_t flag = barrierFlag;
  barrier_uponKernelLaunch_ipc<NRANKS>(mbox, flag, rank);
  flag++;

  const T* srcs[NRANKS];
  for (int i = 0; i < NRANKS; i++) {
    int nbrRank = (rank + i) & (NRANKS - 1);
    srcs[i] = reinterpret_cast<const T*>(devStates[nbrRank].tmpbuff);
  }

  const size_t countPerThread = 16 / sizeof(T);
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = count;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    reinterpret_cast<uint4*>(&recvbuff[idx])[0] =
      vecAdd<T, NRANKS>(srcs, idx);
  }

  barrier_onSameBlockIdx_ipc<NRANKS>(mbox, flag, rank);
}

template <typename T, uint32_t NRANKS>
static inline __device__ void reduceScatter(
    uintptr_t* mbox,
    int rank,
    T* recvbuff,
    size_t recvcount) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const T* srcs[NRANKS];
  for (int i = 0; i < NRANKS; ++i) {
    int nbrRank = (rank + i) & (NRANKS - 1);
    srcs[i] = reinterpret_cast<const T*>(mbox[nbrRank] & ~1UL);
  }

  // direct-access reduce data on rank-th block with 16-byte loads
  const size_t countPerThread = 16 / sizeof(T);
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = recvcount;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    reinterpret_cast<uint4*>(&recvbuff[idx])[0] =
        vecAdd<T, NRANKS>(srcs, idx + rank * recvcount);
  }
}

template <typename T, uint32_t NRANKS>
static inline __device__ void allGather(
    uintptr_t* mbox,
    int rank,
    T* recvbuff,
    size_t sendcount) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const T* srcs[NRANKS];
  int rankOffset[NRANKS];
  const uintptr_t* mboxOnThisBlock = mbox + blockIdx.x * NRANKS;
  for (int i = 0; i < NRANKS; ++i) {
    int nbrRank = (rank + i) & (NRANKS - 1);
    srcs[i] = reinterpret_cast<const T*>(mboxOnThisBlock[nbrRank] & ~1UL);
    rankOffset[i] = nbrRank * sendcount;
  }

  // direct-access all-gather with 16-byte loads
  const size_t countPerThread = 16 / sizeof(T);
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = sendcount;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  for (int i = 0; i < NRANKS; ++i) {
    for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
      reinterpret_cast<uint4*>(&recvbuff[idx + rankOffset[i]])[0] = reinterpret_cast<const uint4*>(&srcs[i][idx])[0];
    }
  }
}

/*
 * Hierarchical algorithm for large messages.  In this algorithm, we
 * avoid every rank fetching all of the data from every other rank
 * that the flat algorithm above does.  Instead, we do two steps:
 * - step1: (reduce-scatter)
 * each rank fetches only a subset of data
 * from all other ranks and reduces locally.
 * - step2: (all-gather)
 * Then we do a second step where the reduced data is Allgathered (by
 * direct copy by each rank).
 */
template <typename T, uint32_t NRANKS>
__global__ void __launch_bounds__(1024) ncclKernel_AllReduce_DDA_Tree(
    uintptr_t barrierFlag,
    DdaDeviceState* devStates,
    int rank,
    const T* sendbuff,
    T* recvbuff,
    size_t count) {
  // always use rank0's barrierMbox as the shared barrier
  uintptr_t* mbox = devStates[0].threadedBarrierMbox;

  // barrier to ensure every rank's sendbuff is ready to read
  barrier_uponKernelLaunch<NRANKS>(
    mbox,
    (reinterpret_cast<uintptr_t>(sendbuff)) | barrierFlag,
    rank);

  const size_t chunkSize = count / NRANKS;

  reduceScatter<T, NRANKS>(
      mbox,
      rank,
      recvbuff + rank * chunkSize,
      chunkSize);

  // make sure the result from RS are observed by all threads in peer devices
  const T* agSendbuff = recvbuff + rank * chunkSize;
  barrier_onSameBlockIdx_releaseAcquire<NRANKS>(
      mbox + NRANKS,
      (reinterpret_cast<uintptr_t>(agSendbuff)) | barrierFlag,
      rank);

  allGather<T, NRANKS>(
      mbox + NRANKS,
      rank,
      recvbuff,
      chunkSize);

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier_onSameBlockIdx<NRANKS>(
      mbox + (1 + gridDim.x) * NRANKS, barrierFlag, rank);
}

// use devStates[rank].tmpbuff as sendbuff and reduce-scatter on recvbuff
template <typename T, uint32_t NRANKS>
static inline __device__ void reduceScatter_ipc(
    DdaDeviceState* devStates,
    int rank,
    T* recvbuff,
    size_t recvcount) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const T* srcs[NRANKS];
  for (int i = 0; i < NRANKS; ++i) {
    int nbrRank = (rank + i) & (NRANKS - 1);
    srcs[i] = reinterpret_cast<const T*>(devStates[nbrRank].tmpbuff);
  }

  // direct-access reduce data on rank-th block with 16-byte loads
  const size_t countPerThread = 16 / sizeof(T);
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = recvcount;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    reinterpret_cast<uint4*>(&recvbuff[idx])[0] =
        vecAdd<T, NRANKS>(srcs, idx + rank * recvcount);
  }
}

// all-gather ipc version
// use "devStates[rank].tmpbuff + rank * sendcount" as the sendbuff
template <typename T, uint32_t NRANKS>
static inline __device__ void allGather_ipc(
    DdaDeviceState* devStates,
    int rank,
    T* recvbuff,
    size_t sendcount) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const T* srcs[NRANKS];
  int rankOffset[NRANKS];
  for (int i = 0; i < NRANKS; ++i) {
    int nbrRank = (rank + i) & (NRANKS - 1);
    srcs[i] = reinterpret_cast<const T*>(devStates[nbrRank].tmpbuff) + nbrRank * sendcount;
    rankOffset[i] = nbrRank * sendcount;
  }

  // direct-access all-gather with 16-byte loads
  const size_t countPerThread = 16 / sizeof(T);
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = sendcount;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int i = 0; i < NRANKS; ++i) {
      reinterpret_cast<uint4*>(&recvbuff[idx + rankOffset[i]])[0] = reinterpret_cast<const uint4*>(&srcs[i][idx])[0];
    }
  }
}

template <typename T, uint32_t NRANKS>
__global__ void __launch_bounds__(1024) ncclKernel_AllReduce_DDA_Tree_ipc(
    uintptr_t barrierFlag,
    DdaDeviceState* devStates,
    int rank,
    T* recvbuff,
    size_t count) {
  // always use rank0's barrierMbox as the shared barrier
  uintptr_t* mbox = devStates[0].ipcBarrierMbox;
  uintptr_t flag = barrierFlag;

  barrier_uponKernelLaunch_ipc<NRANKS>(mbox, flag, rank);
  flag++;

  T* rsRecvbuff =
      reinterpret_cast<T*>(devStates[rank].tmpbuff) + rank * count / NRANKS;
  reduceScatter_ipc<T, NRANKS>(
    devStates,
    rank,
    rsRecvbuff,
    count / NRANKS);

  barrier_onSameBlockIdx_releaseAcquire_ipc<NRANKS>(mbox, flag, rank);
  flag++;

  allGather_ipc<T, NRANKS>(
    devStates,
    rank,
    recvbuff,
    count / NRANKS);

  barrier_onSameBlockIdx_ipc<NRANKS>(mbox, flag, rank);
}

/*
 * Scatter-Gather algorithm for large messages.  The general algorithm
 * flow is as follows:
 *
 * barrier
 * Scatter (using PUT)
 * barrier
 * Local Reduce
 * barrier
 * Gather (using GET)
 * barrier
 *
 * Primary advantages compared with Tree:
 * 1. It avoids an extra local copy operation.  This makes a small amount
 * of difference in performance for medium sized messages.
 * 2. It uses PUT for at least one of the communication operations,
 * which is a little bit faster than the GET operations used in the
 * Tree algorithm.
 */
template <typename T, uint32_t NRANKS>
__global__ void __launch_bounds__(1024) ncclKernel_AllReduce_DDA_ScatGat_ipc(
    uintptr_t barrierFlag,
    DdaDeviceState* devStates,
    int rank,
    T* sendbuff,
    T* recvbuff,
    size_t count) {
  // always use rank0's barrierMbox as the shared barrier
  uintptr_t* mbox = devStates[0].ipcBarrierMbox;
  uintptr_t flag = barrierFlag;
  barrier_uponKernelLaunch_ipc<NRANKS>(mbox, flag, rank);
  flag++;

  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const T* srcs[NRANKS];
  T* remoteTmpPut[NRANKS];
  T* remoteTmpGet[NRANKS];
  T* ltmp[NRANKS];
  T* dsts[NRANKS];
  for (int i = 0; i < NRANKS; ++i) {
    int nbrRank = (rank + i) & (NRANKS - 1);
    srcs[i] = sendbuff + nbrRank * count / NRANKS;
    dsts[i] = recvbuff + nbrRank * count / NRANKS;
    remoteTmpPut[i] = reinterpret_cast<T*>(devStates[nbrRank].tmpbuff) +
      rank * count / NRANKS;
    remoteTmpGet[i] = reinterpret_cast<T*>(devStates[nbrRank].tmpbuff) +
      nbrRank * count / NRANKS;
    ltmp[i] = reinterpret_cast<T*>(devStates[rank].tmpbuff) +
      i * count / NRANKS;
  }

  // direct-access all-to-all with 16-byte stores
  const size_t countPerThread = 16 / sizeof(T);
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = count / NRANKS;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int i = 0; i < NRANKS; ++i) {
      reinterpret_cast<uint4*>(&remoteTmpPut[i][idx])[0] =
        reinterpret_cast<const uint4*>(&srcs[i][idx])[0];
    }
  }

  barrier_onSameBlockIdx_releaseAcquire_ipc<NRANKS>(mbox, flag, rank);
  flag++;

  // local reduction
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    reinterpret_cast<uint4*>(&ltmp[rank][idx])[0] =
      vecAdd<T, NRANKS>((const T**) ltmp, idx);
  }

  barrier_onSameBlockIdx_releaseAcquire_ipc<NRANKS>(mbox, flag, rank);
  flag++;

  // direct-access all-gather with 16-byte loads
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int i = 0; i < NRANKS; ++i) {
      reinterpret_cast<uint4*>(&dsts[i][idx])[0] =
        reinterpret_cast<const uint4*>(&remoteTmpGet[i][idx])[0];
    }
  }

  barrier_onSameBlockIdx_ipc<NRANKS>(mbox, flag, rank);
}

DECL_DDA_FUNC(char);
DECL_DDA_FUNC(uint8_t);
DECL_DDA_FUNC(int32_t);
DECL_DDA_FUNC(uint32_t);
DECL_DDA_FUNC(int64_t);
DECL_DDA_FUNC(uint64_t);
DECL_DDA_FUNC(half);
DECL_DDA_FUNC(float);
DECL_DDA_FUNC(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_DDA_FUNC(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_DDA_FUNC(__nv_fp8_e4m3);
DECL_DDA_FUNC(__nv_fp8_e5m2);
#endif
