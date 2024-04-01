// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include "CtranAlgoDev.h"

__device__ __forceinline__ int loadInt(volatile int* ptr) {
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ void storeInt(volatile int* ptr, int val) {
  asm volatile("st.volatile.global.s32 [%0], %1;" ::"l"(ptr), "r"(val));
}

// Sender updates the step after copied data to internal buffer.
// Only thread 0 from each group is responsible for updating the step.
__device__ __forceinline__ void
setStep(CtranAlgoDeviceBufState* state, int groupIdx, int val) {
  // ensure all threads have finished before setting the step
  __syncthreads();
  // ensure data is visible to other devices before setting the step
  __threadfence_system();
  if (threadIdx.x == 0) {
    storeInt(&state->stepOnSameBlockIdx[groupIdx], val);
  }
}

// Receiver resets the step after copied data out from internal buffer.
// Only thread 0 from each group is responsible for updating the step.
__device__ __forceinline__ void
resetStep(CtranAlgoDeviceBufState* state, int groupIdx, int val) {
  // ensure all threads have finished before setting the step
  __syncthreads();
  if (threadIdx.x == 0) {
    storeInt(&state->stepOnSameBlockIdx[groupIdx], val);
  }
}

// Receiver waits for sender to update the step, indicating that data has been
// copied into internal buffer for receiver to consume. Only thread 0 from each
// group is responsible for updating the step.
__device__ __forceinline__ void
waitStep(CtranAlgoDeviceBufState* state, int groupIdx, int val) {
  if (threadIdx.x == 0) {
    int cur;
    do {
      cur = loadInt(&state->stepOnSameBlockIdx[groupIdx]);
    } while (cur != val);
  }
  // ensure all threads waiting for thread 0 to check step being updated
  __syncthreads();
}

// Thread blocks with the same groupIdx must own the same area of shared memory
// in order to avoid race conditions. When a send and receive have mismatching
// data types (one will have uint4 and the other will have T), each thread
// must access only the shared memory that it would in the uint4 case.
template <typename T1, typename T2>
__device__ __forceinline__ void
copy(T1* dst, const T2* src, size_t count, int groupIdx, int nGroups) {
  int numElemsPerUint4 = sizeof(uint4) / sizeof(T1);
  int uint4BlockDim = blockDim.x * numElemsPerUint4;
  for (int i = 0; i < numElemsPerUint4; i++) {
    int offset = i*blockDim.x;
    const int gtIdx = uint4BlockDim * groupIdx + threadIdx.x + offset;
    for (size_t idx = gtIdx; idx < count; idx += nGroups * uint4BlockDim) {
      dst[idx] = src[idx];
    }
  }
}

// uint4 doesn't have copy constructor for uint4 = volatile uint4
template <>
__device__ __forceinline__ void
copy<uint4, volatile uint4>(uint4* dst, const volatile uint4* src, size_t count, int groupIdx, int nGroups) {
  const int gtIdx = blockDim.x * groupIdx + threadIdx.x;
  for (size_t idx = gtIdx; idx < count; idx += nGroups * blockDim.x) {
    dst[idx].x = src[idx].x;
    dst[idx].y = src[idx].y;
    dst[idx].z = src[idx].z;
    dst[idx].w = src[idx].w;
  }
}

// Copy API with default groupIdx (== blockIdx.x) and ngroups (==gridDim.x)
template <typename T>
__device__ __forceinline__ void copy(T* dst, const T* src, size_t count) {
  copy(dst, src, count, blockIdx.x, gridDim.x);
}

// Checks whether the buffer and size are aligned to 16 bytes so that 16B
// aligned copy (i.e., copy<uint4>) can be used. It checks only one buffer as
// the other side is NCCL internal buffer which is always 16B aligned.
template <typename T>
__device__ __forceinline__ bool canCopy16(const T* buf, size_t count) {
  bool bufAligned = ((uintptr_t)buf % 16) == 0;
  bool sizeAligned = ((size_t)count * sizeof(T) % 16) == 0;
  return bufAligned && sizeAligned;
}

// Checks whether two-sides buffers and size are aligned to 16 bytes so that 16B
// aligned copy (i.e., copy<uint4>) can be used. It is used in self-copy or NVL
// zero-copy cases where both buffers are provided by user.
template <typename T>
__device__ __forceinline__ bool
canCopy16(const T* sendbuff, T* recvbuff, size_t count) {
  bool sendBuffAligned = ((uintptr_t)sendbuff % 16) == 0;
  bool recvBuffAligned = ((uintptr_t)recvbuff % 16) == 0;
  bool sizeAligned = ((size_t)count * sizeof(T) % 16) == 0;
  return sendBuffAligned && recvBuffAligned && sizeAligned;
}

// Sender side of NVL copy to handle varying data count. When count is larger
// than the internal buffer, it is split into multiple steps.
template <typename T>
__device__ __forceinline__ void multiStepsSend(
    const T* sendbuff,
    size_t count,
    T* buf,
    CtranAlgoDeviceBufState* bufState,
    size_t bufCount,
    int groupIdx,
    int ngroups) {
  size_t offset = 0;
  int step = 0;
  while (offset < count) {
    size_t pendingSendCount = count - offset;
    size_t stepCount =
        pendingSendCount > bufCount ? bufCount : pendingSendCount;
    const T* srcPtr = sendbuff + offset;

    // The deviceState is dedicated to localRank->sendPeer NVL copy. Thus,
    // waiting for CTRAN_ALGO_STEP_RESET ensures the completion of previous
    // kernel and also previous step in the current kernel
    waitStep(bufState, groupIdx, CTRAN_ALGO_STEP_RESET);
    // P2P from local src to remote shared region
    copy<T, T>(buf, srcPtr, stepCount, groupIdx, ngroups);
    setStep(bufState, groupIdx, step);
    offset += stepCount;
    step++;
  }
}

// Receiver side of NVL copy to handle varying data count. When count is larger
// than the internal buffer, it is split into multiple steps.
template <typename T>
__device__ __forceinline__ void multiStepsRecv(
    T* recvbuff,
    size_t count,
    const T* buf,
    CtranAlgoDeviceBufState* bufState,
    size_t bufCount,
    int groupIdx,
    int ngroups) {
  size_t offset = 0;
  int step = 0;
  // must be volatile so compiler doesn't reuse cached buf values
  const volatile T* vbuf = reinterpret_cast<const volatile T*>(buf);
  while (offset < count) {
    size_t pendingCount = count - offset;
    size_t stepCount = pendingCount > bufCount ? bufCount : pendingCount;
    T* dstPtr = recvbuff + offset;

    waitStep(bufState, groupIdx, step);
    // D2D from local shared region to local dst
    copy<T, volatile T>(dstPtr, vbuf, stepCount, groupIdx, ngroups);
    resetStep(bufState, groupIdx, CTRAN_ALGO_STEP_RESET);
    offset += stepCount;
    step++;
  }
}
