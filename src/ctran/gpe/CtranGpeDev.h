// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_DEV_H_
#define CTRAN_GPE_DEV_H_

#include <stdint.h>
#include "CtranAlgoDev.h"

struct alignas(16) KernelP2pElem {
  size_t count{0};
  size_t displ{0};
  int peerRank{-1};
  // set by algorithm when submitting a GPE kernel; reclaim will check inuse
  // flags[0:ngroups-1]
  int ngroups{0};
  // set to true when submitting with a GPE kernel; reset by
  // each kernel thread block when done
  volatile bool inuse[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  // allow kernel to access next element in the list
  KernelP2pElem* next{nullptr};
};

struct CtranKernelAllGatherArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t nbytes;
};

// TODO: Placeholder for sendrecv for now since we don't pass NVL sendrecv to
// ctran. Define actual arguments when supporting NVL sendrecv with p2pElems
// pool
struct CtranKernelSendArgs {
  int dummy;
};

struct CtranKernelRecvArgs {
  int dummy;
};

struct CtranKernelSendRecvArgs {
  int dummy;
};

struct CtranKernelAllToAllArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
};

struct CtranKernelAllToAllvArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t selfCount;
  size_t selfSendDispl;
  size_t selfRecvDispl;
  KernelP2pElem* sendElemsList;
  KernelP2pElem* recvElemsList;
};

struct CtranKernelArgs {
  CtranAlgoDeviceState* devState_d{nullptr};
  union {
    CtranKernelAllGatherArgs allgather;
    CtranKernelSendArgs send;
    CtranKernelRecvArgs recv;
    CtranKernelSendRecvArgs sendrecv;
    CtranKernelAllToAllArgs alltoall;
    CtranKernelAllToAllvArgs alltoallv;
  } collective;
};

#endif
