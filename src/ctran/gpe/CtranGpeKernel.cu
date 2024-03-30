// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranGpeKernel.h"
#include "CtranAlgoDev.h"
#include "CtranGpeDev.h"

__global__ void ncclKernelAllGatherCtranDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args) {
  if (flag) {
    ncclKernelStartGpe(flag);
    ncclKernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelAllGatherCtranRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args) {
  if (flag) {
    ncclKernelStartGpe(flag);
    ncclKernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelAllGatherCtranRecDbl(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args) {
  if (flag) {
    ncclKernelStartGpe(flag);
    ncclKernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelSend(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelSendArgs args) {
  if (flag) {
    ncclKernelStartGpe(flag);
    ncclKernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelRecvArgs args) {
  if (flag) {
    ncclKernelStartGpe(flag);
    ncclKernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelSendRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelSendRecvArgs args) {
  if (flag) {
    ncclKernelStartGpe(flag);
    ncclKernelWaitGpeTerminate(flag);
  }
}
