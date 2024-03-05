// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_WIN_H_
#define NCCL_WIN_H_

#include <cstdint>
#include <vector>
#include "nccl.h"
#include "cuda_runtime.h"

struct ncclWin {
    // communicator associated with this window
    ncclComm_t comm;
    // accessible remote pointers from peer participated in this window
    void** remotePtrs;
    // cached cuda IPC handles from each peer
    cudaIpcMemHandle_t* ipcHandles;
    // cached local handle of temporarya llocated buffer, used to de-register the buffer during ncclWinFree
    void* localHdl;
};

#endif // NCCL_WIN_H_
