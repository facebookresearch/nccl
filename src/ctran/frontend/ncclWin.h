// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_WIN_H_
#define CTRAN_WIN_H_

#include <cstdint>
#include <vector>
#include "nccl.h"
#include "cuda_runtime.h"

// constexpr int kNCCLMaxDevices = 16;

struct ncclWin {
    ncclComm_t *comm;
    void** remotePtrs;
    cudaIpcMemHandle_t* ipcHandles;
};

#endif // CTRAN_WIN_H_
