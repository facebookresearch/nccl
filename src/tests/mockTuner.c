// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// This file is a mock implementation of the tuning plugin interface. It's used
// to test the tuning plugin framework without having to build the real tuners.

#include <stdlib.h>
#include "nccl_tuner.h"

__attribute__((visibility("hidden"))) static ncclResult_t ncclTuningMockInit(
    size_t nRanks,
    size_t nNodes,
    ncclDebugLogger_t logFunction) {
  // set a dummy NCCL environment variable
  setenv("NCCL_NET_PLUGIN", "mock", 1);
  return ncclSuccess;
}
__attribute__((visibility("hidden"))) static ncclResult_t
ncclTuningMockGetCollInfo(
    ncclFunc_t collType,
    size_t nBytes,
    int collNetSupport,
    int nvlsSupport,
    int numPipeOps,
    int* algorithm,
    int* protocol,
    int* nChannels) {
  // Randomly picked values for testing
  if (collType == ncclFuncAllReduce && nBytes < 1024) {
    *algorithm = NCCL_ALGO_TREE;
    *protocol = NCCL_PROTO_LL;
    *nChannels = 1;
  } else {
    *algorithm = NCCL_ALGO_RING;
    *protocol = NCCL_PROTO_SIMPLE;
    *nChannels = 8;
  }
  return ncclSuccess;
}
__attribute__((visibility("hidden"))) static ncclResult_t ncclTuningMockDestory(
    void) {
  return ncclSuccess;
}

const ncclTuner_v1_t ncclTunerPlugin_v1 = {
    .name = "mockTuner",
    .init = ncclTuningMockInit,
    .getCollInfo = ncclTuningMockGetCollInfo,
    .destroy = ncclTuningMockDestory};
