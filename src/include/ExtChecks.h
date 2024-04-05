// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "debug.h"

#ifndef CTRAN_CHECKS_H_
#define CTRAN_CHECKS_H_

#define CUDACHECKTHROW(cmd)                                        \
  do {                                                              \
    cudaError_t err = cmd;                                          \
    if (err != cudaSuccess) {                                       \
      WARN(                                                         \
          "%s:%d Cuda failure '%s'",                                \
          __FILE__,                                                 \
          __LINE__,                                                 \
          cudaGetErrorString(err));                                 \
      (void)cudaGetLastError();                                     \
      throw std::runtime_error(                                     \
          std::string("Cuda failure: ") + cudaGetErrorString(err)); \
    }                                                               \
  } while (false)

#define NCCLCHECKTHROW(cmd)                                             \
  do {                                                                   \
    ncclResult_t RES = cmd;                                              \
    if (RES != ncclSuccess && RES != ncclInProgress) {                   \
      WARN("%s:%d -> %d", __FILE__, __LINE__, RES);                      \
      throw std::runtime_error(                                          \
          std::string("NCCL internal failure: ") + std::to_string(RES)); \
    }                                                                    \
  } while (0)

#endif
