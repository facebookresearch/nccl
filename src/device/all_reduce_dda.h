// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include "AlgoUtils.h"

// DDA kernels
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Flat(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, const T* sendbuff, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Tree(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, const T* sendbuff, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Flat_ipc(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Tree_ipc(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_ScatGat_ipc(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, T* sendbuff, T* recvbuff, size_t count);

#define DECL_DDA_FUNC_NRANKS(T, NRANKS)                                \
  template __global__ void ncclKernel_AllReduce_DDA_Flat<T, NRANKS>(   \
    uintptr_t barrierFlag,                                              \
    DdaDeviceState* devStates,                                          \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_Tree<T, NRANKS>(   \
    uintptr_t barrierFlag,                                              \
    DdaDeviceState* devStates,                                          \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_Flat_ipc<T, NRANKS>(   \
    uintptr_t barrierFlag,                                              \
    DdaDeviceState* devStates,                                          \
    int rank,                                                           \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_Tree_ipc<T, NRANKS>(   \
    uintptr_t barrierFlag,                                              \
    DdaDeviceState* devStates,                                          \
    int rank,                                                           \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_ScatGat_ipc<T, NRANKS>(   \
    uintptr_t barrierFlag,                                              \
    DdaDeviceState* devStates,                                          \
    int rank,                                                           \
    T* sendbuff,                                                        \
    T* recvbuff,                                                        \
    size_t count);

#define DECL_DDA_FUNC(T)      \
  DECL_DDA_FUNC_NRANKS(T, 2); \
  DECL_DDA_FUNC_NRANKS(T, 4); \
  DECL_DDA_FUNC_NRANKS(T, 8); \
  DECL_DDA_FUNC_NRANKS(T, 16)
