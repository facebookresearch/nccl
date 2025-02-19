/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"

template <typename T>
__global__ void ncclKernel_AllReduceSparseBlock_Unpack(
    T* unpackBuf,
    const T* packBuf,
    const size_t blockCount,
    const int64_t* unpackIndices,
    const size_t blockLength) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t packOffset = globalId; packOffset < blockCount * blockLength;
       packOffset += blockDim.x * gridDim.x) {
    size_t blockIdx = packOffset / blockLength;
    size_t unpackOffset = unpackIndices[blockIdx] + packOffset % blockLength;
    unpackBuf[unpackOffset] = packBuf[packOffset];
  }
}

#define DECL_UNPACK_KERN(T)                                           \
  template __global__ void ncclKernel_AllReduceSparseBlock_Unpack<T>( \
      T * unpackBuf,                                                  \
      const T* packBuf,                                               \
      const size_t blockCount,                                        \
      const int64_t* unpackIndices,                                   \
      const size_t blockLength);

DECL_UNPACK_KERN(int8_t);
DECL_UNPACK_KERN(uint8_t);
DECL_UNPACK_KERN(int32_t);
DECL_UNPACK_KERN(uint32_t);
DECL_UNPACK_KERN(int64_t);
DECL_UNPACK_KERN(uint64_t);
DECL_UNPACK_KERN(half);
DECL_UNPACK_KERN(float);
DECL_UNPACK_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_UNPACK_KERN(__nv_bfloat16);
#endif
