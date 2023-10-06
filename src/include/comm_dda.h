// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_COMM_DDA_H_
#define NCCL_COMM_DDA_H_

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "checks.h"
#include "ddaThreadSharedMd.h"
#include "ddaMemHandles.h"
#include "ddaPrivateMd.h"

int64_t ncclParamMaxDDARanks(void);
int64_t ncclParamDDAAllreduceTmpbuffSize(void);
int64_t ncclParamDDAAllreduceTreeThresholdNVS(void);
int64_t ncclParamDDAAllreduceTreeThresholdHCM(void);
int64_t ncclParamForceP2pAccess(void);

typedef enum {
  NCCL_DDA_ALLREDUCE_ALGO_DEFAULT,
  NCCL_DDA_ALLREDUCE_ALGO_DDA_IPC,
  NCCL_DDA_ALLREDUCE_ALGO_DDA_THREADED,
} ncclDDAAllReduceAlgo_t;

ncclDDAAllReduceAlgo_t getAllReduceAlgo(const void* sendbuff, void* recvbuff,
                                        size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                        ncclComm* comm);
ncclResult_t allocDDAMd(ncclComm *comm);
ncclResult_t freeDDAMd(ncclComm *comm);

static inline int typeSize(ncclDataType_t datatype)
{
  switch (datatype) {
    case ncclInt8:
      return sizeof(int8_t);

    case ncclUint8:
      return sizeof(uint8_t);

    case ncclInt32:
      return sizeof(int32_t);

    case ncclUint32:
      return sizeof(uint32_t);

    case ncclInt64:
      return sizeof(int64_t);

    case ncclUint64:
      return sizeof(uint64_t);

    case ncclFloat16:
      return sizeof(half);

  case ncclFloat32:
      return sizeof(float);

    case ncclFloat64:
      return sizeof(double);

#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      return sizeof(__nv_bfloat16);
#endif

    default:
      return 0;
  }
}

#endif
