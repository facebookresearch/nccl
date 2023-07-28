/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "nccl.h"

#define NCCL_ALLREDUCE_ALGO__ORIG       (0)
#define NCCL_ALLREDUCE_ALGO__THREADED   (1)

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  char *allreduceAlgoStr = getenv("NCCL_ALLREDUCE_ALGO");
  int allreduceAlgo = (allreduceAlgoStr != nullptr && !strcmp(allreduceAlgoStr, "threaded")) ?
      NCCL_ALLREDUCE_ALGO__THREADED : NCCL_ALLREDUCE_ALGO__ORIG;
  ncclResult_t res;

  if (allreduceAlgo == NCCL_ALLREDUCE_ALGO__THREADED) {
    res = ncclAllReduceThreaded(sendbuff, recvbuff, count, datatype, op, comm, stream);
    if (res != ncclNumResults) {
      return res;
    }
  }

  struct NvtxParamsAllReduce {
    size_t bytes;
    ncclRedOp_t op;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsAllReduce, op)}
  };
  NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
