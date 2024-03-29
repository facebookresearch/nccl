/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "nccl.h"
#include "nccl_cvars.h"
#include "Ctran.h"

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  int nRanks = comm->nRanks;
  size_t rankOffset = sendcount * ncclTypeSize(datatype);
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  // CTRAN allgather: only support inter-node now
  if (ctranInitialized(comm) && comm->localRanks == 1 && nRanks > 1 && rankOffset > getpagesize()) {
    if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctdirect) {
      return ctranAllGatherDirect(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctring) {
      return ctranAllGatherRing(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctrd) {
      return ctranAllGatherRd(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }
  }

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  if (comm->algoDirector && comm->algoDirector->allReduce) {
    // try to get meta customized algo
    auto algo = comm->algoDirector->allReduce->getAlgoAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    if (algo) {
      return algo->allReduce();
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

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsBroadcast {
    size_t bytes;
    int root;
  };
  constexpr nvtxPayloadSchemaEntry_t BroadcastSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsBroadcast, root)}
  };
  NvtxParamsBroadcast payload{count * ncclTypeSize(datatype), root};
  NVTX3_FUNC_WITH_PARAMS(Broadcast, BroadcastSchema, payload)

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsReduce {
    size_t bytes;
    int root;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsReduce, root)},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduce, op)}
  };
  NvtxParamsReduce payload{count * ncclTypeSize(datatype), root, op};
  NVTX3_FUNC_WITH_PARAMS(Reduce, ReduceSchema, payload)

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsReduceScatter {
    size_t bytes;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceScatterSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduceScatter, op)}
  };
  NvtxParamsReduceScatter payload{recvcount * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, ReduceScatterSchema, payload)

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  if (NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctran &&
      ctranSendRecvSupport(peer, comm)) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = ctranSend(sendbuff, count, datatype, peer, comm, stream);
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }

  ctranGroupTrackDefaultOp(comm);

  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  if (NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctran &&
      ctranSendRecvSupport(peer, comm)) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = ctranRecv(recvbuff, count, datatype, peer, comm, stream);
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }

  ctranGroupTrackDefaultOp(comm);

  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}


NCCL_API(
    ncclResult_t,
    ncclAllToAll,
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "ncclAllToAll"));
  NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "ncclAllToAll"));
  if (sendbuff == recvbuff) {
    WARN(
        "Found sendbuff %p == recvbuff %p. In-place ncclAllToAll is not supported.",
        sendbuff,
        recvbuff);
    return ncclInvalidArgument;
  }

  // Do nothing if count is 0
  if (count == 0) {
    return ncclSuccess;
  }

  if (ctranAllToAllSupport(count, datatype, comm) &&
      NCCL_ALLTOALL_ALGO == NCCL_ALLTOALL_ALGO::ctran) {
    return ctranAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
  }

  // fallback to default send/recv based alltoallv

  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++) {
    if (count) {
      NCCLCHECK(ncclSend(
          ((char*)sendbuff) + r * count * ncclTypeSize(datatype),
          count,
          datatype,
          r,
          comm,
          stream));
    }
    if (count) {
      NCCLCHECK(ncclRecv(
          ((char*)recvbuff) + r * count * ncclTypeSize(datatype),
          count,
          datatype,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}


NCCL_API(
    ncclResult_t,
    ncclAllToAllv,
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "ncclAllToAllv"));
  NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "ncclAllToAllv"));
  if (sendbuff == recvbuff) {
    WARN(
        "Found sendbuff %p == recvbuff %p. In-place ncclAllToAllv is not supported.",
        sendbuff,
        recvbuff);
    return ncclInvalidArgument;
  }

  if (ctranInitialized(comm) &&
      NCCL_ALLTOALLV_ALGO == NCCL_ALLTOALLV_ALGO::ctran) {
    return ctranAllToAllv(
        sendbuff,
        sendcounts,
        sdispls,
        recvbuff,
        recvcounts,
        rdispls,
        datatype,
        comm,
        stream);
  }

  // fallback to default send/recv based alltoallv
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++) {
    if (sendcounts[r]) {
      NCCLCHECK(ncclSend(
          ((char*)sendbuff) + sdispls[r] * ncclTypeSize(datatype),
          sendcounts[r],
          datatype,
          r,
          comm,
          stream));
    }
    if (recvcounts[r]) {
      NCCLCHECK(ncclRecv(
          ((char*)recvbuff) + rdispls[r] * ncclTypeSize(datatype),
          recvcounts[r],
          datatype,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}
