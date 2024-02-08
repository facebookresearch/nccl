// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_COMM_H_
#define CTRAN_COMM_H_

#include <memory>
#include "CtranAlgo.h"
#include "CtranGpe.h"
#include "CtranMapper.h"
#include "nccl.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_SENDRECV_ALGO
   type        : enum
   default     : orig
   choices     : orig, ctran
   description : |-
     The algorithm to use for sendrecv communication
     orig - Copy-based communication
     ctran - Ctran-based communication

 - name        : NCCL_ALLGATHER_ALGO
   type        : enum
   default     : orig
   choices     : orig, ctdirect, ctring, ctrd
   description : |-
     The algorithm to use for Allgather communication
     orig - Copy-based ring algorithm
     ctdirect - Ctran-based direct point-to-point algorithm
     ctring - Ctran-based ring algorithm
     ctrd - Ctran-based recursive doubling algorithm

 - name        : NCCL_ALLTOALL_ALGO
   type        : enum
   default     : orig
   choices     : orig, ctran
   description : |-
     The algorithm to use for alltoall communication
     orig - Copy-based communication
     ctran - Ctran-based communication

 - name        : NCCL_ALLTOALLV_ALGO
   type        : enum
   default     : orig
   choices     : orig, ctran
   description : |-
     The algorithm to use for alltoallv communication
     orig - Copy-based communication
     ctran - Ctran-based communication

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

#define CTRAN_COLL_INFO(                                                                                                                     \
    algoStr, sendbuff, recvbuff, count, datatype, peer, comm, stream)                                                                        \
  do {                                                                                                                                       \
    INFO(                                                                                                                                    \
        NCCL_COLL,                                                                                                                           \
        "%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d peer %d comm %p commHash %lu [nranks=%d, localRanks=%d] stream=%p\n", \
        algoStr,                                                                                                                             \
        comm->opCount,                                                                                                                       \
        sendbuff,                                                                                                                            \
        recvbuff,                                                                                                                            \
        count,                                                                                                                               \
        datatype,                                                                                                                            \
        peer,                                                                                                                                \
        comm,                                                                                                                                \
        comm->commHash,                                                                                                                      \
        comm->nRanks,                                                                                                                        \
        comm->localRanks,                                                                                                                    \
        stream);                                                                                                                             \
    comm->opCount++;                                                                                                                         \
  } while (0)

class Ctran {
 public:
  Ctran(ncclComm* comm);
  ~Ctran() = default;

  ncclResult_t commRegister(void* buff, size_t size, void** handle);
  ncclResult_t commDeregister(void* handle);

  std::unique_ptr<CtranMapper> mapper{nullptr};
  std::unique_ptr<CtranGpe> gpe{nullptr};
  std::unique_ptr<CtranAlgo> algo{nullptr};
};

inline bool ctranIsUsed() {
  return (NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctran);
}

ncclResult_t ctranInit(ncclComm* comm);
bool ctranInitialized(ncclComm* comm);
ncclResult_t ctranDestroy(ncclComm* comm);

// Called by ncclCommAbort to release any shared resources before bootstrap
// close. For local resources such as thread and backends, they can be
// released in ctranDestroy.
ncclResult_t CtranAbort(ncclComm* comm);

bool ctranSendRecvSupport(int peer, ncclComm_t comm);
ncclResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t ctranGroupEndHook(void);

ncclResult_t ctranAllGatherDirect(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t ctranAllGatherRd(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t ctranAllGatherRing(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);

bool ctranAllToAllSupport(
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm);

ncclResult_t ctranAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t ctranAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);

#endif // CTRAN_COMM_H_
