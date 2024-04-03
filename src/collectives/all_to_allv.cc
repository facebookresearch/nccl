#include "Ctran.h"
#include "argcheck.h"
#include "comm.h"
#include "nccl.h"

inline ncclResult_t checkBuffCountsAndPointers(const void* buff, const size_t* counts, const char* buffName, ncclComm_t comm, const char* funcName) {
  if(counts == NULL) {
    WARN("Counts pointer is NULL.");
    return ncclInvalidArgument;
  }

  if(buff == NULL) {
    for (int i = 0; i < comm->nRanks; i++) {
      if(counts[i] != 0) {
        WARN("Found NULL %s with nonzero counts.", buffName);
        return ncclInvalidArgument;
      }
    }
  }
  else {
    NCCLCHECK(CudaPtrCheck(buff, comm, buffName, funcName));
  }
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
  NCCLCHECK(checkBuffCountsAndPointers(sendbuff, sendcounts, "sendbuff", comm, "ncclAllToAllv"));
  NCCLCHECK(checkBuffCountsAndPointers(recvbuff, recvcounts, "recvbuff", comm, "ncclAllToAllv"));
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
