
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comm.h"
#include "nccl.h"
#include "argcheck.h"

NCCL_API(ncclResult_t, ncclCommGetUniqueHash, ncclComm_t comm, uint64_t* hash);
ncclResult_t ncclCommGetUniqueHash(ncclComm_t comm, uint64_t* hash) {
  NCCLCHECK(PtrCheck(comm, "CommGetUniqueHash", "comm"));

  *hash = comm->commHash;
  return ncclSuccess;
}
