// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoDirector.h"

#include "AlgoUtils.h"
#include "DdaThreadedData.h"
#include "argcheck.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "nccl_cvars.h"

#include <cassert>

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_DDA_TMPBUFF_SIZE
   type        : uint64_t
   default     : 33554432
   description : |-
     DDA temporary buffer size.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

namespace nccl {
namespace algorithms {

/**
 * per communicator per rank Algorithm Manager that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
AlgoManagerBase::AlgoManagerBase(ncclComm_t comm) : comm_(comm), memHandler_(comm) {
  // get device property (expensive call: 10+ ms)
  CUDACHECKIGNORE(cudaGetDeviceProperties(&devProp_, comm_->cudaDev));
  maxBlocks_ =
      std::min(NCCL_DDA_ALLREDUCE_MAX_BLOCKS, devProp_.multiProcessorCount);

  // allocate host memory
  devStates_ = static_cast<DdaDeviceState*>(
      malloc(sizeof(DdaDeviceState) * comm_->nRanks));

  // allocate device memory
  // we need 3 barriers for Tree Algo
  // 1) [r0, r1, ..., rN-1]
  // 2) [r0, r1, ..., rN-1]@block0, [r0, r1, ..., rN-1]@block1, ...@blockK-1
  // 3) [r0, r1, ..., rN-1]@block0, [r0, r1, ..., rN-1]@block1, ...@blockK-1
  TRACE(NCCL_ALLOC, "Cuda Alloc Size %ld pointer (cudaMalloc)", (1 + 2 * maxBlocks_) * comm_->nRanks * sizeof(uintptr_t));
  CUDACHECKIGNORE(cudaMalloc(
      &threadedBarrierMbox_d_,
      ((1 + 2 * maxBlocks_) * comm_->nRanks) * sizeof(uintptr_t)));
  CUDACHECKIGNORE(cudaMemset(
      threadedBarrierMbox_d_,
      0,
      ((1 + 2 * maxBlocks_) * comm_->nRanks) * sizeof(uintptr_t)));

  // For IPC, we can reuse the same mbox across barrier operations by
  // incrementing the barrierFlag.
  TRACE(NCCL_ALLOC, "Cuda Alloc Size %ld pointer (cudaMalloc)", maxBlocks_ * comm_->nRanks * sizeof(uintptr_t));
  CUDACHECKIGNORE(cudaMalloc(
      &ipcBarrierMbox_d_,
      maxBlocks_ * comm_->nRanks * sizeof(uintptr_t)));
  CUDACHECKIGNORE(cudaMemset(
      ipcBarrierMbox_d_,
      0,
      maxBlocks_ * comm_->nRanks * sizeof(uintptr_t)));

  TRACE(NCCL_ALLOC, "Cuda Alloc Size %ld pointer (cudaMalloc)", NCCL_DDA_TMPBUFF_SIZE);
  CUDACHECKIGNORE(cudaMalloc(&tmpbuff_d_, NCCL_DDA_TMPBUFF_SIZE));
  TRACE(NCCL_ALLOC, "Cuda Alloc Size %ld pointer (cudaMalloc)", sizeof(DdaDeviceState) * comm_->nRanks);
  CUDACHECKIGNORE(
      cudaMalloc(&devStates_d_, sizeof(DdaDeviceState) * comm_->nRanks));

  // exchange handles
  memHandler_.add(threadedBarrierMbox_d_);
  memHandler_.add(ipcBarrierMbox_d_);
  memHandler_.add(tmpbuff_d_);
  memHandler_.exchangeMemHandles();
  for (int rank = 0; rank < comm_->nRanks; ++rank) {
    devStates_[rank].threadedBarrierMbox =
        static_cast<uintptr_t*>(memHandler_.get(rank, 0));
    devStates_[rank].ipcBarrierMbox =
        static_cast<uintptr_t*>(memHandler_.get(rank, 1));
    devStates_[rank].tmpbuff = memHandler_.get(rank, 2);
  }

  CUDACHECKIGNORE(cudaMemcpy(
      devStates_d_,
      devStates_,
      sizeof(DdaDeviceState) * comm_->nRanks,
      cudaMemcpyDefault));

  INFO(NCCL_INIT, "AlgoManagerBase initialized.");
}

AlgoManagerBase::~AlgoManagerBase() {
  // free device memory
  CUDACHECKIGNORE(cudaFree(threadedBarrierMbox_d_));
  CUDACHECKIGNORE(cudaFree(ipcBarrierMbox_d_));
  CUDACHECKIGNORE(cudaFree(tmpbuff_d_));
  CUDACHECKIGNORE(cudaFree(devStates_d_));

  // free host memory
  free(devStates_);

  INFO(NCCL_INIT, "AlgoManagerBase destroyed.");
}

bool AlgoManagerBase::checkNumRanks(size_t numRanks) {
  if (numRanks & (numRanks - 1)) {
    // power of two ranks
    return false;
  }

  if (numRanks == 1) {
    // more than 1 rank
    return false;
  }

  if (numRanks > 8) {
    // only support 2, 4, 8 ranks (single-node)
    return false;
  }

  return true;
}

} // namespace algorithms
} // namespace nccl
