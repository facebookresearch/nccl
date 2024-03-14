// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AlgoAllReduceDdaNvsFlatThreaded.h"

#include "AlgoUtils.h"
#include "all_reduce_dda.h"
#include "comm.h"
#include "debug.h"

namespace nccl {
namespace algorithms {

AlgoAllReduceDdaNvsFlatThreaded::AlgoAllReduceDdaNvsFlatThreaded(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream,
    const DdaDeviceState* devStates_d,
    uintptr_t barrierFlag,
    size_t maxBlocks)
    : sendbuff_(sendbuff),
      recvbuff_(recvbuff),
      count_(count),
      datatype_(datatype),
      op_(op),
      comm_(comm),
      stream_(stream),
      devStates_d_(devStates_d),
      threadedBarrierFlag_(barrierFlag),
      maxBlocks_(maxBlocks) {}

AlgoAllReduceDdaNvsFlatThreaded::~AlgoAllReduceDdaNvsFlatThreaded() {}

template <typename T>
ncclResult_t AlgoAllReduceDdaNvsFlatThreaded::launchKernel() {
  const void* func = nullptr;
  ASSIGN_FUNC_NRANKS(func, ncclKernel_AllReduce_DDA_Flat, comm_->nRanks);

  auto gridBlock =
      getGridAndBlockDims(func, count_, datatype_, maxBlocks_);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  void* args[] = {
      &threadedBarrierFlag_,
      &devStates_d_,
      &comm_->rank,
      &sendbuff_,
      &recvbuff_,
      &count_};
  CUDACHECK(cudaLaunchKernel(func, grid, block, args, 0, stream_));
  return ncclSuccess;
}

ncclResult_t AlgoAllReduceDdaNvsFlatThreaded::allReduce() {
  INFO(NCCL_COLL, "AlgoAllReduceDdaNvsFlatThreaded::allReduce");
  NCCLCHECK(NCCL_TYPED_CALL(datatype_, launchKernel));
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
