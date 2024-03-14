// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AlgoAllReduceDdaNvsTreeIpc.h"

#include "AlgoUtils.h"
#include "all_reduce_dda.h"
#include "comm.h"
#include "debug.h"

namespace nccl {
namespace algorithms {

AlgoAllReduceDdaNvsTreeIpc::AlgoAllReduceDdaNvsTreeIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream,
    const DdaDeviceState* devStates,
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
      devStates_(devStates),
      devStates_d_(devStates_d),
      ipcBarrierFlag_(barrierFlag),
      maxBlocks_(maxBlocks) {}

AlgoAllReduceDdaNvsTreeIpc::~AlgoAllReduceDdaNvsTreeIpc() {}

template <typename T>
ncclResult_t AlgoAllReduceDdaNvsTreeIpc::launchKernel() {
  const void* func = nullptr;
  ASSIGN_FUNC_NRANKS(func, ncclKernel_AllReduce_DDA_Tree_ipc, comm_->nRanks);

  auto gridBlock =
      getGridAndBlockDims(func, count_, datatype_, maxBlocks_);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  void* args[] = {
      &ipcBarrierFlag_,
      &devStates_d_,
      &comm_->rank,
      &recvbuff_,
      &count_};
  CUDACHECK(cudaLaunchKernel(func, grid, block, args, 0, stream_));
  return ncclSuccess;
}

ncclResult_t AlgoAllReduceDdaNvsTreeIpc::allReduce() {
  INFO(NCCL_COLL, "AlgoAllReduceDdaNvsTreeIpc::allReduce");
  // copy src to tmp buffers
  CUDACHECKIGNORE(cudaMemcpyAsync(
        devStates_[comm_->rank].tmpbuff,
        sendbuff_,
        count_ * ncclTypeSize(datatype_),
        cudaMemcpyDefault,
        stream_));
  NCCLCHECK(NCCL_TYPED_CALL(datatype_, launchKernel));
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
