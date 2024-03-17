// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranAlgo.h"
#include <nccl_common.h>
#include <string.h>
#include <memory>
#include <stdexcept>
#include "CtranChecks.h"
#include "DdaThreadedData.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===
 - name        : NCCL_CTRAN_SHARED_DEVBUF_SIZE
   type        : uint64_t
   default     : 8388608
   description : |-
     Size of shared device memory region allocated for each peer for inter-GPU
     communication. In total NCCL_CTRAN_SHARED_DEVBUF_SIZE * number of
     local ranks size of memory will be allocated on each rank.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

CtranAlgo::CtranAlgo(ncclComm* comm) {
  this->comm_ = comm;

  // Initialize inter-process shared device buffer only for multi-process comm.
  // numDdaThreads > 1 means this comm is in multi-threaded mode.
  if (comm->localRanks > 1 && !this->isThreadedComm(comm)) {
    this->sharedRes_ = new SharedResource(comm);
  }

  // Initialize local device state after shared resource creation
  NCCLCHECKTHROW(this->initDevState());
  return;
}

CtranAlgo::~CtranAlgo() {
  if (this->sharedRes_) {
    // Release shared resource if releaseSharedResource is not yet called.
    delete this->sharedRes_;
    this->sharedRes_ = nullptr;
  }

  CUDACHECKIGNORE(cudaFree(this->devState_d));
  this->devState_d = nullptr;

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

bool CtranAlgo::isThreadedComm(ncclComm* comm) {
  const size_t numDdaThreads =
      nccl::algorithms::DdaThreadedData::get()->numRanks(comm->commHash);
  return numDdaThreads > 1;
}

void CtranAlgo::releaseSharedResource() {
  if (this->sharedRes_) {
    delete this->sharedRes_;
    this->sharedRes_ = nullptr;
  }
}

ncclResult_t CtranAlgo::initDevState() {
  CtranAlgoDeviceState tmpDevState;
  memset(&tmpDevState, 0, sizeof(CtranAlgoDeviceState));

  // Copy basic comm info to device state for collective kernel to use
  tmpDevState.localRank = this->comm_->localRank;
  tmpDevState.localRanks = this->comm_->localRanks;
  tmpDevState.bufSize = NCCL_CTRAN_SHARED_DEVBUF_SIZE;
  for (int localRank = 0; localRank < this->comm_->localRanks; localRank++) {
    tmpDevState.localRankToRank[localRank] =
        this->comm_->localRankToRank[localRank];
  }

  // Setup pointers to bufstates and shared buffer of each peers' shared region
  // See description of bufState and buf memory locations in SharedResource.
  if (this->sharedRes_) {
    auto& allPeerToBufStatesMap = tmpDevState.allPeerToBufStatesMap;
    auto& allPeerToBufsMap = tmpDevState.allPeerToBufsMap;

    for (int owner = 0; owner < this->comm_->localRanks; owner++) {
      char* regionPtr_d = (char*)this->sharedRes_->mappedDevShmPtrs[owner];
      void* bufBase_d = (char*)this->sharedRes_->mappedDevShmPtrs[owner] +
          (this->comm_->localRanks - 1) * sizeof(CtranAlgoDeviceBufState);
      for (int i = 0; i < this->comm_->localRanks; i++) {
        // Skip owner itself
        if (i == owner) {
          allPeerToBufStatesMap[owner][i] = nullptr;
          allPeerToBufsMap[owner][i] = nullptr;
          continue;
        }

        int pos = LOCAL_RANK_TO_DEV_REGION_POS(i, owner);
        void* statePtr_d =
            (char*)regionPtr_d + pos * sizeof(CtranAlgoDeviceBufState);

        allPeerToBufStatesMap[owner][i] =
            reinterpret_cast<CtranAlgoDeviceBufState*>(statePtr_d);
        allPeerToBufsMap[owner][i] =
            (char*)bufBase_d + pos * NCCL_CTRAN_SHARED_DEVBUF_SIZE;
      }
    }
  }

  // Copy contents to device
  CUDACHECK(cudaMalloc(&this->devState_d, sizeof(CtranAlgoDeviceState)));
  CUDACHECK(cudaMemcpy(
      this->devState_d,
      &tmpDevState,
      sizeof(CtranAlgoDeviceState),
      cudaMemcpyHostToDevice));

  return ncclSuccess;
}

CtranAlgo::SharedResource::SharedResource(ncclComm* comm) {
  this->comm_ = comm;

  // Create local shared memory region
  // The memory region on each owner rank is divided to (localRanks -1) sets of
  // bufState and buf for each peer, excluding the owner. The format is as
  // below with N localRanks.
  // |bufState_0|bufState_1|...|bufState_N-2|buf_0|buf_1|...|buf_N-2|
  std::vector<cudaIpcMemHandle_t> handles(this->comm_->localRanks);
  size_t shmSize =
      (sizeof(CtranAlgoDeviceBufState) + NCCL_CTRAN_SHARED_DEVBUF_SIZE) *
      (this->comm_->localRanks - 1);

  CUDACHECKTHROW(cudaMalloc(&this->devShmPtr_, shmSize));
  CUDACHECKTHROW(
      cudaIpcGetMemHandle(&handles[this->comm_->localRank], this->devShmPtr_));

  // Initialize device state for each peer
  for (int i = 0; i < this->comm_->localRanks; i++) {
    // Skip owner itself
    if (i == this->comm_->localRank) {
      continue;
    }

    int pos = LOCAL_RANK_TO_DEV_REGION_POS(i, this->comm_->localRank);
    void* statePtr_d =
        (char*)this->devShmPtr_ + pos * sizeof(CtranAlgoDeviceBufState);
    struct CtranAlgoDeviceBufState stateInitialVal;
    for (int j = 0; j < CTRAN_ALGO_MAX_THREAD_BLOCKS; j++) {
      stateInitialVal.stepOnSameBlockIdx[j] = CTRAN_ALGO_STEP_RESET;
    }
    CUDACHECKTHROW(cudaMemcpy(
        statePtr_d,
        &stateInitialVal,
        sizeof(CtranAlgoDeviceBufState),
        cudaMemcpyHostToDevice));
  }

  // Exchange IPC handle with all local ranks
  NCCLCHECKTHROW(bootstrapIntraNodeAllGather(
      this->comm_->bootstrap,
      this->comm_->localRankToRank,
      this->comm_->localRank,
      this->comm_->localRanks,
      handles.data(),
      sizeof(cudaIpcMemHandle_t)));

  // Setup mapped shared memory region pointers for all local ranks
  this->mappedDevShmPtrs.resize(this->comm_->localRanks, nullptr);
  for (int i = 0; i < this->comm_->localRanks; ++i) {
    if (this->comm_->localRank == i) {
      this->mappedDevShmPtrs[i] = this->devShmPtr_;
    } else {
      void* mappedDevPtr = nullptr;
      CUDACHECKTHROW(cudaIpcOpenMemHandle(
          &mappedDevPtr, handles[i], cudaIpcMemLazyEnablePeerAccess));
      this->mappedDevShmPtrs[i] = mappedDevPtr;
    }
  }

  INFO(
      NCCL_INIT | NCCL_ALLOC,
      "CTRAN-ALGO: allocated %ld bytes of device buffer as shared resource on rank %d localRank %d",
      shmSize,
      this->comm_->rank,
      this->comm_->localRank);
  return;
}

CtranAlgo::SharedResource::~SharedResource() {
  for (int i = 0; i < this->comm_->localRanks; ++i) {
    if (this->mappedDevShmPtrs[i] && i != this->comm_->localRank) {
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(this->mappedDevShmPtrs[i]));
    }
  }

  // Ensure all local ranks closed remote IPC handle before free
  NCCLCHECKIGNORE(this->barrier());

  if (this->devShmPtr_) {
    CUDACHECKIGNORE(cudaFree(this->devShmPtr_));
  }
  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

ncclResult_t CtranAlgo::SharedResource::barrier() {
  NCCLCHECK(bootstrapBarrier(
      this->comm_->bootstrap,
      this->comm_->localRankToRank,
      this->comm_->localRank,
      this->comm_->localRanks,
      this->comm_->localRankToRank[0]));
  return ncclSuccess;
}
