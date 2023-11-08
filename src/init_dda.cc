// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <assert.h>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "graph/topo.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM
   type        : bool
   default     : false
   description : |-
     Enable DDA Allreduce for large messages on HCM platforms.

 - name        : NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE
   type        : int
   default     : 33554432
   description : |-
     DDA Allreduce temporary buffer size.

 - name        : NCCL_DDA_MAX_RANKS
   type        : int
   default     : 16
   description : |-
     Message size at which DDA Allreduce switches to the tree algorithm.
     Only applies to HCM-based systems.

 - name        : NCCL_ALLREDUCE_ALGO
   type        : enum
   default     : orig
   choices     : orig, dda
   description : |-
     The algorithm to use for Allreduce communication
     orig - Copy-based algorithm
     dda - Direct Data Access algorithms

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

static std::vector<ddaThreadSharedMd*> ddaThreadSharedMdList;
static std::mutex ddaThreadSharedMdListMutex;

bool operator==(const ncclUniqueId& lhs, const ncclUniqueId& rhs) {
  for (int i = 0; i < sizeof(ncclUniqueId); i++) {
    if (lhs.internal[i] != rhs.internal[i]) {
      return false;
    }
  }

  return true;
}

bool operator==(const ddaThreadSharedMd& lhs, const ddaThreadSharedMd& rhs) {
  return (lhs.commHash == rhs.commHash);
}

ncclDDAAllReduceAlgo_t getAllReduceAlgo(const void* sendbuff, void* recvbuff,
                                        size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                        ncclComm* comm) {
  const auto bytes = count * typeSize(datatype);
  int numDDAThreads = 0;

  if (NCCL_ALLREDUCE_ALGO == NCCL_ALLREDUCE_ALGO::orig) {
    goto algo_default;
  }

  /* allocate dda metadata if not initialized yet, fallback if failed to initialize */
  if (!comm->dda && allocDDAMd(comm) != ncclSuccess) {
    goto algo_default;
  }

  /* first try to see if the threaded DDA algo would work */
  numDDAThreads = comm->dda->threadSharedMd->registeredRanks.size();

  if ((numDDAThreads != comm->nRanks) || /* collective must only contain dda ranks */
      (numDDAThreads & (numDDAThreads - 1)) || /* power of two ranks */
      (numDDAThreads == 1) || /* more than one rank */
      (numDDAThreads > NCCL_DDA_MAX_RANKS) || /* only small rank counts are supported */
      (op != ncclSum) || /* only sum is supported */
      ((uintptr_t)sendbuff % 16) || /* 16-byte alignment */
      ((uintptr_t)recvbuff % 16)) { /* 16-byte alignment */
      goto algo_ipc;
  }

  if (comm->dda->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
    if (bytes < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS) {
      if ((bytes % 16) || /* allow for 16-byte loads */
          (sendbuff == recvbuff)) { /* in-place reduction */
        goto algo_ipc;
      }
    } else { /* bytes >= NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS */
      if (bytes % (16 * comm->nRanks)) { /* allow for 16-byte loads */
        goto algo_ipc;
      }
    }
  } else { /* topoType == NCCL_DDA_TOPO_TYPE__HCM */
    if (bytes < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM) {
      if (bytes % 16) { /* allow for 16-byte loads */
        goto algo_ipc;
      }
      if (bytes > NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE) { /* need tmpbuff */
        goto algo_ipc;
      }
    } else if (NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM) {
      if (bytes % (16 * comm->nRanks)) { /* allow for 16-byte loads */
        goto algo_ipc;
      }
      if (bytes > comm->nRanks * NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE) { /* need tmpbuff */
        goto algo_ipc;
      }
    } else {
      goto algo_ipc;
    }
  }
  return NCCL_DDA_ALLREDUCE_ALGO_DDA_THREADED;

algo_ipc:
  if ((comm->nRanks != comm->localRanks) || /* all ranks must be local */
      (comm->nRanks & (comm->nRanks - 1)) || /* power of two ranks */
      (comm->nRanks == 1) || /* more than one rank */
      (comm->nRanks > NCCL_DDA_MAX_RANKS) || /* only small rank counts are supported */
      (op != ncclSum)) { /* only sum is supported */
    goto algo_default;
  }

  if (comm->dda->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
    if (bytes < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS) {
      if (bytes % 16) { /* allow for 16-byte loads */
        goto algo_default;
      }
    } else { /* bytes >= NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS */
      if (bytes % (16 * comm->nRanks)) { /* allow for 16-byte loads */
        goto algo_default;
      }
    }

    if (bytes > NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE) { /* need tmpbuff for IPC */
      goto algo_default;
    }
  } else { /* topoType == NCCL_DDA_TOPO_TYPE__HCM */
    if (bytes < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM) {
      if (bytes % 16) { /* allow for 16-byte loads */
        goto algo_default;
      }
      if (bytes > NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE / 2) { /* need tmpbuff */
        goto algo_default;
      }
    } else {
      goto algo_default;
    }
  }
  return NCCL_DDA_ALLREDUCE_ALGO_DDA_IPC;

algo_default:
  return NCCL_DDA_ALLREDUCE_ALGO_DEFAULT;
}

ncclResult_t allocDDAMd(ncclComm *comm) {
  ddaThreadSharedMd* threadSharedMd;
  ncclResult_t ret = ncclSuccess;

  ddaThreadSharedMdListMutex.lock();

  /* allocate the ddaThreadSharedMd structure or find an existing
   * one for this commHash */
  threadSharedMd = nullptr;
  for (auto t : ddaThreadSharedMdList) {
    if (t->commHash == comm->commHash) {
      threadSharedMd = t;
      break;
    }
  }
  if (threadSharedMd == nullptr) {
    threadSharedMd = new ddaThreadSharedMd(comm->commHash);
    ddaThreadSharedMdList.push_back(threadSharedMd);
  }

  threadSharedMd->insertRank(comm->rank);

  ddaThreadSharedMdListMutex.unlock();

  comm->dda = new ddaPrivateMd(threadSharedMd, comm);
  INFO(NCCL_INIT, "Initialized DDA for commHash %lu", comm->commHash);

  return ret;
}

ncclResult_t freeDDAMd(ncclComm *comm) {
  ddaThreadSharedMd *threadSharedMd = comm->dda->threadSharedMd;

  ddaThreadSharedMdListMutex.lock();

  threadSharedMd->deleteRank(comm->rank);

  if (threadSharedMd->registeredRanks.empty()) {
    auto threadSharedMdIdx =
        std::remove(ddaThreadSharedMdList.begin(), ddaThreadSharedMdList.end(), threadSharedMd);
    ddaThreadSharedMdList.erase(threadSharedMdIdx, ddaThreadSharedMdList.end());
    delete threadSharedMd;
  }

  ddaThreadSharedMdListMutex.unlock();

  delete comm->dda;

  return ncclSuccess;
}
