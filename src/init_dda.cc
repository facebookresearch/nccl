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

NCCL_PARAM(DDAAllreduceLargeMessageHCM, "DDA_ALLREDUCE_LARGE_MESSAGE_HCM", 0);
NCCL_PARAM(DDAAllreduceTmpbuffSize, "DDA_ALLREDUCE_TMPBUFF_SIZE", 32 * 1024 * 1024);
NCCL_PARAM(MaxDDARanks, "MAX_DDA_RANKS", 16);
NCCL_PARAM(ForceP2pAccess, "FORCE_P2P_ACCESS", 0);

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
  const char* allreduceAlgoStr = getenv("NCCL_ALLREDUCE_ALGO");

  if (allreduceAlgoStr == nullptr || strcmp(allreduceAlgoStr, "dda")) {
    goto algo_default;
  }

  /* first try to see if the threaded DDA algo would work */
  numDDAThreads = comm->dda->threadSharedMd->registeredRanks.size();

  if ((numDDAThreads != comm->nRanks) || /* collective must only contain dda ranks */
      (numDDAThreads & (numDDAThreads - 1)) || /* power of two ranks */
      (numDDAThreads == 1) || /* more than one rank */
      (numDDAThreads > ncclParamMaxDDARanks()) || /* only small rank counts are supported */
      (op != ncclSum) || /* only sum is supported */
      ((uintptr_t)sendbuff % 16) || /* 16-byte alignment */
      ((uintptr_t)recvbuff % 16)) { /* 16-byte alignment */
      goto algo_ipc;
  }

  if (comm->dda->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
    if (bytes < ncclParamDDAAllreduceTreeThresholdNVS()) {
      if ((bytes % 16) || /* allow for 16-byte loads */
          (sendbuff == recvbuff)) { /* in-place reduction */
        goto algo_ipc;
      }
    } else { /* bytes >= ncclParamDDAAllreduceTreeThresholdNVS() */
      if (bytes % (16 * comm->nRanks)) { /* allow for 16-byte loads */
        goto algo_ipc;
      }
    }
  } else { /* topoType == NCCL_DDA_TOPO_TYPE__HCM */
    if (bytes < ncclParamDDAAllreduceTreeThresholdHCM()) {
      if (bytes % 16) { /* allow for 16-byte loads */
        goto algo_ipc;
      }
      if (bytes > ncclParamDDAAllreduceTmpbuffSize()) { /* need tmpbuff */
        goto algo_ipc;
      }
    } else if (ncclParamDDAAllreduceLargeMessageHCM()) {
      if (bytes % (16 * comm->nRanks)) { /* allow for 16-byte loads */
        goto algo_ipc;
      }
      if (bytes > comm->nRanks * ncclParamDDAAllreduceTmpbuffSize()) { /* need tmpbuff */
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
      (comm->nRanks > ncclParamMaxDDARanks()) || /* only small rank counts are supported */
      (op != ncclSum)) { /* only sum is supported */
    goto algo_default;
  }

  if (comm->dda->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
    if (bytes < ncclParamDDAAllreduceTreeThresholdNVS()) {
      if (bytes % 16) { /* allow for 16-byte loads */
        goto algo_default;
      }
    } else { /* bytes >= ncclParamDDAAllreduceTreeThresholdNVS() */
      if (bytes % (16 * comm->nRanks)) { /* allow for 16-byte loads */
        goto algo_default;
      }
    }

    if (bytes > ncclParamDDAAllreduceTmpbuffSize()) { /* need tmpbuff for IPC */
      goto algo_default;
    }
  } else { /* topoType == NCCL_DDA_TOPO_TYPE__HCM */
    if (bytes < ncclParamDDAAllreduceTreeThresholdHCM()) {
      if (bytes % 16) { /* allow for 16-byte loads */
        goto algo_default;
      }
      if (bytes > ncclParamDDAAllreduceTmpbuffSize() / 2) { /* need tmpbuff */
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
