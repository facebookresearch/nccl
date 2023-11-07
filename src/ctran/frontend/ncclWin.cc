// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ncclWin.h"
#include "argcheck.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"

// FIXME: should we expose baseptr to user since users have to query anyway?
NCCL_API(
    ncclResult_t,
    ncclWinAllocShared,
    size_t size,
    ncclComm_t comm,
    ncclWin_t* win);
ncclResult_t ncclWinAllocShared(size_t size, ncclComm_t comm, ncclWin_t* win) {
  bool can_use_win =
      (comm->nNodes == 1 && comm->nRanks == comm->localRanks &&
       comm->ctranMapper);
  int nRanks = comm->nRanks;
  // FIXME: should be support sinlge process communicator?
  if (!can_use_win) {
    WARN(
        "ncclCommWinAllocShared only supports intra-node collectives comm->nNodes %d, comm->nRanks %d, comm->localRanks %d",
        comm->nNodes,
        comm->nRanks,
        comm->localRanks);
    return ncclSuccess;
  }
  // TODO: sanity check to make sure all peers can use IPC

  // allocate resources
  ncclWin* win_ = static_cast<ncclWin*>(malloc(sizeof(ncclWin)));
  win_->remotePtrs = static_cast<void**>(malloc(nRanks * sizeof(void*)));
  win_->ipcHandles = static_cast<cudaIpcMemHandle_t*>(
      malloc(nRanks * sizeof(cudaIpcMemHandle_t)));

  // get memory from mapper's pool
  void *addr, *hdl;
  comm->ctranMapper->getTmpBuf(&addr, size, &hdl);
  // open IPC handle
  CUDACHECK(cudaIpcGetMemHandle(
      (cudaIpcMemHandle_t*)&win_->ipcHandles[comm->rank], (void*)addr));

  // exchange IPC handles
  NCCLCHECK(bootstrapAllGather(
      comm->bootstrap, win_->ipcHandles, sizeof(cudaIpcMemHandle_t)));

  // open IPC handles and cache remote  address
  for (int i = 0; i < nRanks; ++i) {
    void* remoteAddr;
    if (i != comm->rank) {
      CUDACHECK(cudaIpcOpenMemHandle(
          (void**)&remoteAddr,
          win_->ipcHandles[i],
          cudaIpcMemLazyEnablePeerAccess));
    } else {
      remoteAddr = addr;
    }
  }

  *win = win_;
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinSharedQuery,
    int rank,
    ncclComm_t comm,
    ncclWin_t win,
    void** addr);
ncclResult_t
ncclWinSharedQuery(int rank, ncclComm_t comm, ncclWin_t win, void** addr) {
  *addr = win->remotePtrs[rank];

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclWinFree, ncclComm_t comm, ncclWin_t win);
ncclResult_t ncclWinFree(ncclComm_t comm, ncclWin_t win) {
  INFO(NCCL_INIT, "freeing window");
  for (int i = 0; i < comm->nRanks; ++i) {
    if (i != comm->rank) {
      CUDACHECK(cudaIpcCloseMemHandle((void*)win->remotePtrs[i]));
    }
  }
  // FIXME: segfault
  comm->ctranMapper->releaseTmpBuf(win->remotePtrs[comm->rank], nullptr);

  free(win->remotePtrs);
  free(win->ipcHandles);
  free(win);

  INFO(NCCL_INIT, "freed window");
  return ncclSuccess;
}
