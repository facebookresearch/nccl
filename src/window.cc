// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ncclWin.h"
#include <nccl_common.h>
#include <nccl.h>
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
      (comm->nNodes == 1 && comm->nRanks == comm->localRanks);
  int nRanks = comm->nRanks;
  if (!can_use_win) {
    WARN(
        "ncclCommWinAllocShared only supports intra-node communicator, current communicator has nNodes=%d, nRanks=%d, localRanks=%d",
        comm->nNodes,
        comm->nRanks,
        comm->localRanks);
    return ncclInvalidUsage;
  }

  // sanity check to make sure all peers can use IPC
  // exchange peer devices
  int* peerDevs = static_cast<int*>(malloc(nRanks * sizeof(int)));
  peerDevs[comm->rank] = comm->cudaDev;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, (void*)peerDevs, sizeof(int)));

  for (int i = 0; i < nRanks; ++i) {
    int canAccessPeer = 1;
    if (i != comm->rank) {
      CUDACHECK(
          cudaDeviceCanAccessPeer(&canAccessPeer, comm->cudaDev, peerDevs[i]));
    }
    if (canAccessPeer == 0) {
      WARN(
          "Rank %d with GPU-%d cannot access GPU-%d from peer %d",
          comm->rank,
          comm->cudaDev,
          peerDevs[i],
          i);
      return ncclInvalidUsage;
    }
  }
  free(peerDevs);

  // allocate resources
  ncclWin* win_ = static_cast<ncclWin*>(malloc(sizeof(ncclWin)));
  win_->comm = comm;
  win_->remotePtrs = static_cast<void**>(malloc(nRanks * sizeof(void*)));
  win_->ipcHandles = static_cast<cudaIpcMemHandle_t*>(
      malloc(nRanks * sizeof(cudaIpcMemHandle_t)));

  // TODO: use memory pool
  void *addr;
  TRACE(NCCL_ALLOC, "Cuda Alloc Size %ld pointer (cudaMalloc)", size);
  CUDACHECK(cudaMalloc(&addr, size));

  // No need to open IPC for sinlge-process communicator
  if (comm->nRanks > 1) {
    // open IPC handle
    CUDACHECK(cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t*)&win_->ipcHandles[comm->rank], (void*)addr));

    // exchange IPC handles
    NCCLCHECK(bootstrapAllGather(
        comm->bootstrap, win_->ipcHandles, sizeof(cudaIpcMemHandle_t)));
  }

  // open IPC handles and cache remote address
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
    win_->remotePtrs[i] = remoteAddr;
    INFO(
        NCCL_INIT,
        "Rank %d Opened IPC handle for rank %d with remoteAddr %p in ncclWin %p on ncclComm %p (nNodes=%d, nRanks=%d, localRanks=%d, commHash=%lx)",
        comm->rank,
        i,
        remoteAddr,
        win_,
        comm,
        comm->nNodes,
        comm->nRanks,
        comm->localRanks,
        comm->commHash);
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
  if (!comm || !win || comm != win->comm) {
    WARN("Invalid parameter(s) to query shared buffere in ncclWinSharedQuery: comm %p, win %p", comm, win);
    return ncclInvalidUsage;
  }

  *addr = win->remotePtrs[rank];

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclWinFree, ncclComm_t comm, ncclWin_t win);
ncclResult_t ncclWinFree(ncclComm_t comm, ncclWin_t win) {
  INFO(
        NCCL_INIT,
        "Rank %d freeing ncclWin %p on ncclComm %p (nNodes=%d, nRanks=%d, localRanks=%d, commHash=%lx)",
        comm->rank,
        win,
        comm,
        comm->nNodes,
        comm->nRanks,
        comm->localRanks,
        comm->commHash);

  for (int i = 0; i < comm->nRanks; ++i) {
    if (i != comm->rank) {
      CUDACHECK(cudaIpcCloseMemHandle((void*)win->remotePtrs[i]));
    }
  }

  CUDACHECK(cudaFree(win->remotePtrs[comm->rank]));

  free(win->remotePtrs);
  free(win->ipcHandles);
  free(win);

  return ncclSuccess;
}
