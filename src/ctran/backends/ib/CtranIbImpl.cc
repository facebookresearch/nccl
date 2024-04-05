// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <unistd.h>
#include "nccl.h"
#include "checks.h"
#include "CtranIb.h"
#include "CtranIbImpl.h"
#include "CtranIbVc.h"
#include "ExtUtils.h"
#include "ExtChecks.h"

void CtranIb::Impl::bootstrapAccept(CtranIb::Impl *pimpl) {
  NCCL_NAMED_THREAD_START("CTranIbListen");
  while (1) {
    struct ncclSocket sock;
    int peerRank;
    int cmd;

    NCCLCHECKTHROW(ncclSocketInit(&sock));
    NCCLCHECKTHROW(ncclSocketAccept(&sock, &pimpl->listenSocket));
    NCCLCHECKTHROW(ncclSocketRecv(&sock, &cmd, sizeof(int)));
    NCCLCHECKTHROW(ncclSocketRecv(&sock, &peerRank, sizeof(int)));

    if (cmd == BOOTSTRAP_CMD_TERMINATE) {
      NCCLCHECKTHROW(ncclSocketClose(&sock));
      break;
    }

    auto vc = pimpl->vcList[peerRank];

    {
      const std::lock_guard<std::mutex> lock(pimpl->m);

      /* exchange business cards */
      std::size_t size;
      void *localBusCard, *remoteBusCard;
      size = vc->getBusCardSize();
      localBusCard = malloc(size);
      remoteBusCard = malloc(size);
      NCCLCHECKTHROW(vc->getLocalBusCard(localBusCard));
      NCCLCHECKTHROW(ncclSocketRecv(&sock, remoteBusCard, size));
      NCCLCHECKTHROW(ncclSocketSend(&sock, localBusCard, size));

      uint32_t controlQp;
      std::vector<uint32_t> dataQps;
      NCCLCHECKTHROW(vc->setupVc(remoteBusCard, &controlQp, dataQps));
      pimpl->qpToRank[controlQp] = peerRank;
      for (auto qpn : dataQps) {
        pimpl->qpToRank[qpn] = peerRank;
      }

      INFO(
          NCCL_INIT,
          "CTRAN-IB: Established connection: rank %d, peer %d, control qpn %d, data qpns %s",
          pimpl->comm->rank,
          peerRank,
          controlQp,
          vecToStr(dataQps).c_str());

      free(localBusCard);
      free(remoteBusCard);

      /* Ack that the connection is fully established */
      int ack;
      NCCLCHECKTHROW(ncclSocketSend(&sock, &ack, sizeof(int)));
      NCCLCHECKTHROW(ncclSocketRecv(&sock, &ack, sizeof(int)));

      NCCLCHECKTHROW(ncclSocketClose(&sock));

    }
  }
  return;
}

ncclResult_t CtranIb::Impl::bootstrapConnect(int peerRank, int cmd) {
  ncclResult_t res = ncclSuccess;
  auto vc = this->vcList[peerRank];
  uint32_t controlQp;
  std::vector<uint32_t> dataQps;

  this->m.lock();

  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, &allListenSocketAddrs[peerRank]), res, exit);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, &cmd, sizeof(int)), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, &this->comm->rank, sizeof(int)), res, exit);

  if (peerRank == this->comm->rank) {
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, exit);
    goto exit;
  }

  /* exchange business cards */
  std::size_t size;
  void *localBusCard, *remoteBusCard;
  size = vc->getBusCardSize();
  localBusCard = malloc(size);
  remoteBusCard = malloc(size);
  NCCLCHECKGOTO(vc->getLocalBusCard(localBusCard), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, localBusCard, size), res, exit);
  NCCLCHECKGOTO(ncclSocketRecv(&sock, remoteBusCard, size), res, exit);

  NCCLCHECKGOTO(vc->setupVc(remoteBusCard, &controlQp, dataQps), res, exit);
  this->qpToRank[controlQp] = peerRank;
  for (auto qpn : dataQps) {
    this->qpToRank[qpn] = peerRank;
  }

  INFO(
      NCCL_INIT,
      "CTRAN-IB: Established connection: rank %d, peer %d, control qpn %d, data qpns %s",
      this->comm->rank,
      peerRank,
      controlQp,
      vecToStr(dataQps).c_str());

  free(localBusCard);
  free(remoteBusCard);

  /* Ack that the connection is fully established */
  int ack;
  NCCLCHECKGOTO(ncclSocketRecv(&sock, &ack, sizeof(int)), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, &ack, sizeof(int)), res, exit);

  NCCLCHECKGOTO(ncclSocketClose(&sock), res, exit);

exit:
  this->m.unlock();
  return res;
}

ncclResult_t CtranIb::Impl::bootstrapConnect(int peerRank) {
  return this->bootstrapConnect(peerRank, BOOTSTRAP_CMD_SETUP);
}


ncclResult_t CtranIb::Impl::bootstrapTerminate() {
  return this->bootstrapConnect(this->comm->rank, BOOTSTRAP_CMD_TERMINATE);
}

const char *CtranIb::Impl::ibv_wc_status_str(enum ibv_wc_status status) {
  switch (status) {
    case IBV_WC_SUCCESS: return "success";
    case IBV_WC_LOC_LEN_ERR: return "local length error";
    case IBV_WC_LOC_QP_OP_ERR: return "local QP operation error";
    case IBV_WC_LOC_EEC_OP_ERR: return "local EE context operation error";
    case IBV_WC_LOC_PROT_ERR: return "local protection error";
    case IBV_WC_WR_FLUSH_ERR: return "Work Request Flushed Error";
    case IBV_WC_MW_BIND_ERR: return "memory management operation error";
    case IBV_WC_BAD_RESP_ERR: return "bad response error";
    case IBV_WC_LOC_ACCESS_ERR: return "local access error";
    case IBV_WC_REM_INV_REQ_ERR: return "remote invalid request error";
    case IBV_WC_REM_ACCESS_ERR: return "remote access error";
    case IBV_WC_REM_OP_ERR: return "remote operation error";
    case IBV_WC_RETRY_EXC_ERR: return "transport retry counter exceeded";
    case IBV_WC_RNR_RETRY_EXC_ERR: return "RNR retry counter exceeded";
    case IBV_WC_LOC_RDD_VIOL_ERR: return "local RDD violation error";
    case IBV_WC_REM_INV_RD_REQ_ERR: return "remote invalid RD request";
    case IBV_WC_REM_ABORT_ERR: return "aborted error";
    case IBV_WC_INV_EECN_ERR: return "invalid EE context number";
    case IBV_WC_INV_EEC_STATE_ERR: return "invalid EE context state";
    case IBV_WC_FATAL_ERR: return "fatal error";
    case IBV_WC_RESP_TIMEOUT_ERR: return "response timeout error";
    case IBV_WC_GENERAL_ERR: return "general error";
    default: return "unrecognized error";
  }
}
