// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <group.h>
#include <deque>
#include "Ctran.h"
#include "CtranGpe.h"
#include "CtranMapper.h"
#include "comm.h"

thread_local std::deque<struct OpElem*> CtranOpGroup;

static ncclResult_t sendRecvImpl(
    std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  ncclResult_t res = ncclSuccess;
  std::vector<struct OpElem*> sendOpGroup;

  std::vector<struct OpElem*> recvOpGroup;

  for (auto& op : opGroup) {
    if (op->type == OpElem::opType::SEND) {
      sendOpGroup.push_back(op.get());
    } else {
      recvOpGroup.push_back(op.get());
    }
  }

  std::vector<void*> sendMemHdl(sendOpGroup.size());
  std::vector<void*> remoteRecvBuff(sendOpGroup.size());
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKey(
      sendOpGroup.size());
  std::vector<CtranMapperRequest*> sendCtrlReqs(sendOpGroup.size());
  std::vector<CtranMapperRequest*> putReqs(sendOpGroup.size());
  std::vector<bool> putIssued(sendOpGroup.size());

  std::vector<void*> recvMemHdl(recvOpGroup.size());
  std::vector<CtranMapperRequest*> recvCtrlReqs(recvOpGroup.size());
  std::vector<int> recvPeerRanks(recvOpGroup.size());
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranSendRecv"));

  ncclComm_t comm = opGroup.front()->comm;

  std::vector<void*> tmpRegHdls;

  /* issue control messages for send operations */
  for (auto i = 0; i < sendOpGroup.size(); i++) {
    auto op = sendOpGroup[i];
    size_t sendSize = op->send.count * ncclTypeSize(op->send.datatype);
    bool localReg = false;

    NCCLCHECKGOTO(
        comm->ctran->mapper->searchRegHandle(
            op->send.sendbuff, sendSize, &sendMemHdl[i], &localReg),
        res,
        exit);
    if (localReg) {
      tmpRegHdls.push_back(sendMemHdl[i]);
    }

    NCCLCHECKGOTO(
        comm->ctran->mapper->irecvCtrl(
            &remoteRecvBuff[i],
            &remoteAccessKey[i],
            op->send.peerRank,
            &sendCtrlReqs[i]),
        res,
        exit);
    putIssued[i] = false;
  }

  /* issue control messages for recv operations */
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto op = recvOpGroup[i];
    size_t recvSize = op->recv.count * ncclTypeSize(op->recv.datatype);
    bool localReg = false;

    NCCLCHECKGOTO(
        comm->ctran->mapper->searchRegHandle(
            op->recv.recvbuff, recvSize, &recvMemHdl[i], &localReg),
        res,
        exit);
    if (localReg) {
      tmpRegHdls.push_back(recvMemHdl[i]);
    }

    NCCLCHECKGOTO(
        comm->ctran->mapper->isendCtrl(
            op->recv.recvbuff,
            recvMemHdl[i],
            op->recv.peerRank,
            &recvCtrlReqs[i]),
        res,
        exit);
    recvPeerRanks[i] = op->recv.peerRank;
  }

  /* as we recv control msgs, issue PUT operations */
  while (1) {
    bool pendingOps = false;

    for (auto i = 0; i < sendOpGroup.size(); i++) {
      if (putIssued[i] == true) {
        continue;
      } else {
        auto op = sendOpGroup[i];
        size_t sendSize = op->send.count * ncclTypeSize(op->send.datatype);
        bool isComplete;

        NCCLCHECKGOTO(sendCtrlReqs[i]->test(&isComplete), res, exit);
        if (isComplete) {
          timestamp->recvCtrl.push_back(
              CtranMapperTimestampPoint(op->send.peerRank));
          NCCLCHECKGOTO(
              comm->ctran->mapper->iput(
                  op->send.sendbuff,
                  remoteRecvBuff[i],
                  sendSize,
                  op->send.peerRank,
                  sendMemHdl[i],
                  remoteAccessKey[i],
                  true,
                  &putReqs[i]),
              res,
              exit);
          timestamp->putIssued.push_back(
              CtranMapperTimestampPoint(op->send.peerRank));
          putIssued[i] = true;
        } else {
          pendingOps = true;
        }
      }
    }

    if (pendingOps == false) {
      break;
    }
  }

  /* wait for all PUT messages to complete */
  for (auto i = 0; i < sendOpGroup.size(); i++) {
    NCCLCHECKGOTO(putReqs[i]->wait(), res, exit);
    timestamp->putComplete.push_back(
        CtranMapperTimestampPoint(sendOpGroup[i]->send.peerRank));
  }

  /* wait for all control messages and notifications to complete */
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    NCCLCHECKGOTO(recvCtrlReqs[i]->wait(), res, exit);
    NCCLCHECKGOTO(comm->ctran->mapper->waitNotify(recvPeerRanks[i]), res, exit);
  }

  /* deregister temporary registrations */
  for (auto hdl : tmpRegHdls) {
    NCCLCHECKGOTO(comm->ctran->mapper->deregMem(hdl), res, exit);
  }

  comm->ctran->mapper->timestamps.push_back(std::move(timestamp));
  comm->ctran->mapper->reportProfiling();

exit:
  return res;
}

bool ctranSendRecvSupport(int peer, ncclComm_t comm) {
  // TODO: conrrently support Ctran sendrecv only when peer is at remote node.
  // We will include intranode support when bringing in NVL backend.
  if (!ctranInitialized(comm) || comm->rankToNode[peer] == comm->node) {
    return false;
  } else {
    return true;
  }
}

ncclResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      "CtranSend", sendbuff, nullptr, count, datatype, peer, comm, stream);

  ncclResult_t res = ncclSuccess;
  struct OpElem* op;

  op = new OpElem(OpElem::opType::SEND, stream, comm);
  op->send.sendbuff = sendbuff;
  op->send.count = count;
  op->send.datatype = datatype;
  op->send.peerRank = peer;

  CtranOpGroup.push_back(op);

  return res;
}

ncclResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      "CtranRecv", nullptr, recvbuff, count, datatype, peer, comm, stream);

  ncclResult_t res = ncclSuccess;
  struct OpElem* op;

  op = new OpElem(OpElem::opType::RECV, stream, comm);
  op->recv.recvbuff = recvbuff;
  op->recv.count = count;
  op->recv.datatype = datatype;
  op->recv.peerRank = peer;

  CtranOpGroup.push_back(op);

  return res;
}

static inline void setSendRecvKernelArgs(
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    CtranAlgoDeviceState* devState_d,
    CtranKernelArgs* args) {
  // opGroup is unused for now as we don't support NVL path yet.
  args->devState_d = devState_d;
}

ncclResult_t ctranGroupEndHook(void) {
  ncclComm_t comm;
  cudaStream_t stream;

  while (1) {
    std::vector<std::unique_ptr<struct OpElem>> toSubmit;
    std::deque<struct OpElem*> pending;
    bool hasSend = false;
    bool hasRecv = false;

    if (CtranOpGroup.empty()) {
      break;
    }

    // submit ops with the same comm and stream in a single batch
    comm = CtranOpGroup.front()->comm;
    stream = CtranOpGroup.front()->stream;
    while (!CtranOpGroup.empty()) {
      struct OpElem* op = CtranOpGroup.front();
      CtranOpGroup.pop_front();

      if (op->comm == comm && op->stream == stream) {
        toSubmit.push_back(std::unique_ptr<struct OpElem>(op));
        if (op->type == OpElem::opType::SEND) {
          hasSend = true;
        } else if (op->type == OpElem::opType::RECV) {
          hasRecv = true;
        }
      } else {
        // if not belong to this batch, put back to pending queue and handled in
        // next batch
        pending.push_back(op);
      }
    }

    if (hasSend && hasRecv) {
      auto config = KernelConfig(KernelConfig::KernelType::SENDRECV, stream);
      setSendRecvKernelArgs(
          toSubmit, comm->ctran->algo->devState_d, &config.args);
      NCCLCHECK(comm->ctran->gpe->submit(
          std::move(toSubmit),
          sendRecvImpl,
          config,
          reinterpret_cast<void*>(ncclKernelSendRecv)));
    } else if (hasSend) {
      auto config = KernelConfig(KernelConfig::KernelType::SEND, stream);
      setSendRecvKernelArgs(
          toSubmit, comm->ctran->algo->devState_d, &config.args);
      NCCLCHECK(comm->ctran->gpe->submit(
          std::move(toSubmit),
          sendRecvImpl,
          config,
          reinterpret_cast<void*>(ncclKernelSend)));
    } else if (hasRecv) {
      auto config = KernelConfig(KernelConfig::KernelType::RECV, stream);
      setSendRecvKernelArgs(
          toSubmit, comm->ctran->algo->devState_d, &config.args);
      NCCLCHECK(comm->ctran->gpe->submit(
          std::move(toSubmit),
          sendRecvImpl,
          config,
          reinterpret_cast<void*>(ncclKernelRecv)));
    }

    toSubmit.clear();
    // All ops in a group should use the same opCount. Thus, increase opCount if
    // only CTran ops are enqueued; otherwise let default path increase.
    if (comm->ctran->numGroupedDefaultOps == 0) {
      comm->opCount++;
    }
    comm->ctran->numGroupedDefaultOps = 0;

    // handle next batch
    CtranOpGroup = std::move(pending);
  }

  return ncclSuccess;
}

void ctranGroupTrackDefaultOp(ncclComm* comm) {
  if (ctranInitialized(comm)) {
    comm->ctran->numGroupedDefaultOps++;
  }
}
