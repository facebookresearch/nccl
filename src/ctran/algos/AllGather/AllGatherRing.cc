// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <nccl.h>
#include "Ctran.h"
#include "comm.h"

static ncclResult_t impl(std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct OpElem* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  ncclComm_t comm = opGroup.front()->comm;
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  void *sendHdl, *recvHdl;
  bool localRegSend, localRegRecv;
  void* remoteRecvBuff;
  struct CtranMapperRemoteAccessKey remoteAccessKey;
  CtranMapper *mapper = comm->ctran->mapper.get();

  if (nRanks == 1) {
    return ncclSuccess;
  }

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAllGatherRing"));

  CtranMapperRequest* irecvReq;
  CtranMapperRequest* isendReq;
  CtranMapperRequest* iputReq;
  int left = (rank + nRanks - 1) % nRanks;
  int right = (rank + 1) % nRanks;

  NCCLCHECKGOTO(
      mapper->searchRegHandle(
          op->allgather.sendbuff, sendSize, &sendHdl, &localRegSend),
      res,
      exit);

  NCCLCHECKGOTO(
      mapper->searchRegHandle(
          op->allgather.recvbuff, nRanks * sendSize, &recvHdl, &localRegRecv),
      res,
      exit);

  NCCLCHECKGOTO(
      mapper->irecvCtrl(
          &remoteRecvBuff, &remoteAccessKey, right, &irecvReq),
      res,
      exit);
  NCCLCHECKGOTO(
      mapper->isendCtrl(
          op->allgather.recvbuff, recvHdl, left, &isendReq),
      res,
      exit);
  NCCLCHECKGOTO(irecvReq->wait(), res, exit);
  timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(right));

  NCCLCHECKGOTO(
      mapper->iput(
          op->allgather.sendbuff,
          (void*)((uintptr_t)remoteRecvBuff + rank * sendSize),
          sendSize,
          right,
          sendHdl,
          remoteAccessKey,
          true,
          (nRanks > 2) ? nullptr : &iputReq),
      res,
      exit);
  timestamp->putIssued.push_back(CtranMapperTimestampPoint(right));

  for (int i = 0; i < nRanks - 2; i++) {
    int blockId = (rank - i - 1 + nRanks) % nRanks;

    NCCLCHECKGOTO(mapper->waitNotify(left), res, exit);
    NCCLCHECKGOTO(
        mapper->iput(
            (void*)((uintptr_t)op->allgather.recvbuff + blockId * sendSize),
            (void*)((uintptr_t)remoteRecvBuff + blockId * sendSize),
            sendSize,
            right,
            recvHdl,
            remoteAccessKey,
            true,
            (i < nRanks - 3) ? nullptr : &iputReq),
        res,
        exit);
    timestamp->putIssued.push_back(CtranMapperTimestampPoint(right));
  }

  NCCLCHECKGOTO(mapper->waitNotify(left), res, exit);
  NCCLCHECKGOTO(isendReq->wait(), res, exit);

  NCCLCHECKGOTO(iputReq->wait(), res, exit);
  timestamp->putComplete.push_back(CtranMapperTimestampPoint(right));

  if (localRegSend == true) {
    NCCLCHECKGOTO(mapper->deregMem(sendHdl), res, exit);
  }
  if (localRegRecv == true) {
    NCCLCHECKGOTO(mapper->deregMem(recvHdl), res, exit);
  }

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

exit:
  return res;
}

ncclResult_t ctranAllGatherRing(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

  CTRAN_COLL_INFO("ctranAllGatherRing", sendbuff, recvbuff, sendcount, datatype, -1, comm, stream);

  /* copy data for out-of-place allgather */
  if ((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype) !=
      (uintptr_t)sendbuff) {
    CtranMapperRequest* req;
    comm->ctran->mapper->icopy(
        (void*)((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype)),
        sendbuff,
        sendcount * ncclTypeSize(datatype),
        stream,
        &req);
  }

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLGATHER, stream, comm));
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;

  opGroup.push_back(std::move(op));
  NCCLCHECKGOTO(
      comm->ctran->gpe->submit(
          std::move(opGroup),
          impl,
          reinterpret_cast<void*>(ncclKernelAllGatherCtranRing)),
      res,
      fail);

fail:
  return res;
}
