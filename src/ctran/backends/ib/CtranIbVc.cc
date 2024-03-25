// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>
#include "nccl.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "CtranIbBase.h"
#include "CtranIb.h"
#include "CtranIbImpl.h"
#include "CtranIbVc.h"
#include "CtranChecks.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_CTRAN_IB_MAX_QPS
   type        : int
   default     : 1
   description : |-
     Maximum number of QPs to enable, so data can be split across
     multiple QPs.  This allows the communication to take multiple routes
     and is a poor-man's version of fully adaptive routing.

 - name        : NCCL_CTRAN_IB_QP_SCALING_THRESHOLD
   type        : uint64_t
   default     : 1048576
   description : |-
     Threshold for QP scaling.  If T is the threshold, then for message sizes < T,
     a single QP is used.  For [T,2T) message sizes, data is split across two QPs.
     For [2T,3T) message sizes, data is split across three QPs, and so on.
     Once we hit the maximum number of QPs (see NCCL_CTRAN_IB_MAX_QPS), the
     data is split across all available QPs.

 - name        : NCCL_CTRAN_IB_CTRL_TC
   type        : uint64_t
   default     : 192
   description : |-
     Traffic class to use for control QPs. Note: To match NCCL_IB_TC, this directly
     sets the TC field, so multiply your DSCP value by 4.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

#define CTRAN_HARDCODED_MAX_QPS (128)

static inline int getQpNum(size_t nbytes) {
  int numQps = (nbytes / NCCL_CTRAN_IB_QP_SCALING_THRESHOLD) +
      !!(nbytes % NCCL_CTRAN_IB_QP_SCALING_THRESHOLD);
  if (numQps > NCCL_CTRAN_IB_MAX_QPS) {
    numQps = NCCL_CTRAN_IB_MAX_QPS;
  }
  return numQps;
}

// Business card describing the local IB connection info.
struct BusCard {
  enum ibv_mtu mtu;
  uint32_t controlQpn;
  uint32_t dataQpn[CTRAN_HARDCODED_MAX_QPS];
  uint8_t port;
  union {
    struct {
      uint64_t spn;
      uint64_t iid;
    } eth;
    struct {
      uint16_t lid;
    } ib;
  } u;
};

CtranIb::Impl::VirtualConn::VirtualConn(
    struct ibv_context* context,
    struct ibv_pd* pd,
    struct ibv_cq* cq,
    int port,
    int peerRank)
    : peerRank(peerRank), context_(context), pd_(pd), cq_(cq), port_(port) {
  if (NCCL_CTRAN_IB_MAX_QPS > CTRAN_HARDCODED_MAX_QPS) {
    WARN("CTRAN-IB: CTRAN_MAX_QPS set to more than the hardcoded max value (%d)", CTRAN_HARDCODED_MAX_QPS);
  }

  NCCLCHECKTHROW(wrap_ibv_reg_mr(
      &this->sendCtrl_.mr_,
      pd,
      (void*)this->sendCtrl_.cmsg_,
      MAX_CONTROL_MSGS * sizeof(struct ControlMsg),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ));

  NCCLCHECKTHROW(wrap_ibv_reg_mr(
      &this->recvCtrl_.mr_,
      pd,
      (void*)this->recvCtrl_.cmsg_,
      MAX_CONTROL_MSGS * sizeof(struct ControlMsg),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ));

  for (int i = 0; i < MAX_CONTROL_MSGS; i++) {
    this->sendCtrl_.freeMsgs_.push_back(&this->sendCtrl_.cmsg_[i]);
  }

  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    std::deque<CtranIbRequest *> q;
    this->put_.postedWrs_.push_back(q);
  }

  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    std::deque<uint32_t> q;
    this->notifications_.push_back(q);
  }
  return;
}

CtranIb::Impl::VirtualConn::~VirtualConn() {
  NCCLCHECKIGNORE(wrap_ibv_dereg_mr(this->sendCtrl_.mr_));
  NCCLCHECKIGNORE(wrap_ibv_dereg_mr(this->recvCtrl_.mr_));

  if (this->controlQp_ != nullptr) {
    /* we don't need to clean up the posted WQEs; destroying the QP
     * will automatically clear them */
    NCCLCHECKIGNORE(wrap_ibv_destroy_qp(this->controlQp_));
    for (auto qp : this->dataQps_) {
      NCCLCHECKIGNORE(wrap_ibv_destroy_qp(qp));
    }
  }
  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

bool CtranIb::Impl::VirtualConn::isReady() {
  bool r;

  this->m_.lock();
  r = this->isReady_;
  this->m_.unlock();

  return r;
}

void CtranIb::Impl::VirtualConn::setReady() {
  this->m_.lock();
  this->isReady_ = true;
  this->m_.unlock();
}

std::size_t CtranIb::Impl::VirtualConn::getBusCardSize() {
  return sizeof(struct BusCard);
}

ncclResult_t CtranIb::Impl::VirtualConn::getLocalBusCard(void *localBusCard) {
  ncclResult_t res = ncclSuccess;
  struct BusCard *busCard = reinterpret_cast<struct BusCard *>(localBusCard);

  struct ibv_port_attr portAttr;
  NCCLCHECKGOTO(
      wrap_ibv_query_port(this->context_, this->port_, &portAttr), res, exit);
  this->maxMsgSize_ = portAttr.max_msg_sz;
  this->linkLayer_ = portAttr.link_layer;

  /* create QP */
  struct ibv_qp_init_attr initAttr;
  memset(&initAttr, 0, sizeof(struct ibv_qp_init_attr));
  initAttr.send_cq = this->cq_;
  initAttr.recv_cq = this->cq_;
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = MAX_SEND_WR;
  initAttr.cap.max_recv_wr = MAX_CONTROL_MSGS;
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  initAttr.cap.max_inline_data = 0;
  NCCLCHECKGOTO(wrap_ibv_create_qp(&this->controlQp_, this->pd_, &initAttr), res, exit);
  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    struct ibv_qp *qp;
    NCCLCHECKGOTO(wrap_ibv_create_qp(&qp, this->pd_, &initAttr), res, exit);
    this->dataQps_.push_back(qp);
  }

  /* set QP to INIT state */
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = NCCL_IB_PKEY;
  qpAttr.port_num = this->port_;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->controlQp_, &qpAttr,
                       IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
    res, exit);

  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    NCCLCHECKGOTO(
        wrap_ibv_modify_qp(this->dataQps_[i], &qpAttr,
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
        res, exit);
  }

  /* create local business card */
  busCard->port = this->port_;
  busCard->controlQpn = this->controlQp_->qp_num;
  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    busCard->dataQpn[i] = this->dataQps_[i]->qp_num;
  }
  busCard->mtu = portAttr.active_mtu;

  if (this->linkLayer_ == IBV_LINK_LAYER_ETHERNET) {
    union ibv_gid gid;
    NCCLCHECKGOTO(wrap_ibv_query_gid(this->context_, this->port_, NCCL_IB_GID_INDEX, &gid), res, exit);
    busCard->u.eth.spn = gid.global.subnet_prefix;
    busCard->u.eth.iid = gid.global.interface_id;
  } else {
    busCard->u.ib.lid = portAttr.lid;
  }

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::setupVc(void *remoteBusCard, uint32_t *controlQp, std::vector<uint32_t>& dataQps) {
  ncclResult_t res = ncclSuccess;
  struct ibv_qp_attr qpAttr;
  struct BusCard *remoteBusCardStruct = reinterpret_cast<struct BusCard *>(remoteBusCard);

  *controlQp = this->controlQp_->qp_num;
  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    dataQps.push_back(this->dataQps_[i]->qp_num);
    this->qpNumToIdx_[this->dataQps_[i]->qp_num] = i;
  }

  /* set QP to RTR state */
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = remoteBusCardStruct->mtu;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;

  if (this->linkLayer_ == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = remoteBusCardStruct->u.eth.spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = remoteBusCardStruct->u.eth.iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = NCCL_IB_GID_INDEX;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = NCCL_CTRAN_IB_CTRL_TC;
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = remoteBusCardStruct->u.ib.lid;
  }
  qpAttr.ah_attr.sl = NCCL_IB_SL;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = remoteBusCardStruct->port;

  qpAttr.dest_qp_num = remoteBusCardStruct->controlQpn;
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->controlQp_, &qpAttr,
                       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                       IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), res, exit);

  if (this->linkLayer_ == IBV_LINK_LAYER_ETHERNET) {
    // Only use NCCL_CTRAN_IB_CTRL_TC for the control QP; switch back to NCCL_IB_TC for data QPs
    qpAttr.ah_attr.grh.traffic_class = NCCL_IB_TC;
  }

  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    qpAttr.dest_qp_num = remoteBusCardStruct->dataQpn[i];
    NCCLCHECKGOTO(
        wrap_ibv_modify_qp(this->dataQps_[i], &qpAttr,
          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), res, exit);
  }

  /* set QP to RTS state */
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = NCCL_IB_TIMEOUT;
  qpAttr.retry_cnt = NCCL_IB_RETRY_CNT;
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->controlQp_, &qpAttr,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                       IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), res, exit);

  for (int i = 0; i < NCCL_CTRAN_IB_MAX_QPS; i++) {
    NCCLCHECKGOTO(
        wrap_ibv_modify_qp(this->dataQps_[i], &qpAttr,
          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), res, exit);
  }

  /* post control WQEs */
  for (int i = 0; i < MAX_CONTROL_MSGS; i++) {
    NCCLCHECKGOTO(this->postRecvCtrlMsg(&this->recvCtrl_.cmsg_[i]), res, exit);
    this->recvCtrl_.postedMsgs_.push_back(&this->recvCtrl_.cmsg_[i]);

    for (int j = 0; j < NCCL_CTRAN_IB_MAX_QPS; j++) {
      NCCLCHECKGOTO(this->postRecvNotifyMsg(j), res, exit);
    }
  }

  this->setReady();

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::postRecvCtrlMsg(struct ControlMsg *cmsg) {
  ncclResult_t res = ncclSuccess;

  struct ibv_sge sg;
  memset(&sg, 0, sizeof(sg));
  sg.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cmsg));
  sg.length = sizeof(struct ControlMsg);
  sg.lkey = this->recvCtrl_.mr_->lkey;

  struct ibv_recv_wr postWr, *badWr;
  memset(&postWr, 0, sizeof(postWr));
  postWr.wr_id = 0;
  postWr.next = nullptr;
  postWr.sg_list = &sg;
  postWr.num_sge = 1;
  NCCLCHECKGOTO(wrap_ibv_post_recv(this->controlQp_, &postWr, &badWr), res, exit);

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::postSendCtrlMsg(struct ControlMsg *cmsg) {
  ncclResult_t res = ncclSuccess;

  struct ibv_sge sg;
  memset(&sg, 0, sizeof(sg));
  sg.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cmsg));
  sg.length = sizeof(struct ControlMsg);
  sg.lkey = this->sendCtrl_.mr_->lkey;

  struct ibv_send_wr postWr, *badWr;
  memset(&postWr, 0, sizeof(postWr));
  postWr.wr_id = 0;
  postWr.next = nullptr;
  postWr.sg_list = &sg;
  postWr.num_sge = 1;
  postWr.opcode = IBV_WR_SEND;
  postWr.send_flags = IBV_SEND_SIGNALED;
  NCCLCHECKGOTO(wrap_ibv_post_send(this->controlQp_, &postWr, &badWr), res, exit);

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::postPutMsg(const void *sbuf, void *dbuf, std::size_t len_,
    uint32_t lkey, uint32_t rkey, int numQps, bool localNotify, bool notify) {
  ncclResult_t res = ncclSuccess;
  uint64_t offset = 0;

  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  s.recordDeviceTraffic(this->context_, len_);

  for (int i = 0; i < numQps; i++) {
    uint64_t len = len_ / numQps;
    if (i == 0) {
      len += (len_ % numQps);
    }

    while (len > 0) {
      uint64_t toSend = std::min(len, static_cast<uint64_t>(this->maxMsgSize_));

      struct ibv_sge sg;
      memset(&sg, 0, sizeof(sg));
      sg.addr = reinterpret_cast<uint64_t>(sbuf) + offset;
      sg.length = toSend;
      sg.lkey = lkey;

      struct ibv_send_wr wr, *badWr;
      memset(&wr, 0, sizeof(wr));
      wr.wr_id = 0;
      wr.next = nullptr;
      wr.sg_list = &sg;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.send_flags = 0;

      // last packet of this QP
      if (len <= toSend) {
        // if remoteNotify is set, notify remote rank number of QPs to check
        if (notify == true) {
          wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
          wr.imm_data = numQps;
        }
        // if localNotify is set, signal local rank with CQE
        if (localNotify == true) {
          wr.send_flags = IBV_SEND_SIGNALED;
        }
      }
      wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dbuf) + offset;
      wr.wr.rdma.rkey = rkey;

      s.recordQpTraffic(this->dataQps_[i], toSend);

      NCCLCHECKGOTO(wrap_ibv_post_send(this->dataQps_[i], &wr, &badWr), res, exit);

      len -= toSend;
      offset += toSend;
    }
  }

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::postRecvNotifyMsg(int idx) {
  ncclResult_t res = ncclSuccess;

  struct ibv_recv_wr wr, *badWr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.next = nullptr;
  wr.num_sge = 0;

  NCCLCHECKGOTO(wrap_ibv_post_recv(this->dataQps_[idx], &wr, &badWr), res, exit);

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::processCqe(enum ibv_wc_opcode opcode, int qpNum, uint32_t immData) {
  ncclResult_t res = ncclSuccess;

  switch (opcode) {
    case IBV_WC_SEND:
      {
        auto req = this->sendCtrl_.postedReqs_.front();
        this->sendCtrl_.postedReqs_.pop_front();
        req->complete();

        auto cmsg = this->sendCtrl_.postedMsgs_.front();
        this->sendCtrl_.postedMsgs_.pop_front();
        this->sendCtrl_.freeMsgs_.push_back(cmsg);

        if (!this->sendCtrl_.enqueuedWrs_.empty()) {
          auto enqueuedWrs = this->sendCtrl_.enqueuedWrs_.front();
          this->sendCtrl_.enqueuedWrs_.pop_front();

          cmsg = this->sendCtrl_.freeMsgs_.front();
          this->sendCtrl_.freeMsgs_.pop_front();

          cmsg->remoteAddr = reinterpret_cast<uint64_t>(enqueuedWrs->enqueued.send.buf);
          cmsg->rkey = reinterpret_cast<struct ibv_mr *>(enqueuedWrs->enqueued.send.ibRegElem)->rkey;
          NCCLCHECKGOTO(this->postSendCtrlMsg(cmsg), res, exit);

          this->sendCtrl_.postedMsgs_.push_back(cmsg);
          this->sendCtrl_.postedReqs_.push_back(req);
          delete enqueuedWrs;
        }
      }
      break;

    case IBV_WC_RECV:
      {
        auto cmsg = this->recvCtrl_.postedMsgs_.front();
        this->recvCtrl_.postedMsgs_.pop_front();

        if (this->recvCtrl_.enqueuedWrs_.empty()) {
          /* unexpected message */
          struct ControlWr *unexpWrs = new struct ControlWr;
          unexpWrs->enqueued.unex.remoteAddr = cmsg->remoteAddr;
          unexpWrs->enqueued.unex.rkey = cmsg->rkey;
          this->recvCtrl_.unexpWrs_.push_back(unexpWrs);
        } else {
          auto enqueuedWrs = this->recvCtrl_.enqueuedWrs_.front();
          this->recvCtrl_.enqueuedWrs_.pop_front();

          *(enqueuedWrs->enqueued.recv.buf) = reinterpret_cast<void *>(cmsg->remoteAddr);
          enqueuedWrs->enqueued.recv.key->rkey = cmsg->rkey;
          enqueuedWrs->enqueued.recv.req->complete();

          delete enqueuedWrs;
        }

        NCCLCHECKGOTO(this->postRecvCtrlMsg(cmsg), res, exit);
        this->recvCtrl_.postedMsgs_.push_back(cmsg);
      }
      break;

    case IBV_WC_RDMA_WRITE:
      {
        int idx = this->qpNumToIdx_[qpNum];
        auto req = this->put_.postedWrs_[idx].front();
        this->put_.postedWrs_[idx].pop_front();
        req->complete();
      }
      break;

    case IBV_WC_RECV_RDMA_WITH_IMM:
      {
        int idx = this->qpNumToIdx_[qpNum];
        this->notifications_[idx].push_back(immData);
        NCCLCHECKGOTO(this->postRecvNotifyMsg(idx), res, exit);
      }
      break;

    default:
      WARN("CTRAN-IB: Found unknown opcode: %d", opcode);
      res = ncclSystemError;
      goto exit;
  }

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::isendCtrl(void *buf, void *ibRegElem, CtranIbRequest *req) {
  ncclResult_t res = ncclSuccess;

  if (this->sendCtrl_.freeMsgs_.empty()) {
    auto enqueuedWrs = new struct ControlWr;
    enqueuedWrs->enqueued.send.buf = buf;
    enqueuedWrs->enqueued.send.ibRegElem = ibRegElem;
    enqueuedWrs->enqueued.send.req = req;
    this->sendCtrl_.enqueuedWrs_.push_back(enqueuedWrs);
  } else {
    auto cmsg = this->sendCtrl_.freeMsgs_.front();
    this->sendCtrl_.freeMsgs_.pop_front();

    cmsg->remoteAddr = reinterpret_cast<uint64_t>(buf);
    cmsg->rkey = reinterpret_cast<struct ibv_mr *>(ibRegElem)->rkey;
    NCCLCHECKGOTO(this->postSendCtrlMsg(cmsg), res, exit);
    this->sendCtrl_.postedMsgs_.push_back(cmsg);
    this->sendCtrl_.postedReqs_.push_back(req);
  }

exit:
  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::irecvCtrl(void **buf, struct CtranIbRemoteAccessKey *key, CtranIbRequest *req) {
  ncclResult_t res = ncclSuccess;

  if (this->recvCtrl_.unexpWrs_.empty()) {
    auto enqueuedWrs = new struct ControlWr;
    enqueuedWrs->enqueued.recv.buf = buf;
    enqueuedWrs->enqueued.recv.key = key;
    enqueuedWrs->enqueued.recv.req = req;
    this->recvCtrl_.enqueuedWrs_.push_back(enqueuedWrs);
  } else {
    auto unexpWrs = this->recvCtrl_.unexpWrs_.front();
    this->recvCtrl_.unexpWrs_.pop_front();

    *buf = reinterpret_cast<void *>(unexpWrs->enqueued.unex.remoteAddr);
    key->rkey = unexpWrs->enqueued.unex.rkey;
    req->complete();

    delete unexpWrs;
  }

  return res;
}

ncclResult_t CtranIb::Impl::VirtualConn::iput(const void *sbuf, void *dbuf, std::size_t len, void *ibRegElem,
    struct CtranIbRemoteAccessKey remoteAccessKey, bool notify, CtranIbRequest *req) {
  ncclResult_t res = ncclSuccess;
  struct ibv_mr *smr;
  uint32_t rkey;

  smr = reinterpret_cast<struct ibv_mr *>(ibRegElem);
  if (smr == nullptr) {
    WARN("CTRAN-IB: memory registration not found for addr %p", sbuf);
    return ncclSystemError;
  }

  rkey = remoteAccessKey.rkey;
  // QP scaling
  int numQps = getQpNum(len);

  bool localNotify;
  if (req != nullptr) {
    localNotify = true;
    req->setRefCount(numQps);
    for (int i = 0; i < numQps; i++) {
      this->put_.postedWrs_[i].push_back(req);
    }
  } else {
    localNotify = false;
  }

  NCCLCHECK(this->postPutMsg(sbuf, dbuf, len, smr->lkey, rkey, numQps, localNotify, notify));

  return ncclSuccess;
}

ncclResult_t CtranIb::Impl::VirtualConn::checkNotify(bool *notify_) {
  bool notify = false;
  if (!this->notifications_[0].empty()) {
    // Always get numQps from first QP
    auto numQps = this->notifications_[0].front();

    // Check if numQps is valid in case network transfer is corrupted
    if (numQps <= 0 || numQps > NCCL_CTRAN_IB_MAX_QPS) {
      WARN("CTRAN-IB: Invalid numQps %d encoded in remote put", numQps);
      return ncclSystemError;
    }

    // Return true only when received notification from all QPs
    notify = true;
    for (int i = 0; i < numQps; i++) {
      if (this->notifications_[i].empty()) {
        notify = false;
        break;
      }
    }

    // Cleanup current set of notifications only after all QPs' data transfer
    // has completed
    if (notify == true) {
      for (int i = 0; i < numQps; i++) {
        this->notifications_[i].pop_front();
      }
    }
  }

  *notify_ = notify;
  return ncclSuccess;
}
