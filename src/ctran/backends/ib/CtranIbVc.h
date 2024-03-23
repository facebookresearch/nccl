// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_VC_H_
#define CTRAN_IB_VC_H_

#include <vector>
#include <unordered_map>
#include <mutex>
#include <deque>
#include "ibvwrap.h"
#include "CtranIbBase.h"

#define MAX_CONTROL_MSGS (128)

/**
 * Virtual connection to manage the IB backend connection between two peers
 * and the internal data transfer.
 */
class CtranIb::Impl::VirtualConn {
  public:
    // Prepare local resources for the virtual connection.
    // Actual connection happens only when setupVc is called.
    VirtualConn(struct ibv_context *context, struct ibv_pd *pd, struct ibv_cq *cq,
        int port, int peerRank);
    ~VirtualConn();

    // A vc becomes ready after setupVC has been successfully caleld.
    bool isReady();

    // Get the size of local business card so that socket knows the bytes sent
    // to the other rank.
    std::size_t getBusCardSize();

    // Create local IB connection resource (QPs) and get the local business card
    // that describes the local resource. It will be exchanged with the peer via
    // socket (see bootstrapAccept|bootstrapConnect) to setup the remote
    // connection (see setupVc).
    ncclResult_t getLocalBusCard(void *busCard);

    // Setup the IB connection between two peers.
    // Specifically, it updates the local control and data QPs with remote
    // business card info to establish the connection.
    ncclResult_t setupVc(void *remoteBusCard, uint32_t *controlQp, std::vector<uint32_t> &dataQps);

    // Implementation to send control message over the established IB
    // connection. The sendCtrl msg may be queued in sendCtrl_.enqueuedWrs_ if
    // run out of pre-created MAX_CONTROL_MSGS sendCtrl_.freeMsgs_. The queued
    // msgs will be progressed whenever CtranIb calls into progress (see
    // CtranIb::progress).
    ncclResult_t isendCtrl(void *buf, void *ibRegElem, CtranIbRequest *req);

    // Implementation to receive control message over the established IB
    // connection. It first checks if any already received control message is
    // available at recvCtrl_.unexpWrs, if not it enqueues a receive work
    // request (WR) into recvCtrl_.enqueuedWrs_.
    ncclResult_t irecvCtrl(void **buf, struct CtranIbRemoteAccessKey *key, CtranIbRequest *req);

    // Implementation to put data from local sbuf to a dbuf in remote rank over
    // the established IB connection.
    ncclResult_t iput(const void *sbuf, void *dbuf, std::size_t len, void *ibRegElem,
        struct CtranIbRemoteAccessKey remoteAccessKey, bool notify, CtranIbRequest *req);

    // Implementation to process a compeletion queue element (CQE) that received
    // in ctranIb::progress.
    ncclResult_t processCqe(enum ibv_wc_opcode opcode, int qpNum, uint32_t immData);

    // Implementation to check the notification associated with a remote put.
    // It will return true if the notification is received, which indicates the
    // completion of the remote put.
    bool checkNotify();

    // Global rank of remote peer.
    int peerRank;

  private:
    void setReady();
    ncclResult_t postRecvCtrlMsg(struct ControlMsg *cmsg);
    ncclResult_t postSendCtrlMsg(struct ControlMsg *cmsg);
    ncclResult_t postPutMsg(const void *sbuf, void *dbuf, std::size_t len,
        uint32_t lkey, uint32_t rkey, bool localNotify, bool notify);
    ncclResult_t postRecvNotifyMsg(int idx);

    struct ibv_qp *controlQp_{nullptr};
    std::vector<struct ibv_qp *> dataQps_;

    struct {
      struct ControlWr wr_[MAX_CONTROL_MSGS];
      struct ControlMsg cmsg_[MAX_CONTROL_MSGS];
      struct ibv_mr *mr_;
      std::deque<struct ControlMsg *> freeMsgs_;
      std::deque<struct ControlMsg *> postedMsgs_;
      std::deque<CtranIbRequest *> postedReqs_;
      std::deque<struct ControlWr *> enqueuedWrs_;
    } sendCtrl_;
    struct {
      struct ControlWr wr_[MAX_CONTROL_MSGS];
      struct ControlMsg cmsg_[MAX_CONTROL_MSGS];
      struct ibv_mr *mr_;
      std::deque<struct ControlMsg *> postedMsgs_;
      std::deque<struct ControlWr *> unexpWrs_;
      std::deque<struct ControlWr *> enqueuedWrs_;
    } recvCtrl_;
    struct {
      std::vector<std::deque<CtranIbRequest *>> postedWrs_;
    } put_;

    bool isReady_{false};
    struct ibv_context* context_{nullptr};
    struct ibv_pd* pd_{nullptr};
    struct ibv_cq* cq_{nullptr};
    int port_{0};
    uint32_t maxMsgSize_{0};
    uint8_t linkLayer_{0};
    std::mutex m_;
    // notification is based imm_data field in ibv_send_wr which is uint32_t
    std::vector<std::deque<uint32_t>> notifications_;
    std::unordered_map<int,int> qpNumToIdx_;
};

#endif
