// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
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
#include "ExtChecks.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_IB_HCA
   type        : prefixed_stringlist
   prefixes    : ^, =
   default     :
   description : |-
     List of IB HCAs available for NCCL to use. The list is comma-separated;
     port numbers can be specified using the : symbol. An optional prefix ^
     indicates the list is an exclude list. A second optional prefix = indicates
     that the tokens are exact names, otherwise by default NCCL would treat each
     token as a prefix. Examples:
     - mlx5 : Use all ports of all cards starting with mlx5
     - =mlx5_0:1,mlx5_1:1 : Use ports 1 of cards mlx5_0 and mlx5_1.
     - ^=mlx5_1,mlx5_4 : Do not use cards mlx5_1 and mlx5_4.
     (this needs to be renamed to NCCL_IB_HCA_LIST eventually)

 - name        : NCCL_CTRAN_IB_TRAFFIC_PROFILNG
   type        : bool
   default     : false
   description : |-
     Enable IB transport traffic profiling.
     Disabled by default.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

#define CTRAN_IB_ANY_PORT -1

class RoceHca {
 public:
  RoceHca(std::string hcaStr) {
    std::string s = hcaStr;
    std::string delim = ":";

    auto pos = s.find(delim);
    if (pos == std::string::npos) {
      this->name = s;
    } else {
      this->name = s.substr(0, pos);
      s.erase(0, pos + delim.length());
      this->port = std::stoi(s);
    }
  }
  std::string name;
  int port{CTRAN_IB_ANY_PORT};
};

CtranIbSingleton &CtranIbSingleton::getInstance(void) {
  static CtranIbSingleton s;
  return s;
}

CtranIbSingleton::CtranIbSingleton(void) {
  std::vector<RoceHca> hcas;
  // Avoid copy triggered by resize
  hcas.reserve(NCCL_IB_HCA.size());

  for (const auto& hca: NCCL_IB_HCA) {
    // Copy value to each vector element so it can be freed automatically
    hcas.push_back(RoceHca(hca));
  }

  struct ibv_device **devs;
  std::vector<struct ibv_device *> devices;
  int nDevs;

  NCCLCHECKTHROW(wrap_ibv_symbols());
  NCCLCHECKTHROW(wrap_ibv_get_device_list(&devs, &nDevs));

  // Exact match: find each matching device from system returned list following
  // the specified sequence
  if (!NCCL_IB_HCA_PREFIX.compare("=")) {
    for (const auto& d: hcas) {
      for (int i = 0; i < nDevs; i++) {
        std::string nameStr = devs[i]->name;
        if (!nameStr.compare(d.name.c_str())) {
          devices.push_back(devs[i]);
          this->ports.push_back(d.port);
          break;
        }
      }
    }
  } else {
    // For exclude search and prefix search, traverse system returned list and
    // filter based on specified condition
    for (int i = 0; i < nDevs; i++) {
      bool found = false;
      int port = CTRAN_IB_ANY_PORT;
      std::string nameStr = devs[i]->name;

      // Exclude: include only if it does not match with anyone in the excluding list
      if (!NCCL_IB_HCA_PREFIX.compare("^")) {
        bool exclude = false;
        for (const auto& d: hcas) {
          if (!nameStr.compare(d.name.c_str())) {
            exclude = true;
            break;
          }
        }
        found = !exclude;

        // Prefix match: include if match with anyone in the specified list
      } else {
        for (const auto& d: hcas) {
          if (!nameStr.compare(0, d.name.length(), d.name)) {
            found = true;
            port = d.port;
            break;
          }
        }
      }
      if (found) {
        devices.push_back(devs[i]);
        this->ports.push_back(port);
      }
    }
  }

  if (devices.empty()) {
    WARN("CTRAN-IB : No active device found.");
    throw std::bad_alloc();
  }

  for (auto i = 0; i < devices.size(); i++) {
    struct ibv_context *context;
    struct ibv_pd *pd;
    NCCLCHECKTHROW(wrap_ibv_open_device(&context, devices[i]));
    NCCLCHECKTHROW(wrap_ibv_alloc_pd(&pd, context));

    this->contexts.push_back(context);
    this->pds.push_back(pd);
    this->devNames.push_back(devices[i]->name);
  }
  return;
}

CtranIbSingleton::~CtranIbSingleton() {
  {
    std::lock_guard<std::mutex> guard(this->commsMutex_);
    if (this->comms_.size()) {
      for (auto& it : this->comms_) {
        WARN(
            "CTRAN-IB: communicator %p are still alive when CtranIbSingleton is destroyed.",
            it);
      }
    }
  }

  if (NCCL_CTRAN_IB_TRAFFIC_PROFILNG) {
    std::lock_guard<std::mutex> guard(this->trafficRecordMutex_);
    for (auto& it : this->trafficPerDevice_) {
      INFO(
          NCCL_INIT,
          "CTRAN-IB: [traffic profiling] device %s total traffic: %ld bytes",
          it.first.c_str(),
          it.second);
    }
    for (auto& it : this->trafficPerQP_) {
      INFO(
          NCCL_INIT,
          "CTRAN-IB: [traffic profiling] qp %d total traffic: %ld bytes",
          it.first,
          it.second);
    }
  }

  for (auto pd : this->pds) {
    NCCLCHECKIGNORE(wrap_ibv_dealloc_pd(pd));
  }

  for (auto context : this->contexts) {
    NCCLCHECKIGNORE(wrap_ibv_close_device(context));
  }

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

std::unordered_map<std::string, size_t>
CtranIbSingleton::getDeviceTrafficSnapshot(void) {
  std::unordered_map<std::string, size_t> snapshot;
  std::lock_guard<std::mutex> guard(this->trafficRecordMutex_);
  for (auto& it : this->trafficPerDevice_) {
    snapshot[it.first] = it.second;
  }
  return snapshot;
}

std::unordered_map<uint32_t, size_t> CtranIbSingleton::getQpTrafficSnapshot(
    void) {
  std::unordered_map<uint32_t, size_t> snapshot;
  std::lock_guard<std::mutex> guard(this->trafficRecordMutex_);
  for (auto& it : this->trafficPerQP_) {
    snapshot[it.first] = it.second;
  }
  return snapshot;
}

void CtranIbSingleton::recordDeviceTraffic(
    struct ibv_context* ctx,
    size_t nbytes) {
  if (!NCCL_CTRAN_IB_TRAFFIC_PROFILNG)
    return;

  std::lock_guard<std::mutex> guard(this->trafficRecordMutex_);
  auto devName = std::string(ctx->device->name);

  if (this->trafficPerDevice_.count(devName) == 0) {
    this->trafficPerDevice_[devName] = 0;
  }
  this->trafficPerDevice_[devName] += nbytes;
}

void CtranIbSingleton::recordQpTraffic(struct ibv_qp* qp, size_t nbytes) {
  if (!NCCL_CTRAN_IB_TRAFFIC_PROFILNG)
    return;
  std::lock_guard<std::mutex> guard(this->trafficRecordMutex_);
  if (this->trafficPerQP_.count(qp->qp_num) == 0) {
    this->trafficPerQP_[qp->qp_num] = 0;
  }
  this->trafficPerQP_[qp->qp_num] += nbytes;
}

void CtranIbSingleton::commRef(ncclComm* comm) {
  std::lock_guard<std::mutex> guard(this->commsMutex_);
  this->comms_.insert(comm);
}

void CtranIbSingleton::commDeref(ncclComm* comm) {
  std::lock_guard<std::mutex> guard(this->commsMutex_);
  this->comms_.erase(comm);
}

CtranIb::CtranIb(ncclComm* comm) {
  bool foundPort = false;
  char ifName[MAX_IF_NAME_SIZE+1];
  union ncclSocketAddress ifAddr;
  int nIfs;

  this->pimpl_ = std::unique_ptr<Impl>(new Impl());
  this->pimpl_->comm = comm;

  CtranIbSingleton& s = CtranIbSingleton::getInstance();

  this->pimpl_->context = s.contexts[comm->cudaDev];
  this->pimpl_->pd = s.pds[comm->cudaDev];

  struct ibv_device_attr devAttr;
  NCCLCHECKTHROW(wrap_ibv_query_device(this->pimpl_->context, &devAttr));

  // Found available port for the given device
  for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
    struct ibv_port_attr portAttr;
    if (ncclSuccess !=
        wrap_ibv_query_port(this->pimpl_->context, port, &portAttr)) {
      // Allow to continue as long as we can find a usable port
      WARN(
          "CTRAN-IB : Unable to query port %d on device %s",
          port,
          s.devNames[comm->cudaDev].c_str());
      continue;
    }
    if (portAttr.state != IBV_PORT_ACTIVE) {
      continue;
    }
    if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND &&
        portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
      continue;
    }
    if (s.ports[comm->cudaDev] == CTRAN_IB_ANY_PORT ||
        port == s.ports[comm->cudaDev]) {
      this->pimpl_->port = port;
      foundPort = true;
      break;
    }
  }

  if (foundPort) {
    INFO(NCCL_INIT, "CTRAN-IB: using device %s, port %d commHash %lx commDesc %s", s.devNames[comm->cudaDev].c_str(), this->pimpl_->port, comm->commHash, comm->config.commDesc);
  } else {
    WARN("CTRAN-IB : No active port found on device %s. Disable IB backend.", s.devNames[comm->cudaDev].c_str());
    throw std::runtime_error(
        "CTRAN-IB : No active port found on device " +
        s.devNames[comm->cudaDev]);
  }

  /* The max CQEs would not be enough for us in the worst case, where
   * we have a lot of VCs, and there is a lot of posted messages on
   * each of the VCs.  Static partitioning would reduce the number of
   * CQEs available to each VC in the common case.  Instead, we are
   * making an assumption here that the progress thread will pull out
   * completion entries fast enough that we will never overflow the
   * CQ. */
  NCCLCHECKTHROW(wrap_ibv_create_cq(
      &this->pimpl_->cq,
      this->pimpl_->context,
      devAttr.max_cqe,
      nullptr,
      nullptr,
      0));

  // Locally initialize all virtual connections before launching the listen
  // thread
  for (int r = 0; r < this->pimpl_->comm->nRanks; r++) {
    this->pimpl_->vcList.push_back(new CtranIb::Impl::VirtualConn(this->pimpl_->context, this->pimpl_->pd,
          this->pimpl_->cq, this->pimpl_->port, r));
    this->pimpl_->numUnsignaledPuts.push_back(0);
  }

  // Record reference to CtranIbSingleton
  s.commRef(comm);

  nIfs = ncclFindInterfaces(ifName, &ifAddr, MAX_IF_NAME_SIZE, 1);
  if (nIfs <= 0) {
    WARN("CTRAN-IB: No socket interfaces found\n");
    throw std::runtime_error("CTRAN-IB : No socket interfaces found");
  } else {
    INFO(NCCL_INIT, "CTRAN-IB: socket interface set to %s", ifName);
  }

  NCCLCHECKTHROW(ncclSocketInit(&this->pimpl_->listenSocket, &ifAddr));
  NCCLCHECKTHROW(ncclSocketListen(&this->pimpl_->listenSocket));

  this->pimpl_->allListenSocketAddrs = static_cast<ncclSocketAddress*>(
      malloc(this->pimpl_->comm->nRanks * sizeof(ncclSocketAddress)));
  NCCLCHECKTHROW(ncclSocketGetAddr(
      &this->pimpl_->listenSocket,
      &this->pimpl_->allListenSocketAddrs[this->pimpl_->comm->rank]));

  NCCLCHECKTHROW(bootstrapAllGather(
      comm->bootstrap,
      this->pimpl_->allListenSocketAddrs,
      sizeof(ncclSocketAddress)));

  this->pimpl_->listenThread = std::thread{CtranIb::Impl::bootstrapAccept, this->pimpl_.get()};
  return;
}

std::string CtranIb::getIbDevName() {
  return std::string(this->pimpl_->context->device->name);
}

int CtranIb::getIbDevPort() {
  return this->pimpl_->port;
}

CtranIb::~CtranIb(void) {
  CtranIbSingleton& s = CtranIbSingleton::getInstance();

  NCCLCHECKIGNORE(this->pimpl_->bootstrapTerminate());
  this->pimpl_->listenThread.join();

  for (int r = 0; r < this->pimpl_->comm->nRanks; r++) {
    delete this->pimpl_->vcList[r];
  }

  free(this->pimpl_->allListenSocketAddrs);
  NCCLCHECKIGNORE(ncclSocketClose(&this->pimpl_->listenSocket));

  NCCLCHECKIGNORE(wrap_ibv_destroy_cq(this->pimpl_->cq));

  // this comm is being destroyed, thus dereference from CtranIbSingleton
  s.commDeref(this->pimpl_->comm);

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

ncclResult_t CtranIb::regMem(const void *buf, std::size_t len, void **ibRegElem) {
  ncclResult_t res = ncclSuccess;

  int pageSize = getpagesize();
  if (len <= pageSize) {
    WARN("CTRAN-IB: cannot register buffer, size (%lu) <= page size (%d)", len, pageSize);
    res = ncclSystemError;
    goto exit;
  }

  struct ibv_mr *mr;
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, this->pimpl_->pd, (void *) buf, len,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ), res, exit);
  *ibRegElem = reinterpret_cast<void *>(mr);

exit:
  return res;
}

ncclResult_t CtranIb::deregMem(void *ibRegElem) {
  ncclResult_t res = ncclSuccess;

  struct ibv_mr *mr = reinterpret_cast<struct ibv_mr *>(ibRegElem);
  NCCLCHECKGOTO(wrap_ibv_dereg_mr(mr), res, exit);

exit:
  return res;
}

ncclResult_t CtranIb::progress(void) {
  ncclResult_t res = ncclSuccess;

  /* complete as many requests as possible */
  while (1) {
    struct ibv_wc wc;
    int count;

    res = wrap_ibv_poll_cq(this->pimpl_->cq, 1, &wc, &count);
    NCCLCHECKGOTO(res, res, exit);

    if (count == 0) {
      break;
    }

    /* wc.wr_id is valid even if the poll_cq returned an error; use it
     * to gather information about the error */
    this->pimpl_->m.lock();
    auto vc = this->pimpl_->vcList[this->pimpl_->qpToRank[wc.qp_num]];
    this->pimpl_->m.unlock();

    if (wc.status != IBV_WC_SUCCESS) {
      WARN("CTRAN-IB: wrap_ibv_poll_cq failed, peerRank=%d, with status=%d, '%s'",
          vc->peerRank, wc.status, this->pimpl_->ibv_wc_status_str(wc.status));
      res = ncclSystemError;
      goto exit;
    }

    NCCLCHECKGOTO(vc->processCqe(wc.opcode, wc.qp_num, wc.imm_data), res, exit);
  }

  /* we should have pendingOps only if the connection was not
   * established yet.  The below algorithm is a bit inefficient, but
   * that is OK as it should not happen in the critical path. */
  if (!this->pimpl_->pendingOps.empty()) {
    std::vector<int> peerRanks;
    std::vector<struct PendingOp *> tmp = this->pimpl_->pendingOps;
    this->pimpl_->pendingOps.clear();

    for (auto op : tmp) {
      int rank = (op->type == PendingOp::PendingOpType::ISEND_CTRL) ? op->isendCtrl.peerRank :
        op->irecvCtrl.peerRank;

      /* if we already encounted this peer, skip all operations to the
       * same peer; otherwise we might end up sending messages out of
       * order */
      if (std::find(peerRanks.begin(), peerRanks.end(), rank) != peerRanks.end()) {
        this->pimpl_->pendingOps.push_back(op);
        continue;
      }

      auto vc = this->pimpl_->vcList[rank];
      if (op->type == PendingOp::PendingOpType::ISEND_CTRL) {
        if (vc->isReady() == true) {
          NCCLCHECKGOTO(vc->isendCtrl(op->isendCtrl.buf, op->isendCtrl.ibRegElem, op->isendCtrl.req), res, exit);
          delete op;
        } else {
          this->pimpl_->pendingOps.push_back(op);
          peerRanks.push_back(rank);
        }
      } else {
        if (vc->isReady() == true) {
          NCCLCHECKGOTO(vc->irecvCtrl(op->irecvCtrl.buf, op->irecvCtrl.key, op->irecvCtrl.req), res, exit);
          delete op;
        } else {
          this->pimpl_->pendingOps.push_back(op);
          peerRanks.push_back(rank);
        }
      }
    }
  }

exit:
  return res;
}

ncclResult_t CtranIb::isendCtrl(void *buf, void *ibRegElem, int peerRank, CtranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl_->vcList[peerRank];
  if (this->pimpl_->comm->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl_->bootstrapConnect(peerRank), res, exit);
  }

  *req = new CtranIbRequest();
  if (vc->isReady() == true) {
    NCCLCHECKGOTO(vc->isendCtrl(buf, ibRegElem, *req), res, exit);
  } else {
    auto pendingOp = new struct PendingOp;
    pendingOp->type = PendingOp::PendingOpType::ISEND_CTRL;
    pendingOp->isendCtrl.buf = buf;
    pendingOp->isendCtrl.ibRegElem = ibRegElem;
    pendingOp->isendCtrl.peerRank = peerRank;
    pendingOp->isendCtrl.req = *req;
    this->pimpl_->pendingOps.push_back(pendingOp);
  }

exit:
  return res;
}

ncclResult_t CtranIb::irecvCtrl(void **buf, struct CtranIbRemoteAccessKey *key, int peerRank,
    CtranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl_->vcList[peerRank];
  if (this->pimpl_->comm->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl_->bootstrapConnect(peerRank), res, exit);
  }

  *req = new CtranIbRequest();
  if (vc->isReady() == true) {
    NCCLCHECKGOTO(vc->irecvCtrl(buf, key, *req), res, exit);
  } else {
    auto pendingOp = new struct PendingOp;
    pendingOp->type = PendingOp::PendingOpType::IRECV_CTRL;
    pendingOp->irecvCtrl.buf = buf;
    pendingOp->irecvCtrl.key = key;
    pendingOp->irecvCtrl.peerRank = peerRank;
    pendingOp->irecvCtrl.req = *req;
    this->pimpl_->pendingOps.push_back(pendingOp);
  }

exit:
  return res;
}

ncclResult_t CtranIb::iput(const void *sbuf, void *dbuf, std::size_t len, int peerRank, void *ibRegElem,
    struct CtranIbRemoteAccessKey remoteAccessKey, bool notify, CtranIbRequest **req) {
  ncclResult_t res = ncclSuccess;
  CtranIbRequest *r = nullptr;

  if (req != nullptr) {
    *req = new CtranIbRequest();
    r = *req;
    this->pimpl_->numUnsignaledPuts[peerRank] = 0;
  } else {
    this->pimpl_->numUnsignaledPuts[peerRank]++;
    if (this->pimpl_->numUnsignaledPuts[peerRank] == MAX_SEND_WR) {
      r = &this->pimpl_->fakeReq;
      this->pimpl_->numUnsignaledPuts[peerRank] = 0;
    }
  }

  NCCLCHECKGOTO(this->pimpl_->vcList[peerRank]->iput(sbuf, dbuf, len, ibRegElem, remoteAccessKey, notify, r),
      res, exit);

exit:
  return res;
}

ncclResult_t CtranIb::checkNotify(int peerRank, bool* notify) {
  NCCLCHECK(this->progress());
  NCCLCHECK(this->pimpl_->vcList[peerRank]->checkNotify(notify));
  return ncclSuccess;
}

ncclResult_t CtranIb::waitNotify(int peerRank) {
  ncclResult_t res = ncclSuccess;
  bool notify = false;
  while (!notify) {
    NCCLCHECKGOTO(this->progress(), res, exit);
    NCCLCHECKGOTO(
        this->pimpl_->vcList[peerRank]->checkNotify(&notify), res, exit);
  }

exit:
  return res;
}
