// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "ctranMapper.h"
#include "ctranMapperImpl.h"
#include "comm.h"

NCCL_PARAM(CtranProfiling, "CTRAN_PROFILING", 0);

ctranMapper::ctranMapper(ncclComm *comm) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  /* mapperRegElemList */
  this->pimpl->mapperRegElemList = new class ctranAvlTree();

  /* check user preference for backends */
  char *ctranBackendsStr = getenv("NCCL_CTRAN_BACKENDS");
  std::string s;
  if (ctranBackendsStr) {
    s = ctranBackendsStr;
  } else {
    s = "ib";
  }
  std::string delim = ",";

  while (auto pos = s.find(delim)) {
    std::string b = s.substr(0, pos);
    if (b == "nvl") {
      this->pimpl->backends.push_back(ctranMapperBackend::NVL);
    } else if (b == "ib") {
      this->pimpl->backends.push_back(ctranMapperBackend::IB);
    } else {
      WARN("CTRAN-MAPPER: Unknown backend %s specified", b.c_str());
    }
    s.erase(0, pos + delim.length());
    if (pos == std::string::npos) {
      break;
    }
  }

  /* enable backends that are possible */
  std::vector<enum ctranMapperBackend>::iterator it;

  this->pimpl->ctranIb = nullptr;
  it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
      ctranMapperBackend::IB);
  if (it != this->pimpl->backends.end()) {
    try {
      this->pimpl->ctranIb = std::unique_ptr<class ctranIb>(new class ctranIb(comm));
    } catch (const std::bad_alloc& e) {
      WARN("CTRAN: IB backend not enabled");
    }
  }

  this->pimpl->ctranNvl = nullptr;
  it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
      ctranMapperBackend::NVL);
  if (it != this->pimpl->backends.end()) {
    try {
      this->pimpl->ctranNvl = std::unique_ptr<class ctranNvl>(new class ctranNvl(comm));
    } catch (const std::bad_alloc& e) {
      WARN("CTRAN: Nvl backend not enabled");
    }
  }

  for (int i = 0; i < comm->nRanks; i++) {
    /* FIXME: we currently only support NVL for self communication */
    if (i == comm->rank && this->pimpl->ctranNvl != nullptr) {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::NVL);
    } else if (this->pimpl->ctranIb != nullptr) {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::IB);
    } else {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::UNSET);
    }
  }

  CUDACHECKIGNORE(cudaStreamCreateWithFlags(&this->s, cudaStreamNonBlocking));

  this->rank = comm->rank;
  this->commHash = comm->commHash;

  /* Memory pool */
  this->pimpl->memPool = new class ctranMapperMemPool();
  this->pimpl->memPool->regMem(
      [&](const void* buf, std::size_t len, void** hdl) -> ncclResult_t {
          return this->regMem(buf, len, hdl);
      });
}

ctranMapper::~ctranMapper() {
  /* flush timestamps */
  if (ncclParamCtranProfiling() && !this->timestamps.empty()) {
    std::cout << "[CTRAN-MAPPER] Communication Profiling:" << std::endl;
    for (auto& ts : this->timestamps) {
      std::cout << "    collective=" << ts.algo << std::endl;
      std::cout << "    startTime="
        << std::chrono::duration_cast<std::chrono::nanoseconds>(ts.start.time_since_epoch()).count()
        << std::endl;
      for (auto& tsp : ts.recvCtrl) {
        std::cout << "        recvCtrl[" << tsp.peer << "]="
          << std::chrono::duration_cast<std::chrono::nanoseconds>(tsp.now.time_since_epoch()).count()
          << std::endl;
      }
      for (auto& tsp : ts.putIssued) {
        std::cout << "        putIssued[" << tsp.peer << "]="
          << std::chrono::duration_cast<std::chrono::nanoseconds>(tsp.now.time_since_epoch()).count()
          << std::endl;
      }
      for (auto& tsp : ts.putComplete) {
        std::cout << "        putComplete[" << tsp.peer << "]="
          << std::chrono::duration_cast<std::chrono::nanoseconds>(tsp.now.time_since_epoch()).count()
          << std::endl;
      }
    }
    std::cout << std::flush;
  }

  if (this->pimpl->memPool != nullptr) {
    this->pimpl->memPool->deregMem(
      [&](void* hdl) -> ncclResult_t {
          return this->deregMem(hdl);
      });
  }

  std::vector<void *> v = this->pimpl->mapperRegElemList->getAllElems();
  if (!v.empty()) {
    WARN("CTRAN-Mapper: found %lu leaked registrations", v.size());
  }
  for (auto hdl : v) {
    NCCLCHECKIGNORE(this->deregMem(hdl));
    WARN("CTRAN-Mapper: leak hdl %p", hdl);
  }
  delete this->pimpl->mapperRegElemList;

  delete this->pimpl->memPool;

  CUDACHECKIGNORE(cudaStreamDestroy(this->s));
}

ncclResult_t ctranMapper::regMem(const void *buf, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  struct ctranMapperRegElem *mapperRegElem = new struct ctranMapperRegElem;

  cudaPointerAttributes attr;
  CUDACHECKGOTO(cudaPointerGetAttributes(&attr, buf), res, exit);
  if (attr.type != cudaMemoryTypeDevice) {
    WARN("CTRAN-MAPPER: buf %p is not a device buffer\n", buf);
    res = ncclSystemError;
    goto exit;
  }

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->regMem(buf, len, &mapperRegElem->ibRegElem), res, exit);
  }

  if (this->pimpl->ctranNvl != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranNvl->regMem(buf, len, &mapperRegElem->nvlRegElem), res, exit);
  }

  NCCLCHECKGOTO(this->pimpl->mapperRegElemList->insert(buf, len, reinterpret_cast<void *>(mapperRegElem), hdl), res, exit);

exit:
  return res;
}

ncclResult_t ctranMapper::deregMem(void *hdl) {
  ncclResult_t res = ncclSuccess;

  struct ctranMapperRegElem *mapperRegElem;
  NCCLCHECKGOTO(this->pimpl->mapperRegElemList->lookup(hdl, (void **) &mapperRegElem), res, exit);

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->deregMem(mapperRegElem->ibRegElem), res, exit);
  }

  if (this->pimpl->ctranNvl != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranNvl->deregMem(mapperRegElem->nvlRegElem), res, exit);
  }

  NCCLCHECKGOTO(this->pimpl->mapperRegElemList->remove(hdl), res, exit);
  delete mapperRegElem;

exit:
  return res;
}

ncclResult_t ctranMapper::searchRegHandle(const void *buf, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->mapperRegElemList->search(buf, len, hdl), res, exit);

exit:
  return res;
}

ncclResult_t ctranMapper::icopy(void *dbuf, const void *sbuf, std::size_t len, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  *req = new ctranMapperRequest(this);
  CUDACHECKGOTO(cudaMemcpyAsync(dbuf, sbuf, len, cudaMemcpyDefault, this->s), res, exit);

exit:
  return res;
}

ncclResult_t ctranMapper::progress(void) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->progress(), res, exit);
  }
  if (this->pimpl->ctranNvl != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranNvl->progress(), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranMapper::getTmpBuf(void** addr, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  this->tmpBufLock.lock();
  *hdl = nullptr;
  std::size_t bufLen;
  NCCLCHECKGOTO(this->pimpl->memPool->getBuf(len, addr, hdl, &bufLen), res, exit);
  if (*hdl == nullptr) {
    NCCLCHECKGOTO(this->regMem(*addr, bufLen, hdl), res, exit);
  }

exit:
  this->tmpBufLock.unlock();
  return res;
}

ncclResult_t ctranMapper::releaseTmpBuf(void* addr, void *hdl) {
  ncclResult_t res = ncclSuccess;

  this->tmpBufLock.lock();
  NCCLCHECKGOTO(this->pimpl->memPool->release(addr, hdl), res, exit);

exit:
  this->tmpBufLock.unlock();
  return res;
}

ncclResult_t ctranMapper::isendCtrl(void *buf, void *hdl, int rank, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    struct ctranMapperRegElem *mapperRegElem;
    NCCLCHECKGOTO(this->pimpl->mapperRegElemList->lookup(hdl, (void **) &mapperRegElem), res, exit);

    if (req == nullptr) {
      NCCLCHECKGOTO(this->pimpl->ctranIb->isendCtrl(buf, mapperRegElem->ibRegElem, rank, nullptr), res, exit);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->isendCtrl(buf, mapperRegElem->ibRegElem, rank, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return res;
}

ncclResult_t ctranMapper::irecvCtrl(void **buf, struct ctranMapperRemoteAccessKey *key, int rank,
    ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    if (req == nullptr) {
      NCCLCHECKGOTO(this->pimpl->ctranIb->irecvCtrl(buf, &key->ibKey, rank, nullptr), res, exit);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->irecvCtrl(buf, &key->ibKey, rank, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return res;
}

ncclResult_t ctranMapper::iput(const void *sbuf, void *dbuf, std::size_t len, int rank, void *shdl,
    struct ctranMapperRemoteAccessKey remoteAccessKey, bool notify, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    struct ctranMapperRegElem *mapperRegElem;
    NCCLCHECKGOTO(this->pimpl->mapperRegElemList->lookup(shdl, (void **) &mapperRegElem), res, exit);

    if (req == nullptr) {
      NCCLCHECKGOTO(this->pimpl->ctranIb->iput(sbuf, dbuf, len, rank, mapperRegElem->ibRegElem, remoteAccessKey.ibKey,
            notify, nullptr), res, exit);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->iput(sbuf, dbuf, len, rank, mapperRegElem->ibRegElem, remoteAccessKey.ibKey,
            notify, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return res;
}

ncclResult_t ctranMapper::checkNotify(int rank, bool *notify) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->checkNotify(rank, notify), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranMapper::waitNotify(int rank) {
  ncclResult_t res = ncclSuccess;

  bool notify = false;
  while (notify == false) {
    NCCLCHECKGOTO(this->checkNotify(rank, &notify), res, exit);
  }

exit:
  return res;
}

