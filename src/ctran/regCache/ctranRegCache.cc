// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <unistd.h>
#include <iostream>
#include <algorithm>
#include "ctranRegCache.h"
#include "ctranRegCacheImpl.h"
#include "debug.h"

ctranRegCache::ctranRegCache() {
  this->pimpl = std::unique_ptr<ctranRegCache::impl>(new ctranRegCache::impl());
}

ctranRegCache::~ctranRegCache() {
  for (auto e : this->pimpl->root) {
    delete e;
  }
}

ncclResult_t ctranRegCache::insert(const void *addr, std::size_t len, void *val, void **hdl) {
  ncclResult_t res = ncclSuccess;

  struct regElem *e = new struct regElem();

  e->addr = reinterpret_cast<uintptr_t>(addr);
  e->len = len;
  e->val = val;

  this->pimpl->root.push_back(e);
  *hdl = reinterpret_cast<void *>(e);

  return res;
}

ncclResult_t ctranRegCache::remove(void *hdl) {
  ncclResult_t res = ncclSuccess;

  struct regElem *e = reinterpret_cast<struct regElem *>(hdl);
  this->pimpl->root.erase(std::remove(this->pimpl->root.begin(), this->pimpl->root.end(), e), this->pimpl->root.end());

  return res;
}

ncclResult_t ctranRegCache::search(const void *addr, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  *hdl = nullptr;

  for (auto e : this->pimpl->root) {
    if (e->addr <= a && (e->addr + e->len >= a + len)) {
      *hdl = e;
    }
  }

  if (*hdl == nullptr) {
    TRACE("CTRAN-REGCACHE: buffer %p, %lu bytes is not registered", addr, len);
  }

  return res;
}

ncclResult_t ctranRegCache::lookup(void *hdl, void **val) {
  ncclResult_t res = ncclSuccess;

  struct regElem *e = reinterpret_cast<struct regElem *>(hdl);
  *val = e->val;

  return res;
}

std::vector<void *> ctranRegCache::flush() {
  std::vector<void *> v;

  for (auto elem : this->pimpl->root) {
    v.push_back(reinterpret_cast<void *>(elem));
  }

  return v;
}
