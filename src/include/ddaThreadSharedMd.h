// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef DDA_THREAD_SHARED_MD_H_
#define DDA_THREAD_SHARED_MD_H_

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "checks.h"

class ddaThreadSharedMd {
 public:
  ddaThreadSharedMd(uint64_t commHash) {
    this->commHash = commHash;
  }

  void insertRank(int rank) {
    this->registeredRanks.push_back(rank);
  }

  void deleteRank(int rank) {
    this->registeredRanks.erase(std::remove(this->registeredRanks.begin(),
                                            this->registeredRanks.end(), rank),
                                this->registeredRanks.end());
  }

  bool searchRank(int rank) {
    auto got = std::find(this->registeredRanks.begin(), this->registeredRanks.end(), rank);
    return (got != this->registeredRanks.end());
  }

  uint64_t commHash;
  std::vector<int> registeredRanks;
};

#endif
