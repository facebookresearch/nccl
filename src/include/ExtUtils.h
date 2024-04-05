// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef EXT_UTILS_H
#define EXT_UTILS_H

#include "nccl.h"
#include "debug.h"

#include <cstdint>
#include <sstream>
#include <unordered_set>
#include <vector>

static inline std::string uint64ToHexStr(
    const uint64_t val,
    const std::string& prefix = "") {
  std::stringstream ss;
  ss << prefix << std::hex << val;
  return ss.str();
}

static inline std::string hashToHexStr(const uint64_t hash) {
  std::stringstream ss;
  ss << std::hex << hash;
  return ss.str();
}

template <typename T>
static inline std::string vecToStr(
    const std::vector<T>& vec,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (auto it : vec) {
    if (!first) {
      ss << delim;
    }
    ss << it;
    first = false;
  }
  return ss.str();
}


template <typename T>
static inline std::string unorderedSetToStr(
    const std::unordered_set<T>& vec,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (auto it : vec) {
    if (!first) {
      ss << delim;
    }
    ss << it;
    first = false;
  }
  return ss.str();
}

// Find the mapping between "name" and its enum value
template<size_t N> ncclResult_t strToEnum(std::string name, const char * (&mapping)[N], int *variant) {
    const auto it = std::find(std::begin(mapping), std::end(mapping), name);
    if(it!= std::end(mapping)){
        *variant = it - std::begin(mapping);
        return ncclSuccess;
    }

    std::string validValues;
    for(int i = 0; i < N; i++){
        validValues += mapping[i];
        if(i!= N-1) validValues += ", ";
    }

    WARN("Got %s, expected one of %s", name.c_str(), validValues.c_str());
    return ncclInvalidArgument;
}


inline std::string getDatatypeStr(ncclDataType_t type) {
  switch(type) {
    case ncclInt8:
      return "ncclInt8";
    case ncclUint8:
      return "ncclUint8";
    case ncclInt32:
      return "ncclInt32";
    case ncclUint32:
      return "ncclUint32";
    case ncclInt64:
      return "ncclInt64";
    case ncclUint64:
      return "ncclUint64";
    case ncclFloat16:
      return "ncclFloat16";
    case ncclFloat32:
      return "ncclFloat32";
    case ncclFloat64:
      return "ncclFloat64";
    case ncclBfloat16:
      return "ncclBfloat16";
    default:
      return "Unknown type";
  }
}

inline std::string getRedOpStr(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return "ncclSum";
    case ncclProd:
      return "ncclProd";
    case ncclMax:
      return "ncclMax";
    case ncclMin:
      return "ncclMin";
    case ncclAvg:
      return "ncclAvg";
    default:
      return "Unknown op";
  }
}
#endif
