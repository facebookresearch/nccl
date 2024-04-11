// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef EXT_UTILS_H
#define EXT_UTILS_H

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <thread>
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

static inline int64_t getTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto since_epoch = now.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
  int64_t timestamp = seconds.count();
  return timestamp;
}

static inline std::string getUniqueFileSuffix() {
  std::time_t t = std::time(nullptr);
  std::ostringstream time_str;
  time_str << std::put_time(std::localtime(&t), "%Y%m%d-%H%M%S");
  auto threadHash = std::hash<std::thread::id>{}(std::this_thread::get_id());

  return time_str.str() + "-" + std::to_string(threadHash);
}

static inline std::string getThreadUniqueId(std::string tag = "") {
  auto threadHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return std::to_string(threadHash) + (tag.empty() ? "" : "-" + tag);
}

#endif
