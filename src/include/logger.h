// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_LOGGER_H
#define NCCL_LOGGER_H

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

class NcclLogger {
 public:
  void log(const std::string&);
  static NcclLogger& getInstance(FILE* ncclDebugFile) {
    static NcclLogger ncclDebugLogger_t(ncclDebugFile);
    return ncclDebugLogger_t;
  }

  NcclLogger(const NcclLogger&) = delete;
  NcclLogger& operator=(const NcclLogger&) = delete;

 private:
  void stop();
  void logThreadFunc();
  void writeToFile(const std::string& message);

  NcclLogger(FILE*);
  ~NcclLogger();

  std::thread loggerThread_;
  std::unique_ptr<std::queue<std::string>> mergedMsgQueue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  FILE* debugFile;
  std::atomic<bool> stopThread{false};
};

#endif
