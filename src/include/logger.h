// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_LOGGER_H
#define NCCL_LOGGER_H

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include "nccl_cvars.h"

class NcclLogger {
 public:
  static void init(FILE* ncclDebugFile);

  static void log(const std::string& msg, FILE* ncclDebugFile) noexcept;

  NcclLogger(const NcclLogger&) = delete;
  NcclLogger& operator=(const NcclLogger&) = delete;
  ~NcclLogger();

 private:
  void stop();
  void logThreadFunc();
  void writeToFile(const std::string& message);
  void enqueueLog(const std::string&) noexcept;

  NcclLogger(FILE*);

  // Ideally ncclDebugInit (the only function that is supposed to call NcclLogger::init)
  // is already protected to ensure it can only be called once, so we don't need to do
  // it ourselves. But just trying to be super confident that we will never initialize
  // the singleton twice.
  static std::atomic_flag singletonInitialized_;
  static std::unique_ptr<NcclLogger> singleton_;

  std::thread loggerThread_;
  std::unique_ptr<std::queue<std::string>> mergedMsgQueue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  FILE* debugFile;
  std::atomic<bool> stopThread{false};
};

#endif
