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

// Friend class for testing. This class is used to access the internal status of
// NcclLogger.
class NcclLoggerTest;

class NcclLogger {
 public:
  // Friend class for testing
  friend class NcclLoggerTest;

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

  static std::unique_ptr<NcclLogger> singleton_;

  std::thread loggerThread_;
  std::unique_ptr<std::queue<std::string>> mergedMsgQueue_ =
      std::make_unique<std::queue<std::string>>();
  std::mutex mutex_;
  std::condition_variable cv_;
  FILE* debugFile;
  std::atomic<bool> stopThread{false};
};

#endif
