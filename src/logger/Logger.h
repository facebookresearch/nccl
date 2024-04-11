// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_LOGGER_H
#define NCCL_LOGGER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <queue>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include "EventMgr.h"

// Friend class for testing. This class is used to access the internal status of
// NcclLogger.
class NcclLoggerTest;

enum class NcclLoggingMode {
  Stdout,
  Scuba,
  PipToScuba, // Pip to a temporary file that will be uploaded to Scuba.
  File, // Local file.
  None, // No logging.
};

struct NcclLoggingInfo {
  NcclLoggingMode mode = NcclLoggingMode::None;
  std::string scubaTable;
  std::string fileName;
  FILE* fileHandle = nullptr;
};

class NcclLogger {
 public:
  // Friend class for testing
  friend class NcclLoggerTest;

  static void init();

  static void setDebugLoggingMode(FILE* ncclDebugFile);

  static void log(const std::string& msg, FILE* ncclDebugFile) noexcept;

  static void record(std::unique_ptr<LoggerEvent> event) noexcept;
  static void recordStart(
      std::unique_ptr<LoggerEvent> event,
      std::string eventName) noexcept;
  static void recordEnd(
      std::unique_ptr<LoggerEvent> event,
      std::string eventName) noexcept;

  NcclLogger(const NcclLogger&) = delete;
  NcclLogger& operator=(const NcclLogger&) = delete;
  ~NcclLogger();

 private:
  void stop();
  void loggerThreadFn();
  void loggerThreadFnImpl();
  void writeToFile(const std::string& message, FILE* target);
  void enqueueLog(std::unique_ptr<LoggerEvent> event) noexcept;
  static std::string getJobName();
  static int64_t getTimestamp();

  NcclLogger();
  std::string randomStr(const int len);
  void setEventLoggingMode(
      const std::string& file_config,
      std::unique_ptr<NcclLoggingInfo>& loggingInfo);

  static std::unique_ptr<NcclLogger> singleton_;
  static std::atomic<bool> isInitialized_;

  std::thread loggerThread_;

  std::unique_ptr<std::queue<std::unique_ptr<LoggerEvent>>> mergedMsgQueue_ =
      std::make_unique<std::queue<std::unique_ptr<LoggerEvent>>>();

  std::mutex mutex_;
  std::condition_variable cv_;

  // Global configurations for each event and debugging type.
  std::unordered_map<LoggerEventType, std::unique_ptr<NcclLoggingInfo>>
      loggingInfoMap_ = {};

  // Start time of each event for counting event duration.
  std::unordered_map<
      std::string,
      std::stack<std::chrono::time_point<std::chrono::steady_clock>>>
      startTimeMap_ = {};

  std::atomic<bool> stopThread{false};
};

#endif
