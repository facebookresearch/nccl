// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "Logger.h"
#include "EventMgr.h"
#include "ExtUtils.h"
#include "debug.h"
#include "nccl_cvars.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_LOGGER_MODE
   type        : enum
   default     : sync
   choices     : sync, async
   description : |-
     The way to log NCCL messages to stdout or specified by NCCL_DEBUG_FILE.
     sync     - Log NCCL messages synchronously
     async    - Log NCCL messages asynchronously via a background thread
        (for NCCL messages logging file, see also NCCL_DEBUG_FILE)

 - name        : NCCL_COMM_EVENT_LOGGING
   type        : string
   default     : ""
   description : |-
      File location for logging communicator events.
      <local_file>        - Log to a local file. Logs may be interleaved if
multiple Loggers are running scuba:<table_name>  - Log to a scuba table
      pip:<table_name>    - Log to a local file and then upload to scuba
      ""                  - No logging will occur
       (No logging will occur if NCCL_LOGGER_MODE is not async)

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

// Initialize static memeber for NcclLogger
std::unique_ptr<NcclLogger> NcclLogger::singleton_{};
std::atomic<bool> NcclLogger::isInitialized_{false};

void NcclLogger::setDebugLoggingMode(FILE* ncclDebugFile) {
  if (!ncclDebugFile) {
    throw std::runtime_error("Failed to open debug file");
  }

  NcclLogger::init();

  if (ncclDebugFile == stdout) {
    singleton_->loggingInfoMap_[LoggerEventType::DebugEventType]->mode =
        NcclLoggingMode::Stdout;
  } else {
    singleton_->loggingInfoMap_[LoggerEventType::DebugEventType]->mode =
        NcclLoggingMode::File;
  }

  if (singleton_ != nullptr) {
    singleton_->loggingInfoMap_[LoggerEventType::DebugEventType]->fileHandle =
        ncclDebugFile;

    singleton_->writeToFile(
        "NCCL Logger: instantiate the Asynchronous NCCL message logging.\n",
        singleton_->loggingInfoMap_[LoggerEventType::DebugEventType]
            ->fileHandle);
  }
}

void NcclLogger::setEventLoggingMode(
    const std::string& file_path,
    std::unique_ptr<NcclLoggingInfo>& loggingInfo) {
  loggingInfo.reset(new NcclLoggingInfo());

  if (file_path.empty()) {
    return;
  }

  std::string filename;
  if (file_path.substr(0, 6) == "scuba:") {
    std::string table_name = file_path.substr(6);
    loggingInfo->mode = NcclLoggingMode::Scuba;
    loggingInfo->scubaTable = table_name;

  } else if (file_path.substr(0, 4) == "pip:") {
    std::string table_name = file_path.substr(4);
    filename = "/logs/dedicated_log_structured_json.perfpipe_" + table_name +
        "." + getUniqueFileSuffix() + ".scribe";
    loggingInfo->mode = NcclLoggingMode::PipToScuba;
    loggingInfo->scubaTable = table_name;
    loggingInfo->fileName = filename;

  } else {
    filename = file_path;
    loggingInfo->mode = NcclLoggingMode::File;
  }

  if (loggingInfo->mode == NcclLoggingMode::File ||
      loggingInfo->mode == NcclLoggingMode::PipToScuba) {
    loggingInfo->fileHandle = fopen(filename.c_str(), "w");
    if (loggingInfo->fileHandle != nullptr) {
      setbuf(loggingInfo->fileHandle, nullptr); // disable buffering
    }
  }
}

void NcclLogger::log(const std::string& msg, FILE* ncclDebugFile) noexcept {
  // There are three cases where singleton_ is nullptr:
  // 1. NCCL_LOGGER_MODE is not async.
  // 2. NCCL_LOGGER_MODE is async but singleton_ haven't initialized.
  // 3. We are exiting the program and singleton_ has already been destroyed.
  // In all three cases, we should not init singleton and write to the file
  // directly.
  if (singleton_ != nullptr) {
    std::unique_ptr<DebugEvent> event = std::make_unique<DebugEvent>(msg);
    singleton_->enqueueLog(std::move(event));
  } else {
    fwrite(msg.c_str(), 1, msg.size(), ncclDebugFile);
  }
}

void NcclLogger::record(std::unique_ptr<LoggerEvent> event) noexcept {
  if (singleton_ != nullptr) {
    singleton_->enqueueLog(std::move(event));
  }
}

void NcclLogger::recordStart(
    std::unique_ptr<LoggerEvent> event,
    std::string eventName) noexcept {
  if (singleton_ != nullptr) {
    singleton_->startTimeMap_[eventName].push(std::chrono::steady_clock::now());
    singleton_->enqueueLog(std::move(event));
  }
}

void NcclLogger::recordEnd(
    std::unique_ptr<LoggerEvent> event,
    std::string eventName) noexcept {
  if (singleton_ != nullptr) {
    if (singleton_->startTimeMap_[eventName].empty()) {
      event->setTimerDelta(-1);
    } else {
      auto timerBegin = singleton_->startTimeMap_[eventName].top();
      singleton_->startTimeMap_[eventName].pop();
      event->setTimerDelta(
          std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::steady_clock::now() - timerBegin)
              .count() *
          1000);
    }
    singleton_->enqueueLog(std::move(event));
  }
}

void NcclLogger::init() {
  if (!isInitialized_.exchange(true)) {
    if (NCCL_LOGGER_MODE == NCCL_LOGGER_MODE::async) {
      singleton_ = std::unique_ptr<NcclLogger>(new NcclLogger());
    }
  }
}

NcclLogger::NcclLogger() {
  loggerThread_ = std::thread(&NcclLogger::loggerThreadFn, this);
  loggingInfoMap_[LoggerEventType::DebugEventType] =
      std::make_unique<NcclLoggingInfo>();
  loggingInfoMap_[LoggerEventType::CommEventType] =
      std::make_unique<NcclLoggingInfo>();

  setEventLoggingMode(
      NCCL_COMM_EVENT_LOGGING, loggingInfoMap_[LoggerEventType::CommEventType]);
}

void NcclLogger::stop() {
  {
    // Based on documentation, even if the conditional variable is atomic,
    // we still need to lock the mutex to make sure the correct ordering of
    // operations.
    std::lock_guard<std::mutex> lock(mutex_);
    stopThread = true;
  }
  cv_.notify_one();
}

void NcclLogger::writeToFile(const std::string& message, FILE* target) {
  if (target != nullptr) {
    fprintf(target, "%s", message.c_str());
  }
}

NcclLogger::~NcclLogger() {
  stop();

  if (loggerThread_.joinable()) {
    loggerThread_.join();
  }

  for (auto& [loggerType, loggingInfo] : loggingInfoMap_) {
    if (loggerType == LoggerEventType::DebugEventType) {
      continue; // NCCL Debugging file is handled by NCCL main thread
    }

    if (loggingInfo->fileHandle != nullptr &&
        (loggingInfo->mode == NcclLoggingMode::File ||
         loggingInfo->mode == NcclLoggingMode::PipToScuba)) {
      fclose(loggingInfo->fileHandle);
    }
  }
}

void NcclLogger::enqueueLog(std::unique_ptr<LoggerEvent> event) noexcept {
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      mergedMsgQueue_->push(std::move(event));
    }
    cv_.notify_one();
  } catch (const std::exception& e) {
    // Fixme: make the log conform with the NCCL log format by isolating
    // the logic for formatting logs in debug.cc from the logic of logging
    // logs. Otherwise we will be calling the logger again.
    fprintf(
        loggingInfoMap_[LoggerEventType::DebugEventType]->fileHandle,
        "NcclLogger: Encountered exception %s\n",
        e.what());
  } catch (...) {
    fprintf(
        loggingInfoMap_[LoggerEventType::DebugEventType]->fileHandle,
        "NcclLogger: Encountered unknown exception\n");
  }
}

void NcclLogger::loggerThreadFn() {
  loggerThreadFnImpl();
}

void NcclLogger::loggerThreadFnImpl() {
  try {
    while (!stopThread) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !mergedMsgQueue_->empty() || stopThread; });

      std::unordered_map<LoggerEventType, std::string> messages;
      std::deque<std::unique_ptr<LoggerEvent>> scubaOps;

      while (!mergedMsgQueue_->empty()) { // aggressively retrieving events
        auto& event = mergedMsgQueue_->front();
        auto event_type = event->getEventType();
        auto logging_mode = loggingInfoMap_[event_type]->mode;

        if (logging_mode == NcclLoggingMode::Scuba) {
          scubaOps.push_back(std::move(event));
        } else if (
            logging_mode == NcclLoggingMode::Stdout ||
            logging_mode == NcclLoggingMode::PipToScuba ||
            logging_mode == NcclLoggingMode::File) {
          messages[event_type] += event->serialize();
        }

        mergedMsgQueue_->pop();
      }

      lock.unlock();

      // dump messages to file
      for (auto& [event_type, msg] : messages) {
        writeToFile(msg, loggingInfoMap_[event_type]->fileHandle);
        fflush(loggingInfoMap_[event_type]->fileHandle);
      }

      // upload entries to scuba table
      for (auto& event : scubaOps) {
        event->toScuba(loggingInfoMap_[event->getEventType()]->scubaTable);
      }
    }
  } catch (const std::exception& e) {
    fprintf(
        loggingInfoMap_[LoggerEventType::DebugEventType]->fileHandle,
        "Exception in NCCL logger thread: %s\n",
        e.what());
  }
}
