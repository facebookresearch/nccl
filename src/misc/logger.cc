// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "logger.h"
#include "nccl_cvars.h"

#include <stdexcept>
#include "nccl_cvars.h"

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

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

// Initialize static memeber for NcclLogger
std::atomic_flag NcclLogger::singletonInitialized_ = ATOMIC_FLAG_INIT;
std::unique_ptr<NcclLogger> NcclLogger::singleton_{};

void NcclLogger::log(const std::string& msg, FILE* ncclDebugFile) noexcept {
  // There are three cases where singleton_ is nullptr:
  // 1. NCCL_LOGGER_MODE is not async.
  // 2. NCCL_LOGGER_MODE is async but singleton_ haven't initialized.
  // 3. We are exiting the program and singleton_ has already been destroyed.
  // In all three cases, we should not init singleton and write to the file directly.
  if (singleton_ != nullptr) {
    singleton_->enqueueLog(msg);
  } else {
    fwrite(msg.c_str(), 1, msg.size(), ncclDebugFile);
  }
}

void NcclLogger::init(FILE* ncclDebugFile) {
  if (NCCL_LOGGER_MODE == NCCL_LOGGER_MODE::async &&
        !singletonInitialized_.test_and_set()) {
    singleton_ = std::unique_ptr<NcclLogger>(new NcclLogger(ncclDebugFile));
  }
}

NcclLogger::NcclLogger(FILE* ncclDebugFile)
    : mergedMsgQueue_(new std::queue<std::string>()) {
  if (!ncclDebugFile) {
    throw std::runtime_error("Failed to open debug file");
  }
  debugFile = ncclDebugFile;

  writeToFile(
      "NCCL Logger: instantiate the Asynchronous NCCL message logging.\n");
  loggerThread_ = std::thread(&NcclLogger::logThreadFunc, this);
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

void NcclLogger::writeToFile(const std::string& message) {
  fprintf(debugFile, "%s", message.c_str());
}

NcclLogger::~NcclLogger() {
  stop();

  if (loggerThread_.joinable()) {
    loggerThread_.join();
  }
}

void NcclLogger::enqueueLog(const std::string& msg) noexcept{
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      mergedMsgQueue_->push(msg);
    }
    cv_.notify_one();
  } catch (const std::exception& e) {
    // Fixme: make the log conform with the NCCL log format by isolating
    // the logic for formatting logs in debug.cc from the logic of logging
    // logs. Otherwise we will be calling the logger again.
    fprintf(debugFile, "NcclLogger: Encountered exception %s\n", e.what());
  } catch (...) {
    fprintf(debugFile, "NcclLogger: Encountered unknown exception\n");
  }
}

void NcclLogger::logThreadFunc() {
  try {
    while (!stopThread) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !mergedMsgQueue_->empty() || stopThread; });

      std::string batchMsg;
      while (!mergedMsgQueue_->empty()) {
        auto msg = mergedMsgQueue_->front();
        mergedMsgQueue_->pop();
        batchMsg += msg;
      }

      lock.unlock();

      writeToFile(batchMsg);
      fflush(debugFile);
    }
  } catch (const std::exception& e) {
    fprintf(debugFile, "Exception in NCCL logger thread: %s\n", e.what());
  }
}
