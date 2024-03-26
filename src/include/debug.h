/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>
#include <chrono>
#include <mutex>
#include <type_traits>
#include <vector>

#include <limits.h>
#include <string.h>
#include <pthread.h>

#include <sstream>
#include <iomanip>
#include <unordered_map>

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern std::mutex socketMapMutex;
extern std::unordered_map<std::string, std::string> socketIPv6ToHostname;

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern pthread_mutex_t ncclDebugLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen, const char delim);

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) __attribute__ ((format (printf, 5, 6)));

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define TRACE_CALL(...) ncclDebugLog(NCCL_LOG_TRACE, NCCL_CALL, __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::steady_clock::time_point ncclEpoch;
#else
#define TRACE(...)
#endif

void ncclSetThreadName(pthread_t thread, const char *fmt, ...);
// Allow thread itsef to set its own name for logging purpose
void ncclSetMyThreadLoggingName(const char *fmt, ...);

#define NCCL_NAMED_THREAD_START(threadName)       \
  do {                                            \
    ncclSetMyThreadLoggingName(threadName);       \
    INFO(                                         \
        NCCL_INIT,                                \
        "[NCCL THREAD] Starting %s thread at %s", \
        threadName,                               \
        __func__);                                \
  } while (0);

static inline std::string getTime(void) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    now.time_since_epoch()) %
      1000000;

  std::stringstream timeSs;
  timeSs << std::put_time(std::localtime(&now_c), "%FT%T.") << std::setfill('0')
         << std::setw(6) << now_us.count();
  return timeSs.str();
}

#endif
