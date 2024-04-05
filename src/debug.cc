/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <stdlib.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include "param.h"
#include "nccl_cvars.h"
#include "Logger.h"
#include <sstream>

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_SET_THREAD_NAME
   type        : int64_t
   default     : 0
   description : |-
     Change the name of NCCL threads to ease debugging and analysis.
     For more information:
     https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-set-thread-name

 - name        : NCCL_DEBUG
   type        : string
   default     : ""
   description : |-
     The NCCL_DEBUG variable controls the debug information that is
     displayed from NCCL. This variable is commonly used for
     debugging. For more information:
     https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug

 - name        : NCCL_DEBUG_SUBSYS
   type        : string
   default     : ""
   description : |-
     The NCCL_DEBUG_SUBSYS variable allows the user to filter the
     NCCL_DEBUG=INFO output based on subsystems. A comma separated
     list of the subsystems to include in the NCCL debug log traces.
     For more information:
     https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug-subsys

 - name        : NCCL_DEBUG_FILE
   type        : string
   default     : ""
   description : |-
     The NCCL_DEBUG_FILE variable directs the NCCL debug logging
     output to a file. The filename format can be set to
     filename.%h.%p where %h is replaced with the hostname and %p is
     replaced with the process PID. This does not accept the "~"
     character as part of the path, please convert to a relative or
     absolute path first. For more information:
     https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug-file

- name        : NCCL_WARN_ENABLE_DEBUG_INFO
   type        : int64_t
   default     : 0
   description : |-
    Hidden variable. No description provided.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

std::mutex socketMapMutex{};
std::unordered_map<std::string, std::string> socketIPv6ToHostname{};
int ncclDebugLevel = -1;
static int pid = -1;
static char hostname[1024];
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
uint64_t ncclDebugMask = NCCL_INIT|NCCL_ENV; // Default debug sub-system mask is INIT and ENV
FILE *ncclDebugFile = stdout;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;
std::chrono::steady_clock::time_point ncclEpoch;

static __thread int tid = -1;
static __thread char myThreadName[NCCL_THREAD_NAMELEN];

void ncclDebugInit() {
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclDebugLevel != -1) { pthread_mutex_unlock(&ncclDebugLock); return; }
  const char* nccl_debug = NCCL_DEBUG.c_str();
  int tempNcclDebugLevel = -1;
  if (NCCL_DEBUG.empty()) {
    tempNcclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    tempNcclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    tempNcclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    tempNcclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    tempNcclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    tempNcclDebugLevel = NCCL_LOG_TRACE;
  }

  /* Parse the NCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  const char* ncclDebugSubsysEnv = NCCL_DEBUG_SUBSYS.c_str();
  if (!NCCL_DEBUG_SUBSYS.empty()) {
    int invert = 0;
    if (ncclDebugSubsysEnv[0] == '^') { invert = 1; ncclDebugSubsysEnv++; }
    ncclDebugMask = invert ? ~0ULL : 0ULL;
    char *ncclDebugSubsys = strdup(ncclDebugSubsysEnv);
    char *subsys = strtok(ncclDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = NCCL_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = NCCL_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = NCCL_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = NCCL_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = NCCL_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = NCCL_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = NCCL_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = NCCL_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = NCCL_ALLOC;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = NCCL_CALL;
      } else if (strcasecmp(subsys, "PROXY") == 0) {
        mask = NCCL_PROXY;
      } else if (strcasecmp(subsys, "NVLS") == 0) {
        mask = NCCL_NVLS;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) ncclDebugMask &= ~mask; else ncclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(ncclDebugSubsys);
  }

  // Cache pid and hostname
  getHostName(hostname, 1024, '.');
  pid = getpid();

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = NCCL_DEBUG_FILE.c_str();
  if (tempNcclDebugLevel > NCCL_LOG_VERSION && !NCCL_DEBUG_FILE.empty()) {
    int c = 0;
    char debugFn[PATH_MAX+1] = "";
    char *dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && c < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c-1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", pid);
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = ncclDebugFileEnv[c-1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != nullptr) {
        setbuf(file, nullptr); // disable buffering
        ncclDebugFile = file;
      }
    }
  }

  ncclEpoch = std::chrono::steady_clock::now();
  if (NCCL_LOGGER_MODE == NCCL_LOGGER_MODE::async){
    NcclLogger::init(ncclDebugFile);
  }
  __atomic_store_n(&ncclDebugLevel, tempNcclDebugLevel, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&ncclDebugLock);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (__atomic_load_n(&ncclDebugLevel, __ATOMIC_ACQUIRE) == -1) ncclDebugInit();
  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) { level = NCCL_LOG_INFO; flags = ncclDebugNoWarn; }

  size_t logLen = 0;
  va_list vargs;
  va_start(vargs, fmt);
  logLen += std::vsnprintf(nullptr, 0, fmt, vargs);
  va_end(vargs);

  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN) {
    pthread_mutex_lock(&ncclDebugLock);
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vargs);
    va_end(vargs);
    pthread_mutex_unlock(&ncclDebugLock);
  }
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) return;

  if (tid == -1) {
    tid = syscall(SYS_gettid);
    // All NCCL internal threads should already labeled via ncclSetMyThreadLoggingName.
    // Thus, remaining threads are from users.
    strcpy(myThreadName, "main");
  }

  int cudaDev;
  if (!(level == NCCL_LOG_TRACE && flags == NCCL_CALL)) {
    cudaGetDevice(&cudaDev);
  }

  std::stringstream logStream;
  if (level == NCCL_LOG_WARN) {
    // \n%s %s:%d:%d [%d][%s] %s:%d NCCL WARN
    logStream << "\n" << getTime().c_str() << " " << hostname << ":" << pid << ":"
              << tid << " [" << cudaDev << "][" << myThreadName << "]"
              << filefunc << ":" << line << " NCCL WARN ";
  } else if (level == NCCL_LOG_INFO) {
    // %s %s:%d:%d [%d][%s] NCCL INFO
    logStream << getTime().c_str() << " " << hostname << ":" << pid << ":"
              << tid << " [" << cudaDev << "][" << myThreadName << "]"
              << " NCCL INFO ";
  } else if (level == NCCL_LOG_TRACE && flags == NCCL_CALL) {
    // "%s %s:%d:%d [%s] NCCL CALL "
    logStream << getTime().c_str() << " " << hostname << ":" << pid << ":"
              << tid << " [" << cudaDev << "][" << myThreadName << "]"
              << " NCCL CALL ";
  } else if (level == NCCL_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
    // "%s %s:%d:%d [%d][%s] %f %s:%d NCCL TRACE "
    logStream << getTime().c_str() << " " << hostname << ":" << pid << ":"
              << tid << " [" << cudaDev << "][" << myThreadName << "] "
              << timestamp << " " << filefunc << ":" << line << " NCCL TRACE ";
  }

  if (logStream.str().size() > 0) {
    std::vector<char> buffer(logLen + 1); // +1 for null terminator
    va_list vargs;
    va_start(vargs, fmt);
    // vsnprintf copy at most buf_size - 1 characters
    std::vsnprintf(buffer.data(), buffer.size(), fmt, vargs);
    va_end(vargs);
    logStream << buffer.data() << "\n";

    auto logStr = logStream.str();
    // logging to specified stdout/stderr/file
    NcclLogger::log(logStr, ncclDebugFile);

    // also print to stderr if we're logging into file
    if (ncclDebugFile != stdout && ncclDebugFile != stderr &&
        level == NCCL_LOG_WARN) {
      fprintf(stderr, "%s", logStr.c_str());
      fflush(stderr);
    }
  }
}

void ncclSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (NCCL_SET_THREAD_NAME != 1) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
}

void ncclSetMyThreadLoggingName(const char *fmt, ...) {
  // Update tid so that ncclDebugLog won't re-query it via pthread_getname_np
  tid = syscall(SYS_gettid);
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(myThreadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
}
