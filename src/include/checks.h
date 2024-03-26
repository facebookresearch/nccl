/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"
#include <assert.h>

// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(false)

#define CUDACHECKGOTO(cmd, RES, label) do {                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        RES = ncclUnhandledCudaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd) do {  \
    cudaError_t err = cmd;         \
    if( err != cudaSuccess ) {     \
        WARN("%s:%s:%d Cuda failure '%s'", __FILE__, __func__, __LINE__, cudaGetErrorString(err)); \
        (void) cudaGetLastError(); \
    }                              \
} while(false)

// Use of abort should be aware of potential memory leak risk
// and place a signal handler to catch it and trigger termination processing
#define CUDACHECKABORT(cmd)                               \
  do {                                                    \
    cudaError_t err = cmd;                                \
    if (err != cudaSuccess) {                             \
      WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
      abort();                                            \
    }                                                     \
  } while (false)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    WARN("Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, RES, label) do { \
  if ((statement) == -1) {    \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0);

#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0);

#define NEQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0);

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0);

#define EQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0);

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, ncclGetErrorString(RES));    \
    return RES; \
  } \
} while (0);

#define NCCLCHECKGOTO(call, RES, label) do { \
  RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, ncclGetErrorString(RES));    \
    goto label; \
  } \
} while (0);

// Report failure but clear error and continue
#define NCCLCHECKIGNORE(call)                                 \
  do {                                                        \
    ncclResult_t RES = call;                                  \
    if (RES != ncclSuccess && RES != ncclInProgress) {        \
      WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, ncclGetErrorString(RES)); \
    }                                                         \
  } while (0)

// Use of abort should be aware of potential memory leak risk
// and place a signal handler to catch it and trigger termination processing
#define NCCLCHECKABORT(call)                           \
  do {                                                 \
    ncclResult_t RES = call;                           \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, ncclGetErrorString(RES));    \
      abort();                                         \
    }                                                  \
  } while (0)

#define NCCLWAIT(call, cond, abortFlagPtr) do {         \
  volatile uint32_t* tmpAbortFlag = (abortFlagPtr);     \
  ncclResult_t RES = call;                \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, ncclGetErrorString(RES));    \
    return ncclInternalError;             \
  }                                       \
  if (tmpAbortFlag) NEQCHECK(*tmpAbortFlag, 0); \
} while (!(cond));

#define NCCLWAITGOTO(call, cond, abortFlagPtr, RES, label) do { \
  volatile uint32_t* tmpAbortFlag = (abortFlagPtr);             \
  RES = call;                             \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) WARN("%s:%s:%d -> %d (%s)", __FILE__, __func__, __LINE__, RES, ncclGetErrorString(RES));    \
    goto label;                           \
  }                                       \
  if (tmpAbortFlag) NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label); \
} while (!(cond));

#define NCCLCHECKTHREAD(a, args) do { \
  if (((args)->ret = (a)) != ncclSuccess && (args)->ret != ncclInProgress) { \
    WARN("%s:%s:%d -> %d [Async thread]", __FILE__, __func__, __LINE__, (args)->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    WARN("%s:%s:%d -> %d [Async thread]", __FILE__, __func__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

#define NCCLARGCHECK(statement, ...) \
  do {                               \
    if (!(statement)) {              \
      WARN(__VA_ARGS__);             \
      return ncclInvalidArgument;    \
    }                                \
  } while (0);

#endif
