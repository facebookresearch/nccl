// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#ifdef ENABLE_FB_INTERNAL
#include "internal.h"
#else

// define wrapper of internal upload API and always return false
#define ncclFbLogSample(args...) \
  do {                        \
  } while (0)

#define ncclIsFbPath(path) (false)
#define ncclFbUpload(args...) \
  do {                        \
  } while (0)

#define COLLTRACE_IO_FB_DURING_RUN(result, rank_) \
  do {                                            \
  } while (0)
#endif

#ifdef ENABLE_IN_TRAINER_TUNE
#include "trainer.h"
#else
#define ncclFbGetTrainerIteration() (0)
#endif
