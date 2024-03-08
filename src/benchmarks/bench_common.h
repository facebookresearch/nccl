// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef BENCH_COMMON_CH_
#define BENCH_COMMON_CH_

#include <csignal>
#include <cstdio>

#define BENCH_ERR(fmt, ...)            \
  do {                                 \
    fprintf(                           \
        stderr,                        \
        "%s:%d: Benchmark ERROR:" fmt, \
        __FILE__,                      \
        __LINE__,                      \
        ##__VA_ARGS__);                \
  } while (0)

static inline void benchAbortSignalHandler(int signal) {
  if (signal == SIGABRT) {
    BENCH_ERR("SIGABRT received\n");
    std::_Exit(EXIT_FAILURE);
  }
}

static inline void benchAbortSignalSetup(void) {
  // Setup handler
  auto previous_handler = std::signal(SIGABRT, benchAbortSignalHandler);
  if (previous_handler == SIG_ERR) {
    BENCH_ERR("SIGABRT handler setup failed\n");
    std::_Exit(EXIT_FAILURE);
  }
}

#endif
