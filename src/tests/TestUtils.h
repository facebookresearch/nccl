// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TESTS_UTILS_H_
#define TESTS_UTILS_H_

/**
 * This file defines common utilities for tests without adding dependency to
 * CUDA nor NCCL
 */
#include <filesystem>
#include <string>
#include <thread>

// Generate temporary file path under system tmp directory to avoid filename
// collision when multiple tests are remote executed in parallel
std::string getTestFilePath(
    std::string prefix = "",
    std::string extension = "") {
  auto tmpPath = std::filesystem::temp_directory_path();
  auto threadHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
  tmpPath /= prefix + std::to_string(threadHash) + extension;
  return tmpPath.string();
}

#endif
