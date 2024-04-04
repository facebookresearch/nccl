#include <stdlib.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Logger.h"

const static std::string kTestStr = "test1";
constexpr size_t kBufferSize = 4096;

class NcclLoggerTest : public ::testing::Test {
 public:
  NcclLoggerTest() = default;
  void SetUp() override {
    this->testFilePath = getTestFilePath();
    this->logFile = fopen(this->testFilePath.c_str(), "w");
    if (this->logFile != nullptr) {
      setbuf(this->logFile, nullptr); // disable buffering to align with NCCL
    }
  }

  void TearDown() override {
    if (this->logFile != nullptr) {
      fclose(this->logFile);
    }
    std::error_code ec;
    // Use noexcept version of remove to avoid throwing exceptions
    std::filesystem::remove(testFilePath, ec);
  }

  void finishLogging() {
    closeLoggerSingleton();
    closeWriteFile();
  }

  void closeLoggerSingleton() {
    NcclLogger::singleton_.reset();
  }

  bool closeWriteFile() {
    auto res = fclose(this->logFile) == 0;
    this->logFile = nullptr;
    return res;
  }

  std::string getTestFilePath() {
    auto tmpPath = std::filesystem::temp_directory_path();
    auto threadHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto objHash = std::hash<void *>{}(static_cast<void *>(this));
    tmpPath /= "loggertest" + std::to_string(threadHash^objHash);
    return tmpPath.string();
  }

  std::string readLogs() {
    auto testFd = fopen(this->testFilePath.c_str(), "r");
    assert(testFd != nullptr);
    char buffer[kBufferSize];
    auto len = fread(buffer, 1, kBufferSize, testFd);
    return std::string(buffer, len);
  }

 protected:
  FILE* logFile;
  std::string testFilePath;
};

TEST_F(NcclLoggerTest, SyncLogStdout) {
  testing::internal::CaptureStdout();

  NcclLogger::log(kTestStr, stdout);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(NcclLoggerTest, SyncLogSpecifyDebugFile) {
  NcclLogger::log(kTestStr, this->logFile);
  this->finishLogging();

  auto output = this->readLogs();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(NcclLoggerTest, AsyncLogStdout) {
  NcclLogger::init(stdout);

  testing::internal::CaptureStdout();

  NcclLogger::log(kTestStr, stdout);
  // Give async thread a chance to run
  sleep(3);
  this->finishLogging();

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(NcclLoggerTest, AsyncLogSpecifyDebugFile) {
  NcclLogger::init(this->logFile);
  NcclLogger::log(kTestStr, this->logFile);
  // Give async thread a chance to run
  sleep(3);
  this->finishLogging();

  auto output = this->readLogs();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}
