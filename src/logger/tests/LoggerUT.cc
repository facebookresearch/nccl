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
#include "TestUtils.h"
#include "nccl_cvars.h"

const static std::string kTestStr = "test1";
constexpr size_t kBufferSize = 4096;

class NcclLoggerTest : public ::testing::Test {
 public:
  NcclLoggerTest() = default;
  void SetUp() override {
    this->testCommEventFilePath = getTestFilePath("commtest", ".log");
    this->testFilePath = getTestFilePath("loggertest", ".log");
    this->logFile = fopen(this->testFilePath.c_str(), "w");
    if (this->logFile != nullptr) {
      setbuf(this->logFile, nullptr); // disable buffering to align with NCCL
    }
  }

  void TearDown() override {
    unsetenv("NCCL_LOGGER_MODE");
    unsetenv("NCCL_COMM_EVENT_LOGGING");

    if (this->logFile != nullptr) {
      fclose(this->logFile);
    }
    std::error_code ec;
    // Use noexcept version of remove to avoid throwing exceptions
    std::filesystem::remove(testFilePath, ec);
    std::filesystem::remove(testCommEventFilePath, ec);
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

  std::string readLogs(std::string filename) {
    if (filename.empty()) {
      return "";
    }

    auto testFd = fopen(filename.c_str(), "r");
    if (testFd == nullptr) {
      return "";
    }

    char buffer[kBufferSize];
    auto len = fread(buffer, 1, kBufferSize, testFd);
    return std::string(buffer, len);
  }

  std::string getDebugTestFilePath() {
    return this->testFilePath;
  }
  std::string getCommTestFilePath() {
    return this->testCommEventFilePath;
  }

 protected:
  FILE* logFile;
  std::string testFilePath;
  std::string testCommEventFilePath;
};

TEST_F(NcclLoggerTest, SyncLogStdout) {
  setenv("NCCL_LOGGER_MODE", "sync", 1);
  ncclCvarInit();

  testing::internal::CaptureStdout();
  NcclLogger::init();
  NcclLogger::log(kTestStr, stdout);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(NcclLoggerTest, SyncLogSpecifyDebugFile) {
  setenv("NCCL_LOGGER_MODE", "sync", 1);
  ncclCvarInit();

  NcclLogger::init();
  NcclLogger::log(kTestStr, this->logFile);
  this->finishLogging();

  auto output = this->readLogs(getDebugTestFilePath());
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(NcclLoggerTest, AsyncLogStdout) {
  setenv("NCCL_LOGGER_MODE", "async", 1);
  ncclCvarInit();

  NcclLogger::init();
  NcclLogger::setDebugLoggingMode(stdout);

  testing::internal::CaptureStdout();

  NcclLogger::log(kTestStr, stdout);
  // Give async thread a chance to run
  sleep(3);
  this->finishLogging();

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(NcclLoggerTest, AsyncLogSpecifyDebugFile) {
  setenv("NCCL_LOGGER_MODE", "async", 1);
  ncclCvarInit();

  NcclLogger::init();
  NcclLogger::setDebugLoggingMode(this->logFile);
  NcclLogger::log(kTestStr, this->logFile);
  // Give async thread a chance to run
  sleep(3);
  this->finishLogging();

  auto output = this->readLogs(getDebugTestFilePath());
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

//
//  Event logging tests
//

TEST_F(NcclLoggerTest, EventSyncStdout) {
  // no stdout event logging
  setenv("NCCL_LOGGER_MODE", "sync", 1);
  ncclCvarInit();

  testing::internal::CaptureStdout();
  NcclLogger::init();
  unsigned long long commId = 12881726743803089884ULL;
  uint64_t commHash = 17952292033056090124ULL;
  NcclLogger::record(std::make_unique<CommEvent>(
      commId, commHash, 1, 3, "CommInit START", "CommInitRank"));

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::IsEmpty());
}

TEST_F(NcclLoggerTest, EventAsyncStdout) {
  // no stdout event logging
  setenv("NCCL_LOGGER_MODE", "async", 1);
  ncclCvarInit();

  testing::internal::CaptureStdout();
  NcclLogger::init();
  unsigned long long commId = 12881726743803089884ULL;
  uint64_t commHash = 17952292033056090124ULL;
  NcclLogger::record(std::make_unique<CommEvent>(
      commId, commHash, 1, 3, "CommInit START", "CommInitRank"));

  // Give async thread a chance to run
  sleep(3);
  this->finishLogging();

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::IsEmpty());
}

TEST_F(NcclLoggerTest, CommSyncEventFile) {
  // no event logging without logging thread
  setenv("NCCL_LOGGER_MODE", "sync", 1);
  setenv("NCCL_COMM_EVENT_LOGGING", getCommTestFilePath().c_str(), 1);
  ncclCvarInit();

  NcclLogger::init();
  unsigned long long commId = 12881726743803089884ULL;
  uint64_t commHash = 17952292033056090124ULL;
  NcclLogger::record(std::make_unique<CommEvent>(
      commId, commHash, 1, 3, "CommInit START", "CommInitRank"));
  this->finishLogging();

  auto output = this->readLogs(this->testCommEventFilePath);
  EXPECT_THAT(output, testing::IsEmpty());
}

TEST_F(NcclLoggerTest, CommAsyncEventFile) {
  setenv("NCCL_LOGGER_MODE", "async", 1);
  setenv("NCCL_COMM_EVENT_LOGGING", getCommTestFilePath().c_str(), 1);

  ncclCvarInit();

  NcclLogger::init();
  unsigned long long commId = 12881726743803089884ULL;
  uint64_t commHash = 17952292033056090124ULL;
  NcclLogger::record(std::make_unique<CommEvent>(
      commId, commHash, 1, 3, "Init START", "CommInitRank"));

  // Give async thread a chance to run
  sleep(3);
  this->finishLogging();

  auto output = this->readLogs(getCommTestFilePath());

  // FIXME: Ideally, we should check exactly the output string or if it follows the
  // scuba/JSON format But doing so without lib support is too complex to be in the scope
  EXPECT_THAT(output, testing::HasSubstr("Init START"));
}
