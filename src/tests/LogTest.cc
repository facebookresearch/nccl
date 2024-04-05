// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include "TestUtils.h"
#include "debug.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

class LogTest : public ::testing::Test {
 public:
  LogTest() {
    ncclCvarInit();
  }

  void SetUp() override {
    // allow to re-initialize logger
    ncclDebugLevel = -1;
    if (ncclDebugFile != stderr && ncclDebugFile != stdout) {
      fclose(ncclDebugFile);
    }
  }

  void TearDown() override {}
};

TEST_F(LogTest, Info) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("INFO"));
  const std::string kTestStr = "Testing INFO";

  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", kTestStr.c_str());

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, LongInfo) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("INFO"));
  std::string kTestStr = "Testing long INFO,";

  // prepare log longer than 1024 chars as statically set previously
  do {
    kTestStr += kTestStr;
  } while (kTestStr.size() < 3000);

  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", kTestStr.c_str());

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, Warn) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("WARN"));
  const std::string kTestStr = "Testing WARN";

  testing::internal::CaptureStdout();
  WARN("%s", kTestStr.c_str());

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, LongWarn) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("WARN"));
  std::string kTestStr = "Testing long WARN,";

  // prepare log longer than 1024 chars as statically set previously
  do {
    kTestStr += kTestStr;
  } while (kTestStr.size() < 3000);

  testing::internal::CaptureStdout();
  WARN("%s", kTestStr.c_str());

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST_F(LogTest, InfoOff) {
  auto envGuard = EnvRAII(NCCL_DEBUG, std::string("WARN"));
  const std::string kTestStr = "Testing INFO when NCCL_DEBUG=WARN";

  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", kTestStr.c_str());

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, Not(testing::HasSubstr(kTestStr)));
}

TEST_F(LogTest, InfoToFile) {
  const std::string kDebugFile = getTestFilePath("nccl_logtest", ".log");
  const std::string kTestStr = "Testing INFO to FILE";
  auto envGuard0 = EnvRAII(NCCL_DEBUG_FILE, kDebugFile);
  auto envGuard1 = EnvRAII(NCCL_DEBUG, std::string("INFO"));

  INFO(NCCL_ALL, "%s", kTestStr.c_str());

  // check file exists and has the expected log
  EXPECT_TRUE(std::filesystem::exists(kDebugFile));
  std::ifstream filein(kDebugFile);
  std::string line;
  std::getline(filein, line);
  EXPECT_THAT(line, testing::HasSubstr(kTestStr));

  std::filesystem::remove(kDebugFile);
}

TEST_F(LogTest, InfoWarnToFile) {
  const std::string kDebugFile = getTestFilePath("nccl_logtest", ".log");
  const std::string kTestInfoStr = "Testing INFO to FILE";
  const std::string kTestWarnStr = "Testing WARN to FILE";

  auto envGuard0 = EnvRAII(NCCL_DEBUG_FILE, kDebugFile);
  auto envGuard1 = EnvRAII(NCCL_DEBUG, std::string("INFO"));

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();

  INFO(NCCL_ALL, "%s", kTestInfoStr.c_str());
  WARN("%s", kTestWarnStr.c_str());

  std::string stdoutContent = testing::internal::GetCapturedStdout();
  std::string stderrContent = testing::internal::GetCapturedStderr();

  // check both INFO and WARN logs are in the file
  EXPECT_TRUE(std::filesystem::exists(kDebugFile));
  std::ifstream filein(kDebugFile);
  std::string line0, line1, line2;
  std::getline(filein, line0);
  std::getline(filein, line1); // account for heading line break in WARN log
  std::getline(filein, line2);
  EXPECT_THAT(line0, testing::HasSubstr(kTestInfoStr));
  EXPECT_THAT(line2, testing::HasSubstr(kTestWarnStr));

  // check INFO is NOT printed to stdout, and WARN is printed to stderr
  EXPECT_THAT(stdoutContent, Not(testing::HasSubstr(kTestInfoStr)));
  EXPECT_THAT(stderrContent, testing::HasSubstr(kTestWarnStr));

  std::filesystem::remove(kDebugFile);
}
