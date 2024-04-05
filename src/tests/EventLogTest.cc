// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <string>
#include "ExtUtils.h"
#include "checks.h"
#include "core.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn on NCCL INFO logging as the event log level
    setenv("NCCL_DEBUG", "INFO", 1);
  }
  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class EventLogTest : public ::testing::Test {
 public:
  EventLogTest() = default;

 protected:
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECKABORT(cudaSetDevice(this->localRank));
  }

  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

#define OMIT_LOG_START \
  { testing::internal::CaptureStdout(); }
#define OMIT_LOG_END \
  { auto ignore = testing::internal::GetCapturedStdout(); }

TEST_F(EventLogTest, CommInit) {
  testing::internal::CaptureStdout();
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(nullptr, comm);

  std::string stdoutContent = testing::internal::GetCapturedStdout();
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr("CommInitRank - Init START before bootstrap"));
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr("CommInitRank - Init bootstrap COMPLETE"));

  uint64_t commHash = 0;
  NCCLCHECK_TEST(ncclCommGetUniqueHash(comm, &commHash));
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr(
          "commHash " + hashToHexStr(commHash) + " CommInitRank - Init START"));
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr(
          "commHash " + hashToHexStr(commHash) +
          " CommInitRank - Init COMPLETE"));

  OMIT_LOG_START;
  NCCLCHECK_TEST(ncclCommDestroy(comm));
  OMIT_LOG_END;
}

TEST_F(EventLogTest, CommSplit) {
  OMIT_LOG_START;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(nullptr, comm);
  OMIT_LOG_END;

  testing::internal::CaptureStdout();
  ncclComm_t newcomm = NCCL_COMM_NULL;
  auto res = ncclSuccess;

  // Split into two groups, one with odd ranks and one with even ranks
  res = ncclCommSplit(
      comm, this->globalRank % 2, this->globalRank, &newcomm, nullptr);
  ASSERT_EQ(res, ncclSuccess);
  EXPECT_NE(newcomm, (ncclComm_t)(NCCL_COMM_NULL));

  std::string stdoutContent = testing::internal::GetCapturedStdout();
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr("CommSplit - Init START before bootstrap"));
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr("CommSplit - Init bootstrap COMPLETE"));

  auto parentRankStr = std::to_string(this->globalRank);
  auto colorStr = std::to_string(this->globalRank % 2);
  auto keyStr = std::to_string(this->globalRank);
  uint64_t commHash = 0;
  NCCLCHECK_TEST(ncclCommGetUniqueHash(newcomm, &commHash));

  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr(
          "rank " + parentRankStr + " color " + colorStr + " key " + keyStr +
          " - CommSplit START"));
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr(
          "commHash " + hashToHexStr(commHash) + " CommSplit - Init START"));
  EXPECT_THAT(
      stdoutContent,
      ::testing::HasSubstr(
          "commHash " + hashToHexStr(commHash) + " CommSplit - Init COMPLETE"));

  OMIT_LOG_START;
  NCCLCHECK_TEST(ncclCommDestroy(newcomm));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
  OMIT_LOG_END;
}

TEST_F(EventLogTest, CommDestroy) {
  OMIT_LOG_START;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(nullptr, comm);
  OMIT_LOG_END;

  testing::internal::CaptureStdout();
  NCCLCHECK_TEST(ncclCommDestroy(comm));

  std::string stdoutContent = testing::internal::GetCapturedStdout();
  EXPECT_THAT(stdoutContent, ::testing::HasSubstr(" - Destroy START"));
  EXPECT_THAT(stdoutContent, ::testing::HasSubstr(" - Destroy COMPLETE"));
}

TEST_F(EventLogTest, CommAbort) {
  OMIT_LOG_START;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(nullptr, comm);
  OMIT_LOG_END;

  testing::internal::CaptureStdout();
  NCCLCHECK_TEST(ncclCommAbort(comm));

  std::string stdoutContent = testing::internal::GetCapturedStdout();
  EXPECT_THAT(stdoutContent, ::testing::HasSubstr(" - Abort START"));
  EXPECT_THAT(stdoutContent, ::testing::HasSubstr(" - Abort COMPLETE"));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
