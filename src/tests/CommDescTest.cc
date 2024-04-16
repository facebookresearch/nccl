// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// #include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cstdlib>
#include "checks.h"
#include "comm.h"
#include "core.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    setenv("NCCL_DEBUG", "INFO", 0);
  }
  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class commDescTest : public ::testing::Test {
 public:
  commDescTest() = default;

 protected:
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(commDescTest, getUndefinedCommDesc) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(nullptr, comm);

  EXPECT_EQ(comm->config.commDesc, "undefined");

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(commDescTest, getDefinedCommDesc) {
  ncclUniqueId ncclId;
  if (globalRank == 0)
    NCCLCHECK_TEST(ncclGetUniqueId(&ncclId));
  MPICHECK_TEST(
      MPI_Bcast((void*)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));
  CUDACHECK_TEST(cudaSetDevice(this->localRank));

  ncclComm_t comm;
  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  inputConfig.commDesc = "test_description";

  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &comm, numRanks, ncclId, globalRank, &inputConfig));
  ASSERT_NE(nullptr, comm);

  EXPECT_NE(comm->config.commDesc, "undefined");
  EXPECT_EQ(comm->config.commDesc, inputConfig.commDesc);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
