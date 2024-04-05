// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <unordered_map>
#include "comm.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class CommGetUniqueHashTest : public ::testing::Test {
 public:
  CommGetUniqueHashTest() = default;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CommGetUniqueHashTest, DefaultComm) {
  auto res = ncclSuccess;

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  uint64_t commHash = 0;
  res = ncclCommGetUniqueHash(comm, &commHash);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(commHash, comm->commHash);

  // check all ranks have the same commHash
  uint64_t commHashMin = 0, commHashMax = 0;
  MPI_Allreduce(
      &commHash, &commHashMin, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(
      &commHash, &commHashMax, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  EXPECT_TRUE(commHashMin == commHashMax && commHashMax == commHash);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CommGetUniqueHashTest, ParentChildComm) {
  auto res = ncclSuccess;

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  // Split into two groups, one with odd ranks and one with even ranks
  ncclComm_t childComm = NCCL_COMM_NULL;
  NCCLCHECK_TEST(ncclCommSplit(
      comm, this->globalRank % 2, this->globalRank, &childComm, nullptr));
  EXPECT_NE(childComm, (ncclComm_t)(NCCL_COMM_NULL));

  uint64_t commHash = 0, childCommHash = 0;
  res = ncclCommGetUniqueHash(comm, &commHash);
  ASSERT_EQ(res, ncclSuccess);

  res = ncclCommGetUniqueHash(childComm, &childCommHash);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(childCommHash, childComm->commHash);
  EXPECT_NE(childCommHash, commHash);

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CommGetUniqueHashTest, InvalidComm) {
  auto res = ncclSuccess;

  ncclComm_t comm = NCCL_COMM_NULL;
  uint64_t commHash = 0;
  res = ncclCommGetUniqueHash(comm, &commHash);
  ASSERT_EQ(res, ncclInvalidArgument);
}

TEST_F(CommGetUniqueHashTest, DISABLED_TwoChildCommsSameColor) {
  auto res = ncclSuccess;

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  // Make two child comms from commSplit with same color, compare commHash
  // between them
  ncclComm_t childComms[2] = {NCCL_COMM_NULL, NCCL_COMM_NULL};
  for (int i = 0; i < 2; i++) {
    NCCLCHECK_TEST(ncclCommSplit(
        comm, this->globalRank % 2, this->globalRank, &childComms[i], nullptr));
    EXPECT_NE(childComms[i], (ncclComm_t)(NCCL_COMM_NULL));
  }

  uint64_t childCommHashs[2] = {0, 0};
  for (int i = 0; i < 2; i++) {
    res = ncclCommGetUniqueHash(childComms[i], &childCommHashs[i]);
    ASSERT_EQ(res, ncclSuccess);

    EXPECT_EQ(childCommHashs[i], childComms[i]->commHash);
  }

  EXPECT_NE(childCommHashs[0], childCommHashs[1]);

  for (int i = 0; i < 2; i++) {
    NCCLCHECK_TEST(ncclCommDestroy(childComms[i]));
  }
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
