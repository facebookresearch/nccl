// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include "checks.h"
#include "tests_common.cuh"
#include "core.h"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn off NCCL debug logging, allow user to turn on via command line
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class ncclWinTest : public ::testing::Test {
 public:
  ncclComm_t comm;
  ncclWinTest() = default;

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

TEST_F(ncclWinTest, winCreation) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  ncclWin_t win = nullptr;
  size_t sizeBytes = 4096;
  auto res = ncclWinAllocShared(sizeBytes, comm, &win);
  EXPECT_EQ(res, ncclSuccess);

  EXPECT_THAT(win, testing::NotNull());

  for (int peer = 0; peer < this->numRanks; ++peer) {
    void* remoteAddr = nullptr;
    res = ncclWinSharedQuery(peer, comm, win, &remoteAddr);
    EXPECT_EQ(res, ncclSuccess);
    EXPECT_THAT(remoteAddr, testing::NotNull());
  }

  res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}


TEST_F(ncclWinTest, getRemoteData) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  cudaStream_t stream = 0;
  ncclWin_t win = nullptr;
  size_t sizeBytes = 4096;
  auto res = ncclWinAllocShared(sizeBytes, comm, &win);
  EXPECT_EQ(res, ncclSuccess);

  void* localWinAddr = nullptr;
  ncclWinSharedQuery(this->globalRank, comm, win, &localWinAddr);
  EXPECT_THAT(localWinAddr, testing::NotNull());

  std::vector<char> expectedVals(sizeBytes, 0);
  for (int i = 0; i < sizeBytes; ++i) {
    expectedVals[i] = this->globalRank * sizeBytes + i;
  }
  CUDACHECKIGNORE(cudaMemcpy(localWinAddr, expectedVals.data(), sizeBytes, cudaMemcpyDefault));
  CUDACHECKIGNORE(cudaDeviceSynchronize());

  // simple Allreduce as barrier before get data from other ranks
  void *buf;
  CUDACHECKIGNORE(cudaMalloc(&buf, sizeof(char)));
  NCCLCHECK_TEST(ncclAllReduce(buf, buf, 1, ncclChar, ncclSum, comm, stream));
  CUDACHECKIGNORE(cudaFree(buf));

  void* remoteData_host = malloc(sizeBytes);
  srand(time(NULL));
  for (int peer = 0; peer < this->numRanks; ++peer) {
    if (peer != this->globalRank) {
        memset(remoteData_host, rand(), sizeBytes);
        void* remoteAddr = nullptr;
        res = ncclWinSharedQuery(peer, comm, win, &remoteAddr);
        EXPECT_EQ(res, ncclSuccess);
        EXPECT_THAT(remoteAddr, testing::NotNull());

        CUDACHECKIGNORE(cudaMemcpy(remoteData_host, remoteAddr, sizeBytes, cudaMemcpyDefault));
        for (int i = 0; i < sizeBytes; ++i) {
          EXPECT_EQ(((char*)remoteData_host)[i], (char)(peer * sizeBytes + i));
        }
    }
  }
  free(remoteData_host);

  res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
