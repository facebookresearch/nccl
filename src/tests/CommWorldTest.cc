// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include "checks.h"
#include "tests_common.cuh"
#include "collectives.h"

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

class CommWorldTest : public ::testing::Test {
 public:
  ncclComm_t comm;
  CommWorldTest() = default;

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

TEST_F(CommWorldTest, commWorldInit) {
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  EXPECT_THAT(NCCL_COMM_WORLD, testing::NotNull());

  int expectedCommSize = 0;
  int expectedCommRank = 0;
  int expectedCommDev = 0;

  NCCLCHECK_TEST(ncclCommCount(comm, &expectedCommSize));
  NCCLCHECK_TEST(ncclCommUserRank(comm, &expectedCommRank));
  NCCLCHECK_TEST(ncclCommCuDevice(comm, &expectedCommDev));

  int worldCommSize = -1;
  int worldCommRank = -1;
  int worldCommDev = -1;

  NCCLCHECK_TEST(ncclCommCount(NCCL_COMM_WORLD, &worldCommSize));
  NCCLCHECK_TEST(ncclCommUserRank(NCCL_COMM_WORLD, &worldCommRank));
  NCCLCHECK_TEST(ncclCommCuDevice(NCCL_COMM_WORLD, &worldCommDev));

  // Check that the comm world is initialized correctly
  EXPECT_EQ(expectedCommSize, worldCommSize);
  EXPECT_EQ(expectedCommRank, worldCommRank);
  EXPECT_EQ(expectedCommDev, worldCommDev);

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);
}

TEST_F(CommWorldTest, multiComms) {
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  ncclComm_t comm2 = createNcclComm(this->globalRank, this->numRanks, this->localRank);

  EXPECT_THAT(NCCL_COMM_WORLD, testing::NotNull());

  int expectedCommSize = 0;
  int expectedCommRank = 0;
  int expectedCommDev = 0;

  NCCLCHECK_TEST(ncclCommCount(comm, &expectedCommSize));
  NCCLCHECK_TEST(ncclCommUserRank(comm, &expectedCommRank));
  NCCLCHECK_TEST(ncclCommCuDevice(comm, &expectedCommDev));

  int worldCommSize = -1;
  int worldCommRank = -1;
  int worldCommDev = -1;

  NCCLCHECK_TEST(ncclCommCount(NCCL_COMM_WORLD, &worldCommSize));
  NCCLCHECK_TEST(ncclCommUserRank(NCCL_COMM_WORLD, &worldCommRank));
  NCCLCHECK_TEST(ncclCommCuDevice(NCCL_COMM_WORLD, &worldCommDev));

  // Check that the comm world is initialized correctly
  EXPECT_EQ(expectedCommSize, worldCommSize);
  EXPECT_EQ(expectedCommRank, worldCommRank);
  EXPECT_EQ(expectedCommDev, worldCommDev);

  NCCLCHECK_TEST(ncclCommDestroy(comm2));

  // NCCL_COMM_WORLD should not be destroyed yet
  EXPECT_THAT(NCCL_COMM_WORLD, testing::NotNull());

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  // NCCL_COMM_WORLD should be destroyed
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);
}

TEST_F(CommWorldTest, commWorldColl) {
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  EXPECT_THAT(NCCL_COMM_WORLD, testing::NotNull());

  void* buf, *testbuf;
  size_t count = 1024;
  ncclDataType_t dt = ncclFloat32;
  cudaStream_t stream = 0;
  srand(time(NULL));
  char initVal = rand();
  float *expectedVals = (float*) malloc(count * ncclTypeSize(dt));
  float *observedVals = (float*) malloc(count * ncclTypeSize(dt));

  CUDACHECKIGNORE(cudaMalloc(&buf, count * ncclTypeSize(dt)));
  CUDACHECKIGNORE(cudaMemset(buf, initVal, count * ncclTypeSize(dt)));

  auto res = ncclAllReduce(buf, buf, count, dt, ncclSum, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaMalloc(&testbuf, count * ncclTypeSize(dt)));
  CUDACHECKIGNORE(cudaMemset(testbuf, initVal, count * ncclTypeSize(dt)));

  res = ncclAllReduce(testbuf, testbuf, count, dt, ncclSum, NCCL_COMM_WORLD, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaMemcpy(expectedVals, buf, count * ncclTypeSize(dt), cudaMemcpyDefault));
  CUDACHECKIGNORE(cudaMemcpy(observedVals, testbuf, count * ncclTypeSize(dt), cudaMemcpyDefault));

  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(expectedVals[i], observedVals[i]);
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  // NCCL_COMM_WORLD should be destroyed
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  CUDACHECKIGNORE(cudaFree(buf));
  CUDACHECKIGNORE(cudaFree(testbuf));
  free(expectedVals);
  free(observedVals);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
