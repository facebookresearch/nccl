#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "comm.h"
#include "mpi.h"
#include <gtest/gtest.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    setenv("NCCL_DEBUG", "WARN", 0);
    // Initialize CVAR so that we can overwrite global variable in each test
    initEnv();
  }

  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class ProxyTraceTest : public ::testing::Test {
 public:
  ProxyTraceTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void runAllReduce(const int nColl, ncclComm_t comm) {
    for (int i = 0; i < nColl; i++) {
      NCCLCHECK_TEST(ncclAllReduce(
          sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    }
  }

 protected:
  int count{32*1024*1024};
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  cudaStream_t stream;
};

TEST_F(ProxyTraceTest, QueryHangSendRecv) {
  int size = 32*1024*1024;

  auto comm = createNcclComm(this->globalRank, this->numRanks, this->localRank);

  if (comm->nNodes < 2) {
    ncclCommDestroy(comm);
    GTEST_SKIP() << "Skipping test since nNodes < 2";
  }

  runAllReduce(100, comm);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Manually cause one rank to not be available.
  // Since we are just checking the failure returned, how exactly it failed
  // doesn't matter to us.
  if (globalRank == 1) {
    ncclCommAbort(comm);
    sleep(20);
  } else {
    sleep(3);
    runAllReduce(1, comm);
    sleep(15);
    ncclResult_t result;
    NCCLCHECK_TEST(ncclCommGetAsyncError(comm, &result));
    EXPECT_EQ(result, ncclRemoteError);
    ncclCommAbort(comm);
  }

}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
