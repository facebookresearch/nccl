#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "checks.h"
#include "logger.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn on NCCL debug logging
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv(
        "NCCL_DEBUG_SUBSYS", "INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV,ALLOC", 1);
  }

  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class LoggerTest : public ::testing::Test {
 public:
  LoggerTest() = default;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    ASSERT_NE(nullptr, comm);

    CUDACHECKABORT(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));

    CUDACHECKIGNORE(cudaDeviceSynchronize());
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  void run() {
    char expectedVal;
    constexpr size_t count = 8192;
    void *sendbuf = nullptr, *recvbuf = nullptr;
    size_t sendBytes, recvBytes;

    sendBytes = count * sizeof(int);
    recvBytes = sendBytes * this->numRanks;

    expectedVal = rand();

    CUDACHECKIGNORE(cudaMalloc(&sendbuf, sendBytes));
    CUDACHECKIGNORE(cudaMalloc(&recvbuf, recvBytes));
    CUDACHECKIGNORE(
        cudaMemset(sendbuf, expectedVal * this->globalRank, sendBytes));
    CUDACHECKIGNORE(cudaMemset(recvbuf, rand(), recvBytes));
    // correct data for in-place allgather
    CUDACHECKIGNORE(cudaMemset(
        (char*)recvbuf + this->globalRank * sendBytes,
        expectedVal * this->globalRank,
        sendBytes));


    auto res =
        ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
    EXPECT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    CUDACHECK_TEST(cudaFree(sendbuf));
    CUDACHECK_TEST(cudaFree(recvbuf));

  }

 protected:


  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(LoggerTest, SyncLogStdout) {
    setenv("NCCL_LOGGER_MODE", "sync", 1);
    unsetenv("NCCL_DEBUG_FILE");
    ncclCvarInit();
    ASSERT_EQ(std::string(getenv("NCCL_LOGGER_MODE")), "sync");
    run();
}

TEST_F(LoggerTest, SyncLogDebugfile) {
    setenv("NCCL_LOGGER_MODE", "sync", 1);
    setenv("NCCL_DEBUG_FILE", "/tmp/nccl_debug_sync.log", 1);
    ncclCvarInit();
    ASSERT_EQ(std::string(getenv("NCCL_LOGGER_MODE")), "sync");
    run();
}

TEST_F(LoggerTest, AsyncLogStdout) {
    setenv("NCCL_LOGGER_MODE", "async", 1);
    unsetenv("NCCL_DEBUG_FILE");
    ncclCvarInit();
    ASSERT_EQ(std::string(getenv("NCCL_LOGGER_MODE")), "async");
    run();
}

TEST_F(LoggerTest, AsyncLogDebugfile) {
    setenv("NCCL_LOGGER_MODE", "async", 1);
    setenv("NCCL_DEBUG_FILE", "/tmp/nccl_debug_async.log", 1);
    ncclCvarInit();
    ASSERT_EQ(std::string(getenv("NCCL_LOGGER_MODE")), "async");
    run();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
