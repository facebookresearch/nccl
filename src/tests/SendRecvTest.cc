// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "Ctran.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

static bool VERBOSE = true;

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

class SendRecvTest : public ::testing::Test {
 public:
  SendRecvTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              this->globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

  void prepareBufs(const size_t count, bool registFlag = false) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));

    int expectedVal = comm->rank * 100 + 1;
    assignChunkValue(sendBuf, count, expectedVal);
    assignChunkValue(recvBuf, count, -1);

    if (registFlag) {
      NCCLCHECK_TEST(
          ncclCommRegister(comm, sendBuf, count * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(
          ncclCommRegister(comm, recvBuf, count * sizeof(int), &recvHandle));
    }
  }

  void checkResults(const int sendRank, const size_t count) {
    int expectedVal = sendRank * 100 + 1;
    int errs = checkChunkValue(recvBuf, count, expectedVal);
    EXPECT_EQ(errs, 0) << "Rank " << this->globalRank
                       << " checked result from rank " << sendRank << " at "
                       << recvBuf << " with " << errs << " errors";
  }

  void releaseBufs(bool registFlag = false) {
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void runSend(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    int sendRank, recvRank;
    prepareBufs(count, true);

    // only odd ranks send, and even ranks receive
    if (comm->rank % 2) {
      sendRank = comm->rank;
      recvRank = (comm->rank + 1) % comm->nRanks;
    } else {
      sendRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;
      recvRank = comm->rank;
    }

    if (VERBOSE) {
      printf(
          "Rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);
    }

    for (int x = 0; x < 5; x++) {
      if (comm->rank == sendRank) {
        NCCLCHECK_TEST(
            ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
      } else if (comm->rank == recvRank) {
        NCCLCHECK_TEST(
            ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
      }
      auto opCount = comm->opCount;
      EXPECT_EQ(opCount, x + 1);
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    if (comm->rank == recvRank) {
      checkResults(sendRank, commCount);
    }
    releaseBufs(true);
  }

  void runGroupedSend(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    int sendRank, recvRank;
    prepareBufs(count, true);

    // only odd ranks send, and even ranks receive
    if (comm->rank % 2) {
      sendRank = comm->rank;
      recvRank = (comm->rank + 1) % comm->nRanks;
    } else {
      sendRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;
      recvRank = comm->rank;
    }

    if (VERBOSE) {
      printf(
          "Rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);
    }

    if (comm->rank == sendRank) {
      ncclGroupStart();
      for (int x = 0; x < 5; x++) {
        NCCLCHECK_TEST(
            ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
      }
      ncclGroupEnd();
    } else if (comm->rank == recvRank) {
      ncclGroupStart();
      for (int x = 0; x < 5; x++) {
        NCCLCHECK_TEST(
            ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
      }
      ncclGroupEnd();
    }

    // expect all grouped ops to be counted as one
    auto opCount = comm->opCount;
    EXPECT_EQ(opCount, 1);

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    if (comm->rank == recvRank) {
      checkResults(sendRank, commCount);
    }
    releaseBufs(true);
  }

  void runGroupedSendRecv(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    int sendRank, recvRank;
    prepareBufs(count, true);

    // every rank sends to the next and receives from previous
    sendRank = (comm->rank + 1) % comm->nRanks;
    recvRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;

    if (VERBOSE) {
      printf(
          "Rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);
    }

    for (int x = 0; x < 5; x++) {
      ncclGroupStart();
      NCCLCHECK_TEST(
          ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
      NCCLCHECK_TEST(
          ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
      ncclGroupEnd();
      // expect grouped sendrecv counted as one
      auto opCount = comm->opCount;
      EXPECT_EQ(opCount, x + 1);
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    checkResults(sendRank, commCount);
    releaseBufs(true);
  }

  void runSendRecvSelf(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    int sendRank, recvRank;
    prepareBufs(count, true);

    for (int x = 0; x < 5; x++) {
      ncclGroupStart();
      NCCLCHECK_TEST(
          ncclSend(sendBuf, commCount, ncclInt, comm->rank, comm, stream));
      NCCLCHECK_TEST(
          ncclRecv(recvBuf, commCount, ncclInt, comm->rank, comm, stream));
      ncclGroupEnd();
      // expect grouped sendrecv counted as one
      auto opCount = comm->opCount;
      EXPECT_EQ(opCount, x + 1);
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    checkResults(comm->rank, commCount);
    releaseBufs(true);
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;

  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};
};

TEST_F(SendRecvTest, Default) {
  runSend();
}

TEST_F(SendRecvTest, DefaultGroupdSend) {
  runGroupedSend();
}

TEST_F(SendRecvTest, DefaultGroupedSendRecv) {
  runGroupedSendRecv();
}

TEST_F(SendRecvTest, DefaultSendRecvSelf) {
  runSendRecvSelf();
}

TEST_F(SendRecvTest, Ctran) {
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::ctran;
  runSend();
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO_DEFAULT;
}

TEST_F(SendRecvTest, CtranGroupedSend) {
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::ctran;
  runGroupedSend();
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO_DEFAULT;
}

TEST_F(SendRecvTest, CtranGroupedSendRecv) {
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::ctran;
  runGroupedSendRecv();
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO_DEFAULT;
}

TEST_F(SendRecvTest, CtranSendRecvSelf) {
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::ctran;
  runSendRecvSelf();
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO_DEFAULT;
}


int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
