// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <unistd.h>
#include "Ctran.h"
#include "checks.h"
#include "tests_common.cuh"

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

class CtranTest : public ::testing::Test {
 public:
  CtranTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    srand(time(NULL));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CtranTest, sendRecv) {
  auto res = ncclSuccess;
  // test various size and various num of max QP, intensionally make some sizes
  // not aligned
  std::vector<size_t> counts = {4096, 65536, 2097155, 1073741819, 2147483648};
  std::vector<int> numMaxQps = {1, 4, 8, 16};
  const size_t pageSize = getpagesize();
  ncclDataType_t dt = ncclFloat32;

  for (auto numMaxQp : numMaxQps) {
    NCCL_CTRAN_IB_MAX_QPS = numMaxQp;

    ncclComm_t comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    ASSERT_NE(nullptr, comm);
    ASSERT_NE(nullptr, comm->ctran);

    for (auto count : counts) {
      // always allocate buffer in page size
      size_t bufSize = ((count * ncclTypeSize(dt)) / pageSize + 1) * pageSize;
      size_t sendSize = count * ncclTypeSize(dt);
      int sendRank = 0;
      char* buf;
      cudaStream_t stream = 0;
      void* hdl;
      CUDACHECKIGNORE(cudaMalloc(&buf, bufSize));

      NCCLCHECK_TEST(ncclCommRegister(comm, buf, bufSize, &hdl));

      if (this->globalRank == sendRank) {
        printf(
            "Rank %d sendRank %d send to others with count %ld numMaxQP %d\n",
            comm->rank,
            sendRank,
            count,
            numMaxQp);

        CUDACHECKIGNORE(cudaMemset(buf, 1, bufSize));
        for (int i = 0; i < this->numRanks; ++i) {
          if (i != this->globalRank) {
            res = ctranSend(buf, count, dt, i, comm, stream);
            EXPECT_EQ(res, ncclSuccess);
          }
        }
      } else {
        CUDACHECKIGNORE(cudaMemset(buf, rand(), bufSize));
        res = ctranRecv(buf, count, dt, sendRank, comm, stream);
        EXPECT_EQ(res, ncclSuccess);
      }

      res = ctranGroupEndHook();
      EXPECT_EQ(res, ncclSuccess);
      CUDACHECKIGNORE(cudaStreamSynchronize(stream));

      // First deregister buffer to catch potential 'remote access error' caused
      // by incomplete ctranSend when ctranRecv has returned incorrectly.
      // Delaying it after check can lead to false positive since ctranSend may
      // eventually complete.
      NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));

      if (this->globalRank != sendRank) {
        std::vector<char> observedVals(sendSize);
        CUDACHECKIGNORE(
            cudaMemcpy(observedVals.data(), buf, sendSize, cudaMemcpyDefault));
        int errs = checkChunkValue(buf, sendSize, (char)1);
        EXPECT_EQ(errs, 0);
      }

      CUDACHECKIGNORE(cudaFree(buf));
    }
    NCCLCHECK_TEST(ncclCommDestroy(comm));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
