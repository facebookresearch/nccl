// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <memory>
#include "nccl_cvars.h"
#include "tuner.h"

class tunerTest : public ::testing::Test {
 public:
  tunerTest() {
    ncclCvarInit();
  }
};

// Load and close default tuner plugin
TEST_F(tunerTest, loadAndCloseTuner) {
  ncclResult_t res = ncclSuccess;
  ncclTuner_t* tuner = nullptr;
  NCCL_TUNER_PLUGIN = ""; // load mock tuner built with UT

  res = ncclLoadTunerPlugin(&tuner);
  EXPECT_EQ(res, ncclSuccess);
  EXPECT_NE(tuner, nullptr);
  res = ncclCloseTunerPlugin(&tuner);
  EXPECT_EQ(res, ncclSuccess);
}

TEST_F(tunerTest, getCollInfo) {
  ncclResult_t res = ncclSuccess;
  ncclTuner_t* tuner = nullptr;
  NCCL_TUNER_PLUGIN = ""; // load mock tuner built with UT

  res = ncclLoadTunerPlugin(&tuner);
  EXPECT_EQ(res, ncclSuccess);
  EXPECT_NE(tuner, nullptr);

  ncclFunc_t collType = ncclFuncAllReduce;
  size_t nBytes = 256;
  int collNetSupport = 0;
  int nvlsSupport = 0;
  int numPipeOps = 0;
  int algorithm = NCCL_ALGO_UNDEF;
  int protocol = NCCL_PROTO_UNDEF;
  int nChannels = 1;

  res = tuner->getCollInfo(
      collType,
      nBytes,
      collNetSupport,
      nvlsSupport,
      numPipeOps,
      &algorithm,
      &protocol,
      &nChannels);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_EQ(algorithm, NCCL_ALGO_TREE);
  EXPECT_EQ(protocol, NCCL_PROTO_LL);
  EXPECT_EQ(nChannels, 1);

  // Change message size, expect different result
  // see mockTuner.c for the expected result from mock tuner
  nBytes = 1 << 20; // 1MB
  res = tuner->getCollInfo(
      collType,
      nBytes,
      collNetSupport,
      nvlsSupport,
      numPipeOps,
      &algorithm,
      &protocol,
      &nChannels);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_EQ(algorithm, NCCL_ALGO_RING);
  EXPECT_EQ(protocol, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(nChannels, 8);

  res = ncclCloseTunerPlugin(&tuner);
  EXPECT_EQ(res, ncclSuccess);
}

/* this has to be the last test because NCCL won't attemp loading tuner anymore
 * if an invalid plugin is provided */
TEST_F(tunerTest, invalidTunerName) {
  ncclResult_t res = ncclSuccess;
  ncclTuner_t* tuner = nullptr;
  NCCL_TUNER_PLUGIN = "invalid";

  res = ncclLoadTunerPlugin(&tuner);
  EXPECT_EQ(res, ncclSuccess);
  EXPECT_EQ(tuner, nullptr);
}
