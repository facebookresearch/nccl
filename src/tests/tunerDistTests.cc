// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <memory>
#include "comm.h"
#include "nccl_cvars.h"
#include "tuner.h"

class tunerDistTest : public ::testing::Test {
 public:
  tunerDistTest() = default;
};

TEST_F(tunerDistTest, init) {
  setenv("NCCL_NET_PLUGIN", "none", 1);
  ncclComm* comm = nullptr;
  ncclCommInitAll(&comm, 1, nullptr);
  EXPECT_NE(comm, nullptr);

  // check if CVAR is set correctly
  EXPECT_EQ(NCCL_NET_PLUGIN, "mock");
  // check if the environment variable is updated correctly
  EXPECT_STREQ(getenv("NCCL_NET_PLUGIN"), "mock");

  ncclCommDestroy(comm);
}
