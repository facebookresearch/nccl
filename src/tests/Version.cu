// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <gtest/gtest.h>
#include <nccl.h>

TEST(Version, Code) {
  EXPECT_EQ(NCCL_VERSION_CODE, 22003);
}
