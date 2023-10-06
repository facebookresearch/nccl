// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <gtest/gtest.h>
#include <nccl.h>
#include "../include/comm_dda.h"

TEST(CommDDATest, Clique) {
  CUDACHECKIGNORE(cudaSetDevice(0));

  uint64_t x = 0;
  auto md = std::make_unique<ddaThreadSharedMd>(x);
  md->insertRank(0);
  md->insertRank(1);

  EXPECT_EQ(md->registeredRanks.size(), 2);

  md->deleteRank(1);
  md->insertRank(2);
  md->insertRank(3);

  EXPECT_EQ(md->registeredRanks.size(), 3);
  EXPECT_EQ(md->searchRank(0), true);
  EXPECT_EQ(md->searchRank(1), false);
}
