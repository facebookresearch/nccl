// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <unistd.h>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include "ProxyMock.h"
#include "comm.h"
#include "json/json.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

static bool VERBOSE = true;

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    setenv("NCCL_DEBUG", "WARN", 0);
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_PROXYTRACE", "trace", 0);
  }

  void TearDown() override {
    finalizeMpi();
  }

  ~MPIEnvironment() override {}
};

class CommDumpTest : public ::testing::Test {
 public:
  CommDumpTest() = default;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // Prepare data for sanity check after commSplit
    CUDACHECK_TEST(cudaMalloc(&this->dataBuf, sizeof(int) * this->dataCount));
  }

  void initData(int myRank) {
    std::vector<int> initVals(this->dataCount);
    for (int i = 0; i < this->dataCount; i++) {
      initVals[i] = i * myRank;
    }
    CUDACHECK_TEST(cudaMemcpy(
        this->dataBuf,
        initVals.data(),
        sizeof(int) * this->dataCount,
        cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(this->dataBuf));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* dataBuf{nullptr};
  const int dataCount{65536};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(CommDumpTest, DISABLED_SingleComm) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;

  res = ncclCommDump(this->comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  // commHash is intentially stored as hex string for readability
  std::stringstream commHashSs;
  commHashSs << "\"" << std::hex << comm->commHash << "\"";
  std::string commHashStr = commHashSs.str();

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], commHashStr);
  EXPECT_EQ(dump.count("rank"), 1);
  EXPECT_EQ(dump["rank"], std::to_string(this->comm->rank));
  EXPECT_EQ(dump.count("localRank"), 1);
  EXPECT_EQ(dump["localRank"], std::to_string(this->comm->localRank));
  EXPECT_EQ(dump.count("node"), 1);
  EXPECT_EQ(dump["node"], std::to_string(this->comm->node));

  // common metadata is dumped only on rank 0
  if (this->comm->rank == 0) {
    EXPECT_EQ(dump.count("nRanks"), 1);
    EXPECT_EQ(dump["nRanks"], std::to_string(this->comm->nRanks));
    EXPECT_EQ(dump.count("localRanks"), 1);
    EXPECT_EQ(dump["localRanks"], std::to_string(this->comm->localRanks));
    EXPECT_EQ(dump.count("nNodes"), 1);
    EXPECT_EQ(dump["nNodes"], std::to_string(this->comm->nNodes));
    EXPECT_EQ(dump.count("rings"), 1);
    if (dump.count("rings")) {
      Json::Value ringsObjs;
      std::stringstream(dump["rings"]) >> ringsObjs;
      EXPECT_EQ(ringsObjs.size(), this->comm->nChannels);
      for (int i = 0; i < this->comm->nChannels; i++) {
        EXPECT_EQ(ringsObjs[i].size(), this->comm->nRanks);
      }
    }
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);

  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);
}

TEST_F(CommDumpTest, DumpAfterColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  // commHash is intentially stored as hex string for readability
  std::stringstream commHashSs;
  commHashSs << std::hex << comm->commHash;
  std::string commHashStr = commHashSs.str();

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        this->comm,
        this->stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. We need add a hook to check for proxy
  // completion
  sleep(3);

  res = ncclCommDump(this->comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    Json::Value ctPastCollsObjs;
    std::stringstream(dump["CT_pastColls"]) >> ctPastCollsObjs;
    EXPECT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asUInt64(), i);
    }
  }

  // Proxy trace would be empty if nNodes == 1
  if (dump.count("PT_pastColls") && comm->nNodes > 1) {
    Json::Value ptPastCollsObjs;
    std::stringstream(dump["PT_pastColls"]) >> ptPastCollsObjs;
    EXPECT_EQ(ptPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ptPastCollsObjs[i]["commHash"].asString(), commHashStr);
      EXPECT_EQ(ptPastCollsObjs[i]["opCount"].asUInt64(), i);
    }
  }

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    Json::Value ctPendingCollsObjs;
    std::stringstream(dump["CT_pendingColls"]) >> ctPendingCollsObjs;
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColl")) {
    EXPECT_EQ(dump["CT_currentColl"], "null");
  }

  if (dump.count("PT_activeOps")) {
    Json::Value ptActiveOpsObjs;
    std::stringstream(dump["PT_activeOps"]) >> ptActiveOpsObjs;
    EXPECT_EQ(ptActiveOpsObjs.size(), 0);
  }

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, DISABLED_DumpDuringColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  if (comm->nNodes < 2) {
    GTEST_SKIP() << "Skipping test since nNodes < 2";
  }

  // commHash is intentially stored as hex string for readability
  std::stringstream commHashSs;
  commHashSs << std::hex << comm->commHash;
  std::string commHashStr = commHashSs.str();

  // Manually set the hanging point at opCount 5
  constexpr int hangOpCount = 5;
  constexpr int hangRank = 0;
  NCCL_PROXYMOCK_NET_SEND_FAILURE.clear();
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(hangOpCount));
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(hangRank));
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back("-1");
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back("-1");
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back("1"); // match only once
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back("30"); // delay 30 seconds

  // Manually re-initialze state of the mock instance
  auto& instance = ProxyMockNetSendFailure::getInstance();
  instance.initialize();

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        this->comm,
        this->stream));
  }

  // Wait till the hanging point is reached
  sleep(10);

  res = ncclCommDump(this->comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);
  EXPECT_EQ(dump.count("PT_activeColls"), 1);

  // Check records are dumped correctly and simply check if can be
  // parsed as json entries.

  // PastColl: Except some ranks may stuck at the hanging opCount but some
  // others may have finished and stuck at the next.
  if (dump.count("CT_pastColls")) {
    Json::Value ctPastCollsObjs;
    std::stringstream(dump["CT_pastColls"]) >> ctPastCollsObjs;
    size_t numPasts = ctPastCollsObjs.size();
    // For CollTrace, we know rank 0 must be hanging at hangOpCount
    if (comm->rank == hangRank) {
      EXPECT_EQ(numPasts, hangOpCount);
    } else {
      EXPECT_TRUE(numPasts == hangOpCount || numPasts == hangOpCount + 1);
    }
  }

  if (dump.count("PT_pastColls") && comm->nNodes > 1) {
    Json::Value ptPastCollsObjs;
    std::stringstream(dump["PT_pastColls"]) >> ptPastCollsObjs;
    size_t numPasts = ptPastCollsObjs.size();
    // For ProxyTrace, since rank A's proxy thread may serve rank B's network
    // op, we cannot assume a exact hang point based on rank
    EXPECT_TRUE(numPasts == hangOpCount || numPasts == hangOpCount + 1);
  }

  // Pending collectives
  if (dump.count("CT_pendingColls")) {
    Json::Value ctPendingCollsObjs;
    std::stringstream(dump["CT_pendingColls"]) >> ctPendingCollsObjs;
    size_t numPending = ctPendingCollsObjs.size();
    if (comm->rank == hangRank) {
      // should hang exactly at hangOpCount, and 1 current
      EXPECT_EQ(numPending, numColls - hangOpCount - 1);
    } else {
      // may hang at hangOpCount or next
      EXPECT_TRUE(
          numPending == numColls - hangOpCount - 1 ||
          numPending == numColls - hangOpCount - 2);
    }
  }

  if (dump.count("CT_currentColl")) {
    EXPECT_NE(dump["CT_currentColl"], "null");
    Json::Value ctCurrentCollsObj;
    std::stringstream(dump["CT_currentColl"]) >> ctCurrentCollsObj;
    if (comm->rank == hangRank) {
      EXPECT_EQ(ctCurrentCollsObj["opCount"].asUInt64(), hangOpCount);
      EXPECT_EQ(ctCurrentCollsObj["opName"], "AllReduce");
    }
  }

  if (dump.count("PT_activeOps")) {
    Json::Value ptActiveOpsObjs;
    std::stringstream(dump["PT_activeOps"]) >> ptActiveOpsObjs;
    EXPECT_GT(ptActiveOpsObjs.size(), 0);

    for (auto& op : ptActiveOpsObjs) {
      EXPECT_TRUE(op["rank"].asInt() >= 0 && op["rank"].asInt() < comm->nRanks);
      EXPECT_TRUE(
          op["remoteRank"].asInt() >= 0 &&
          op["remoteRank"].asInt() < comm->nRanks);
      EXPECT_TRUE(op["opCount"].asUInt64() >= 0);
      EXPECT_TRUE(op["coll"].asString() == "AllReduce");
      EXPECT_TRUE(
          op["opType"].asString() == "SEND" ||
          op["opType"].asString() == "RECV");
      EXPECT_EQ(op["commHash"].asString(), commHashStr);

      // Each rank may hang at hangOpCount and/or hangOpCount + 1 and may see
      // active ops in both opCounts
      EXPECT_TRUE(
          op["opCount"].asUInt64() == hangOpCount ||
          op["opCount"].asUInt64() == hangOpCount + 1);
    }

    // Skip cross-rank check as already covered in ProxyTraceDistTest
  }

  if (dump.count("PT_activeColls")) {
    Json::Value ptActiveCollsObjs;
    std::stringstream(dump["PT_activeColls"]) >> ptActiveCollsObjs;
    EXPECT_GT(ptActiveCollsObjs.size(), 0);

    for (auto& coll : ptActiveCollsObjs) {
      EXPECT_EQ(coll["commHash"].asString(), commHashStr);

      // Each rank may hang at hangOpCount and/or hangOpCount + 1 and may see
      // active ops in both opCounts
      EXPECT_TRUE(
          coll["opCount"].asUInt64() == hangOpCount ||
          coll["opCount"].asUInt64() == hangOpCount + 1);
      EXPECT_EQ(coll["coll"].asString(), "AllReduce");
      EXPECT_GT(coll["channelIds"].size(), 0);
    }
  }

  // Now let's wait for all communication to finish
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, DumpFromSubComm) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;
  ncclComm_t newcomm = NCCL_COMM_NULL;

  if (comm->nNodes < 2) {
    GTEST_SKIP() << "Skipping test since nNodes < 2";
  }

  // Only odd ranks create subcomm
  if (this->globalRank % 2 == 0) {
    res = ncclCommSplit(
        this->comm, NCCL_SPLIT_NOCOLOR, this->globalRank, &newcomm, nullptr);
    ASSERT_EQ(res, ncclSuccess);
    EXPECT_EQ(newcomm, (ncclComm_t)(NCCL_COMM_NULL));
    // no color ranks skip the test
    return;
  } else {
    res = ncclCommSplit(this->comm, 1, this->globalRank, &newcomm, nullptr);
    ASSERT_EQ(res, ncclSuccess);
  }

  // Manually set the hanging point at opCount 5 on the hangRank
  constexpr int hangOpCount = 5;
  constexpr int hangRank = 0;
  NCCL_PROXYMOCK_NET_SEND_FAILURE.clear();

  if (newcomm->rank == hangRank) {
    char hostname[1024];
    gethostname(hostname, 1024);
    int cuDev;
    CUDACHECK_TEST(cudaGetDevice(&cuDev));

    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(hangOpCount));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::string(hostname));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(cuDev)); // cuDev
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back("-1"); // step
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back("1"); // match only once
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back("30"); // delay 30 seconds
  }

  // Manually re-initialze state of the mock instance
  auto& instance = ProxyMockNetSendFailure::getInstance();
  instance.initialize();

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        newcomm,
        this->stream));
  }

  // Wait till the hanging point is reached
  sleep(10);

  res = ncclCommDump(newcomm, dump);
  ASSERT_EQ(res, ncclSuccess);

  if (newcomm->rank == hangRank && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }

  NCCLCHECK_TEST(ncclCommDestroy(newcomm));

}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
