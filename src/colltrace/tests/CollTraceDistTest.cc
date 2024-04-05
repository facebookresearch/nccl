// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <CollTrace.h>
#include <ExtUtils.h>
#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <rfe/scubadata/ScubaData.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "Ctran.h"
#include "checks.h"
#include "json/json.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn on NCCL debug logging for verbose testing
    // Allow user to change via command line
    setenv("NCCL_DEBUG", "INFO", 0);
    setenv("NCCL_DEBUG_SUBSYS", "INIT,COLL", 0);

    // Initialize CVAR so that we can overwrite global variable in each test
    initEnv();
  }

  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class CollTraceTest : public ::testing::Test {
 public:
  CollTraceTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void prepareAllreduce(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

  void prepareAllToAll(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
  }

  void prepareSendRecv(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

  bool prepareDumpDir(const std::string& dir) {
    try {
      // always re-create a fresh dir to ensure output files are up-to-date
      if (std::filesystem::exists(dir)) {
        std::filesystem::remove_all(dir);
      }
      std::filesystem::create_directories(dir);
    } catch (const std::exception& e) {
      printf(
          "Rank %d failed to create directory %s: %s\n",
          this->globalRank,
          dir.c_str(),
          e.what());
      return false;
    }
    return true;
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  cudaStream_t stream;
};

TEST_F(CollTraceTest, TraceFeatureEnableCollTrace) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  testing::internal::CaptureStdout();
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  std::string output = testing::internal::GetCapturedStdout();
  //
  EXPECT_THAT(
      output, testing::HasSubstr("enabled features: trace - Init COMPLETE"));
  EXPECT_THAT(
      output,
      testing::Not(testing::HasSubstr("COLLTRACE initialization failed")));
}

TEST_F(CollTraceTest, VerboseAllReduce) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  comm->collTrace->waitForWorkerFinishQueue();

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "COLLTRACE: opCount " << std::hex << i << " opName AllReduce";
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
}

TEST_F(CollTraceTest, VerboseAllToAll) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareAllToAll(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->collTrace->waitForWorkerFinishQueue();

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "COLLTRACE: opCount " << std::hex << i << " opName SendRecv";
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
}

TEST_F(CollTraceTest, VerboseSendRecv) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareSendRecv(count);
  int peer = (this->globalRank + 1) % this->numRanks;
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, peer, comm, stream));
    NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, peer, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->collTrace->waitForWorkerFinishQueue();

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "COLLTRACE: opCount " << std::hex << i << " opName SendRecv";
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
}

TEST_F(CollTraceTest, VerboseSendOrRecv) {
  if (this->numRanks % 2) {
    GTEST_SKIP() << "This test requires even number of ranks";
  }

  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareSendRecv(count);
  for (int i = 0; i < nColl; i++) {
    // even rank sends to odd rank (e.g, 0->1, 2->3)
    if (this->globalRank % 2 == 0) {
      int peer = this->globalRank + 1;
      NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, peer, comm, stream));
    } else {
      int peer = this->globalRank - 1;
      NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, peer, comm, stream));
    }
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->collTrace->waitForWorkerFinishQueue();

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    if (this->globalRank % 2 == 0) {
      ss << "COLLTRACE: opCount " << std::hex << i << " opName Send";
    } else {
      ss << "COLLTRACE: opCount " << std::hex << i << " opName Recv";
    }
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
}

TEST_F(CollTraceTest, DumpAllFinished) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->collTrace != nullptr);
  comm->collTrace->waitForWorkerFinishQueue();
  auto dump = comm->collTrace->dump();
  EXPECT_EQ(dump.pastColls.size(), nColl);
  EXPECT_EQ(dump.currentColl, nullptr);
}

TEST_F(CollTraceTest, DumpWithUnfinished) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->collTrace != nullptr);
  comm->collTrace->waitForWorkerFinishQueue();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  auto dump = comm->collTrace->dump();

  EXPECT_TRUE(dump.pastColls.size() >= nColl);
  EXPECT_TRUE(dump.pendingColls.size() <= nColl);
  int hasPending = dump.currentColl == nullptr ? 0 : 1;
  EXPECT_TRUE(dump.pendingColls.size() + dump.pendingColls.size() == nColl * 2 - hasPending);
  for (auto& coll: dump.pastColls) {
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
  }
  for (auto& coll: dump.pendingColls) {
    EXPECT_EQ(coll.startTs.time_since_epoch().count(), 0);
  }
}

TEST_F(CollTraceTest, TestSerializedDump) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->collTrace != nullptr);
  comm->collTrace->waitForWorkerFinishQueue();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  auto dump = comm->collTrace->dump();

  EXPECT_TRUE(dump.pastColls.size() >= nColl);
  EXPECT_TRUE(dump.pendingColls.size() <= nColl);
  int hasPending = dump.currentColl == nullptr ? 0 : 1;
  EXPECT_TRUE(dump.pendingColls.size() + dump.pendingColls.size() == nColl * 2 - hasPending);
  constexpr std::string_view startTsStr = "\"startTs\": ";
  for (auto& coll: dump.pastColls) {
    auto serialized = coll.serialize(true);
    EXPECT_THAT(serialized, testing::HasSubstr(startTsStr));
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
  }
  for (auto& coll: dump.pendingColls) {
    auto serialized = coll.serialize(true);
    EXPECT_THAT(serialized, testing::HasSubstr(startTsStr));
    EXPECT_EQ(coll.startTs.time_since_epoch().count(), 0);
  }
}

TEST_F(CollTraceTest, TestScubaDump) {
#ifndef ENABLE_FB_INTERNAL
  GTEST_SKIP() << "This test requires FB internal build";
#endif

  // overwrite CollTrace features before creating comm
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"fb"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  auto testCallback = [](void* scubaSample) {
    auto sample = static_cast<facebook::rfe::ScubaData::Sample*>(scubaSample);
    std::stringstream ss;
    ss << "COLLTRACE TEST: logging to scuba:"
       << " commHash: " << std::hex << sample->getIntValue("commHash")
       << " opCount: " << std::hex << sample->getIntValue("opCount")
       << " stream: " << std::hex << sample->getIntValue("stream")
       << " iteration: " << std::hex << sample->getIntValue("iteration")
       << " opName: " << sample->getNormalValue("opName")
       << " sendbuff: " << std::hex << sample->getIntValue("sendbuff")
       << " recvbuff: " << std::hex << sample->getIntValue("recvbuff")
       << " count: " << std::hex << sample->getIntValue("count") << std::endl;
    std::string traceLog = ss.str();
    fwrite(traceLog.c_str(), 1, traceLog.size(), stdout);
  };

  scubaTestCallback = testCallback;

  testing::internal::CaptureStdout();

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->collTrace != nullptr);
  comm->collTrace->waitForWorkerFinishQueue();

  auto dump = comm->collTrace->dump();

  std::string output = testing::internal::GetCapturedStdout();
  for (auto& coll : dump.pastColls) {
    std::stringstream ss;
    ss << "COLLTRACE TEST: logging to scuba:"
       << " commHash: " << std::hex << coll.info.comm->commHash
       << " opCount: " << std::hex << coll.opCount << " stream: " << std::hex
       << reinterpret_cast<uint64_t>(coll.stream) << " iteration: " << std::hex
       << coll.iteration << " opName: " << coll.info.opName
       << " sendbuff: " << std::hex
       << reinterpret_cast<uint64_t>(coll.info.sendbuff)
       << " recvbuff: " << std::hex
       << reinterpret_cast<uint64_t>(coll.info.recvbuff)
       << " count: " << std::hex << static_cast<uint64_t>(coll.info.count)
       << std::endl;
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
}

TEST_F(CollTraceTest, ReportToLog) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"file"});
  auto dirGuard =
      EnvRAII(NCCL_COLLTRACE_DIR, std::string{"/tmp/colltrace_test"});

  if (!prepareDumpDir(NCCL_COLLTRACE_DIR)) {
    GTEST_SKIP() << "Failed to create dump directory. Skipping test.";
  }

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // CollTrace thread can be slower on remote execution, thus the results may
  // not be filled yet. Explictly wait for the results to be filled.
  EXPECT_TRUE(comm->collTrace != nullptr);
  comm->collTrace->waitForWorkerFinishQueue();

  EXPECT_TRUE(comm->collTrace->dumpResultsToFile());

  // Each rank checks the file dumped for itself
  const std::string fname = NCCL_COLLTRACE_DIR + "/comm" +
      hashToHexStr(comm->commHash) + "_rank" + std::to_string(comm->rank) +
      "_online.json";
  EXPECT_TRUE(std::filesystem::exists(fname));

  printf("Checking dumped file %s\n", fname.c_str());

  std::ifstream logFile(fname);
  Json::Value jsonLog;
  logFile >> jsonLog;
  uint64_t opCount = 0;
  for (auto& entry : jsonLog) {
    EXPECT_EQ(entry["opCount"].asUInt64(), opCount++);
    EXPECT_EQ(entry["opName"], "AllReduce");
    EXPECT_EQ(
        entry["sendbuff"].asUInt64(), reinterpret_cast<uint64_t>(sendBuf));
    EXPECT_EQ(
        entry["recvbuff"].asUInt64(), reinterpret_cast<uint64_t>(recvBuf));
    EXPECT_TRUE(entry.isMember("count"));
    EXPECT_EQ(entry["count"].asUInt64(), count);
    EXPECT_EQ(entry["datatype"], "ncclInt32");
    EXPECT_EQ(entry["redOp"], "ncclSum");
    EXPECT_EQ(entry["root"], 0);
    // Do not seriously check value for following fields since we don't know
    // the exact value.
    EXPECT_TRUE(entry.isMember("algorithm"));
    EXPECT_TRUE(entry.isMember("protocol"));
    EXPECT_TRUE(entry.isMember("pattern"));
    EXPECT_TRUE(entry.isMember("channelId"));
    EXPECT_GE(entry["nChannels"], 1);
    EXPECT_GE(entry["nThreads"], 1);
    EXPECT_GT(entry["latencyUs"], 0);
  }
  logFile.close();
}

TEST_F(CollTraceTest, TestRecordNoDropBelowLimit) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto recordGuard =
      EnvRAII(NCCL_COLLTRACE_RECORD_MAX, NCCL_COLLTRACE_RECORD_MAX_DEFAULT);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  if (NCCL_COLLTRACE_RECORD_MAX_DEFAULT <= 1) {
    GTEST_SKIP()
        << "NCCL_COLLTRACE_RECORD_MAX_DEFAULT is too small. Skipping test.";
  }
  const int nColl = std::max(NCCL_COLLTRACE_RECORD_MAX - 5, 1);

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  EXPECT_THAT(comm->collTrace, testing::NotNull());

  comm->collTrace->waitForWorkerFinishQueue();

  auto traceDump = comm->collTrace->dump();
  EXPECT_EQ(traceDump.pastColls.size(), nColl);
}

TEST_F(CollTraceTest, TestRecordNoDropByEnv) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto recordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = std::max(NCCL_COLLTRACE_RECORD_MAX_DEFAULT, 100) * 5;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_THAT(comm->collTrace, testing::NotNull());

  comm->collTrace->waitForWorkerFinishQueue();

  auto traceDump = comm->collTrace->dump();
  EXPECT_EQ(traceDump.pastColls.size(), nColl);
}

TEST_F(CollTraceTest, TestRecordDropOverLIMIT) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto recordGuard =
      EnvRAII(NCCL_COLLTRACE_RECORD_MAX, NCCL_COLLTRACE_RECORD_MAX_DEFAULT);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = std::max(NCCL_COLLTRACE_RECORD_MAX_DEFAULT, 100) * 5;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_THAT(comm->collTrace, testing::NotNull());

  comm->collTrace->waitForWorkerFinishQueue();

  auto traceDump = comm->collTrace->dump();
  EXPECT_EQ(traceDump.pastColls.size(), NCCL_COLLTRACE_RECORD_MAX);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
