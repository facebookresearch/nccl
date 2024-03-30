
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <iostream>
#include "CtranAlgoDev.h"
#include "CtranGpe.h"
#include "CtranGpeDev.h"
#include "CtranGpeImpl.h"
#include "CtranGpeKernel.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

class CtranGpeTest : public ::testing::Test {
 public:
  CtranGpe* gpe;
  int cudaDev;
  ncclComm_t dummyComm;
  CtranAlgoDeviceState* dummyDevState_d {nullptr};

  CtranGpeTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    gpe = nullptr;

    CUDACHECK_TEST(cudaMalloc(&dummyDevState_d, sizeof(CtranAlgoDeviceState)));
  }
  void TearDown() override {
    if (gpe != nullptr) {
      delete gpe;
    }
    CUDACHECK_TEST(cudaFree(dummyDevState_d));
  }
};

class CtranGpeKernelTest : public ::testing::Test {
 public:
  volatile int* testFlag;
  CtranAlgoDeviceState* dummyDevState_d {nullptr};
  int cudaDev;
  CtranGpeKernelTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    CUDACHECKIGNORE(cudaSetDevice(cudaDev));
    CUDACHECKIGNORE(
        cudaHostAlloc((void**)&testFlag, sizeof(int), cudaHostAllocDefault));
    *testFlag = UNSET;

    CUDACHECK_TEST(cudaMalloc(&dummyDevState_d, sizeof(CtranAlgoDeviceState)));
  }
  void TearDown() override {
    CUDACHECKIGNORE(cudaFreeHost((void*)testFlag));
    CUDACHECK_TEST(cudaFree(dummyDevState_d));
  }
};

constexpr std::string_view kExpectedOutput = "CtranGpeTestAlgoFunc Called";
static ncclResult_t CtranGpeTestAlgoFunc(
    std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  std::cout << kExpectedOutput;
  return ncclSuccess;
}

__global__ void CtranGpeTestKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelAllGatherArgs args) {
  int* a = const_cast<int*>(reinterpret_cast<const int*>(args.sendbuff));
  int expValInt = reinterpret_cast<int>(args.recvbuff);
  size_t count = args.nbytes;

  if (flag) {
    ncclKernelStartGpe(flag);
  }

  for (int i = 0; i < count; i++) {
    a[i] = expValInt;
  }

  if (flag) {
    ncclKernelWaitGpeTerminate(flag);
  }
}

__global__ void CtranGpeTestTerminateKernel(int* flag) {
  ncclKernelStartGpe(flag);
  ncclKernelWaitGpeTerminate(flag);
}

constexpr int numP2pElems = 10;
__global__ void CtranGpeTestP2pElemsKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelAllGatherArgs args) {
  KernelP2pElem* elemList = const_cast<KernelP2pElem*>(
      reinterpret_cast<const KernelP2pElem*>(args.sendbuff));
  KernelP2pElem* elem = elemList;
  int numElems = numP2pElems;
  // consume only numP2pElems amount of objects
  while (numElems) {
    elem->inuse[blockIdx.x] = false;
    elem = elem->next;
    numElems--;
  }
}

TEST_F(CtranGpeTest, gpeThread) {
  gpe = new CtranGpe(cudaDev);
  EXPECT_THAT(gpe, testing::NotNull());
}

TEST_F(CtranGpeTest, SubmitOpBadArgs) {
  ncclResult_t res = ncclSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev);

  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::SEND, dummyComm);
  op->send.sendbuff = nullptr;
  op->send.count = 0;
  op->send.datatype = ncclInt8;
  op->send.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  auto kernelConfig = KernelConfig(KernelConfig::KernelType::SEND, nullptr);
  kernelConfig.args.devState_d = dummyDevState_d;

  /* NOTE: invalid CUDA kernel should return error code */
  res =
      gpe->submit(std::move(ops), &CtranGpeTestAlgoFunc, kernelConfig, nullptr);
  EXPECT_NE(res, ncclSuccess);

  // Invalid devState_d should be checked and return ncclInternalError
  kernelConfig.args.devState_d = nullptr;
  res = gpe->submit(
      std::move(ops),
      &CtranGpeTestAlgoFunc,
      kernelConfig,
      reinterpret_cast<void*>(CtranGpeTestKernel));
  EXPECT_EQ(res, ncclInternalError);
}

constexpr int count = 1024;
constexpr int kKernelpdatedVal = 100;

TEST_F(CtranGpeTest, SubmitOpKernel) {
  ncclResult_t res = ncclSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev);
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  int* a = nullptr;
  CUDACHECK_TEST(cudaMalloc(&a, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(a, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::RECV, dummyComm);
  op->recv.recvbuff = nullptr;
  op->recv.count = 0;
  op->recv.datatype = ncclInt8;
  op->recv.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  // Use ALLGATHER kernel config to pass test variables
  auto config = KernelConfig(KernelConfig::KernelType::ALLGATHER, stream);
  ctranKernelSetAllGatherArgs(
      a,
      reinterpret_cast<void*>(kKernelpdatedVal),
      count,
      dummyDevState_d,
      &config.args);

  testing::internal::CaptureStdout();

  res = gpe->submit(
      std::move(ops),
      &CtranGpeTestAlgoFunc,
      config,
      reinterpret_cast<void*>(CtranGpeTestKernel));

  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaStreamDestroy(stream));
  delete gpe;
  gpe = nullptr;

  // check GPE hostFn has been called
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput));

  // check kernel has been called
  std::vector<int> a_host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      a_host.data(), a, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(a_host, testing::Each(kKernelpdatedVal));
}

TEST_F(CtranGpeTest, SubmitOnlyKernel) {
  ncclResult_t res = ncclSuccess;
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  int* a = nullptr;
  CUDACHECK_TEST(cudaMalloc(&a, sizeof(int) * count));
  CUDACHECK_TEST(cudaMemset(a, 0, sizeof(int) * count));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<std::unique_ptr<struct OpElem>> emptyOps;

  // Use ALLGATHER kernel config to pass test variables
  auto config = KernelConfig(KernelConfig::KernelType::ALLGATHER, stream);
  ctranKernelSetAllGatherArgs(
      a,
      reinterpret_cast<void*>(kKernelpdatedVal),
      count,
      dummyDevState_d,
      &config.args);

  // empty OpGroup would launch only kernel
  res = gpe->submit(
      std::move(emptyOps),
      nullptr,
      config,
      reinterpret_cast<void*>(CtranGpeTestKernel));
  EXPECT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // check kernel has been called
  std::vector<int> a_host(count, 0);
  CUDACHECK_TEST(cudaMemcpy(
      a_host.data(), a, sizeof(int) * count, cudaMemcpyDeviceToHost));
  EXPECT_THAT(a_host, testing::Each(kKernelpdatedVal));

  CUDACHECK_TEST(cudaFree(a));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

TEST_F(CtranGpeKernelTest, launchTerminateStallKernel) {
  dim3 grid = {1, 1, 1};
  dim3 blocks = {1, 1, 1};
  void* args[] = {&testFlag};
  auto res = cudaLaunchKernel(
      reinterpret_cast<void*>(CtranGpeTestTerminateKernel),
      grid,
      blocks,
      args,
      0,
      0);

  EXPECT_EQ(res, cudaSuccess);

  while (*testFlag != KERNEL_STARTED) {
    EXPECT_THAT(*testFlag, testing::Not(KERNEL_TERMINATE));
  }

  *testFlag = KERNEL_TERMINATE;
  res = cudaStreamSynchronize(0);

  EXPECT_EQ(res, cudaSuccess);
}

TEST_F(CtranGpeKernelTest, SubmitKernelWithP2pElems) {
  // Ensure NCCL_CTRAN_NUM_KERNEL_P2PELEMS has been set
  ncclCvarInit();
  auto gpe = std::unique_ptr<CtranGpe>(new CtranGpe(cudaDev));
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  // Allocate p2pElems
  KernelP2pElem* elemList = nullptr;
  constexpr int ngroups = 5;
  NCCLCHECK_TEST(gpe->allocKernelP2pElems(numP2pElems, ngroups, &elemList));

  // Check allocated number of p2pElems is as expected
  int nAllocated = 0;
  KernelP2pElem* elem = elemList;
  while (elem) {
    elem = elem->next;
    nAllocated++;
  }
  EXPECT_EQ(nAllocated, numP2pElems);

  // Use ALLGATHER kernel config to pass test variables and launch with ngroups
  // gridSize to consume the elems
  std::vector<std::unique_ptr<struct OpElem>> emptyOps;
  auto config = KernelConfig(KernelConfig::KernelType::ALLGATHER, stream);
  ctranKernelSetAllGatherArgs(elemList, nullptr, 0, dummyDevState_d, &config.args);
  config.numBlocks = ngroups;

  // Empty OpGroup would launch only kernel
  NCCLCHECK_TEST(gpe->submit(
      std::move(emptyOps),
      nullptr,
      config,
      reinterpret_cast<void*>(CtranGpeTestP2pElemsKernel)));

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Check each element has been consumed by kernel
  elem = elemList;
  while (elem) {
    std::vector<int> inuse(elem->inuse, elem->inuse + ngroups);
    EXPECT_THAT(inuse, testing::Each(false));
    elem = elem->next;
  }

  // Skip check for reclaim which is an internal operation and triggered in GPE
  // destructor. Coverd by separate UT

  CUDACHECK_TEST(cudaStreamDestroy(stream));
}
