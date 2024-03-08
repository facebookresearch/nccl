// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include "all_reduce_sparse_block.cuh"
#include "tests_common.cuh"

template <typename T>
static __global__ void setDeviceBufValKernel(T* buf, float val, size_t count) {
  for (size_t i = 0; i < count; i++)
    buf[i] = floatToType<T>(val);
}

template <typename T>
static __global__ void
convertBufValToFloatKernel(T* buf, float* floatBuf, size_t count) {
  for (size_t i = 0; i < count; i++)
    floatBuf[i] = toFloat<T>(buf[i]);
}

template <typename T>
static void setDeviceBufVal(T* buf, float val, size_t count) {
  dim3 grid = {256, 1, 1};
  dim3 block = {1024, 1, 1};
  void* args[3] = {&buf, &val, &count};
  void* fn = (void*)setDeviceBufValKernel<T>;
  CUDACHECK_TEST(cudaLaunchKernel(fn, grid, block, args));
}

template <typename T>
static void convertBufValToFloat(T* buf, float* floatBuf, size_t count) {
  dim3 grid = {256, 1, 1};
  dim3 block = {1024, 1, 1};
  void* args[3] = {&buf, &floatBuf, &count};
  void* fn = (void*)convertBufValToFloatKernel<T>;
  CUDACHECK_TEST(cudaLaunchKernel(fn, grid, block, args));
}

template <typename T>
class AllreduceSparseBlockUnpackTest : public testing::Test {
 public:
  void run(
      size_t blockCount,
      size_t blockLength,
      int64_t* unpackIndices,
      size_t recvCount) {
    blockCount_ = blockCount;
    blockLength_ = blockLength;
    unpackIndices_ = unpackIndices;
    recvCount_ = recvCount;

    CUDACHECK_TEST(cudaSetDevice(0));
    initBuffers();

    dim3 grid = {256, 1, 1};
    dim3 block = {1024, 1, 1};
    void* args[5] = {
        &unpackbuf_d_,
        &packbuf_d_,
        &blockCount_,
        &unpackIndices_d_,
        &blockLength_};
    void* fn = (void*)ncclKernel_AllReduceSparseBlock_Unpack<T>;
    CUDACHECK_TEST(cudaLaunchKernel(fn, grid, block, args));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void verify() {
    size_t currBlock = 0;
    size_t offset = 0;

    CUDACHECK_TEST(cudaSetDevice(0));
    // convert to Float host buffer to support half and bfloat16
    convertBufValToFloat(unpackbuf_d_, unpackbufFloat_d_, recvCount_);
    CUDACHECK_TEST(cudaMemcpy(
        unpackbufFloat_,
        unpackbufFloat_d_,
        sizeof(float) * recvCount_,
        cudaMemcpyDefault));

    // Traverse unpacked buffer;
    // if find a block then check if it equals to copied dataVal_; otherwise
    // it should be unchanged and equal to original unsetVal_
    while (offset < recvCount_) {
      // Find a block
      if (offset == unpackIndices_[currBlock]) {
        for (int i = 0; i < blockLength_; i++) {
          EXPECT_FLOAT_EQ(unpackbufFloat_[offset++], dataVal_);
        }
        currBlock++;
        assert(currBlock <= blockCount_);
      } else {
        // stride
        EXPECT_FLOAT_EQ(unpackbufFloat_[offset], unsetVal_);
        offset++;
      }
    }
  }

  void TearDown() override {
    freeBuffers();
  }

 private:
  // value for data block and unset unpackbuf;
  // use float so that can convert to all types including half and bfloat16
  const float dataVal_{9};
  const float unsetVal_{0};

  bool initialized_{false};

  // cached user input arguments for each test
  size_t blockCount_{0};
  size_t blockLength_{0};
  int64_t* unpackIndices_{nullptr};
  size_t recvCount_{0};

  // internally created device buffers for each test
  T* unpackbuf_d_{nullptr};
  T* packbuf_d_{nullptr};
  int64_t* unpackIndices_d_{nullptr};
  float* unpackbufFloat_d_{nullptr};

  // internal created host buffers for each test
  float* unpackbufFloat_{nullptr};

  void initBuffers(void) {
    size_t packNumElems = blockCount_ * blockLength_;

    unpackbufFloat_ = (float*)malloc(sizeof(float) * recvCount_);

    CUDACHECK_TEST(cudaMalloc((void**)&packbuf_d_, sizeof(T) * packNumElems));
    CUDACHECK_TEST(cudaMalloc((void**)&unpackbuf_d_, sizeof(T) * recvCount_));
    CUDACHECK_TEST(
        cudaMalloc((void**)&unpackbufFloat_d_, sizeof(float) * recvCount_));
    CUDACHECK_TEST(
        cudaMalloc((void**)&unpackIndices_d_, sizeof(int64_t) * blockCount_));

    setDeviceBufVal(packbuf_d_, dataVal_, packNumElems);
    setDeviceBufVal(unpackbuf_d_, unsetVal_, recvCount_);
    CUDACHECK_TEST(cudaMemcpy(
        unpackIndices_d_,
        unpackIndices_,
        sizeof(int64_t) * blockCount_,
        cudaMemcpyHostToDevice));

    initialized_ = true;
  }

  void freeBuffers() {
    if (!initialized_)
      return;
    CUDACHECK_TEST(cudaSetDevice(0));
    CUDACHECK_TEST(cudaFree(packbuf_d_));
    CUDACHECK_TEST(cudaFree(unpackbuf_d_));
    CUDACHECK_TEST(cudaFree(unpackbufFloat_d_));
    CUDACHECK_TEST(cudaFree(unpackIndices_d_));
    free(unpackbufFloat_);
  }
};

using UnpackTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    half,
    float,
    double,
#if defined(__CUDA_BF16_TYPES_EXIST__)
    __nv_bfloat16,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    __nv_fp8_e4m3,
    __nv_fp8_e5m2,
#endif
    >;
TYPED_TEST_SUITE(AllreduceSparseBlockUnpackTest, UnpackTypes);

// Small data size to test unpack with single thread blocks
TYPED_TEST(AllreduceSparseBlockUnpackTest, UnpackTypeSmall) {
  std::vector<int64_t> unpackIndices{0, 8, 84, 120};
  this->run(
      unpackIndices.size() /*blockCount*/,
      4 /*blockLength*/,
      unpackIndices.data(),
      1024 /*recvCount*/);
  this->verify();
}

// Large data size to test unpack with multiple thread blocks
TYPED_TEST(AllreduceSparseBlockUnpackTest, UnpackTypeLarge) {
  std::vector<int64_t> unpackIndices{8, 3000, 9600, 12000, 15000};
  this->run(
      unpackIndices.size() /*blockCount*/,
      2000 /*blockLength*/,
      unpackIndices.data(),
      65536 /*recvCount*/);
  this->verify();
}
