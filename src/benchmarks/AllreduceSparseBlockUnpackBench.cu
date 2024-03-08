// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <CLI11/CLI11.hpp>
#include <nccl.h>
#include "bench_common.h"
#include "checks.h"
#include "all_reduce_sparse_block.cuh"
#include "tests_common.cuh"

template <typename T>
static __global__ void setDeviceBufValKernel(T* buf, float val, size_t count) {
  for (size_t i = 0; i < count; i++)
    buf[i] = floatToType<T>(val);
}

template <typename T>
static ncclResult_t setDeviceBufVal(T* buf, float val, size_t count) {
  dim3 grid = {256, 1, 1};
  dim3 block = {1024, 1, 1};
  void* args[3] = {&buf, &val, &count};
  void* fn = (void*)setDeviceBufValKernel<T>;
  CUDACHECK(cudaLaunchKernel(fn, grid, block, args));
  return ncclSuccess;
}

template <typename T>
class AllreduceSparseBlockUnpackBench {
 public:
  unsigned int bestNumThreadBlocks{0};
  unsigned int bestThreadBlockSize{0};

  AllreduceSparseBlockUnpackBench() {
    // turn on nccl log to report errors in CUDACHECK*/NCCLCHECK*
    setenv("NCCL_DEBUG", "INFO", 1);

    CUDACHECKABORT(cudaEventCreate(&start_));
    CUDACHECKABORT(cudaEventCreate(&stop_));
    CUDACHECKABORT(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDACHECKABORT(cudaOccupancyMaxPotentialBlockSize(
        (int*)&bestNumThreadBlocks,
        (int*)&bestThreadBlockSize,
        ncclKernel_AllReduceSparseBlock_Unpack<T>));
  }

  ~AllreduceSparseBlockUnpackBench() {
    CUDACHECKABORT(cudaEventDestroy(start_));
    CUDACHECKABORT(cudaEventDestroy(stop_));
    CUDACHECKABORT(cudaStreamDestroy(stream_));
  }

  ncclResult_t
  initBuffers(size_t inBlockCount, size_t inBlockLength, size_t inBlockStride) {
    blockCount_ = inBlockCount;
    blockLength_ = inBlockLength;
    blockStride_ = inBlockStride;
    size_t packNumElems = blockCount_ * blockLength_;
    size_t unpackNumElems = blockCount_ * blockStride_;

    // generate unpack indices based on stride
    // indices={stride, 2 * stride, 3 * stride, ...,(n-1) * stride} for
    // blockCount n
    std::vector<int64_t> unpackIndices;
    for (int i = 0; i < blockCount_; i++) {
      unpackIndices.push_back(i * blockStride_);
    }

    CUDACHECK(cudaMalloc((void**)&packbuf_d_, sizeof(T) * packNumElems));
    CUDACHECK(cudaMalloc((void**)&unpackbuf_d_, sizeof(T) * unpackNumElems));
    CUDACHECK(
        cudaMalloc((void**)&unpackIndices_d_, sizeof(int64_t) * blockCount_));

    NCCLCHECK(setDeviceBufVal(packbuf_d_, dataVal_, packNumElems));
    NCCLCHECK(setDeviceBufVal(unpackbuf_d_, unsetVal_, unpackNumElems));
    CUDACHECK(cudaMemcpy(
        unpackIndices_d_,
        unpackIndices.data(),
        sizeof(int64_t) * blockCount_,
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    return ncclSuccess;
  }

  ncclResult_t freeBuffers() {
    CUDACHECK(cudaFree(packbuf_d_));
    CUDACHECK(cudaFree(unpackbuf_d_));
    CUDACHECK(cudaFree(unpackIndices_d_));

    return ncclSuccess;
  }

  ncclResult_t
  run(unsigned int numThreadBlocks, unsigned int threadBlockSize, int numIter) {
    dim3 grid = {numThreadBlocks, 1, 1};
    dim3 block = {threadBlockSize, 1, 1};
    void* args[5] = {
        (void*)&unpackbuf_d_,
        (void*)&packbuf_d_,
        (void*)&blockCount_,
        (void*)&unpackIndices_d_,
        (void*)&blockLength_};
    void* fn = (void*)ncclKernel_AllReduceSparseBlock_Unpack<T>;

    CUDACHECK(cudaEventRecord(start_, stream_));
    for (int r = 0; r < numIter; r++) {
      CUDACHECK(cudaLaunchKernel(fn, grid, block, args, 0, stream_));
    }
    CUDACHECK(cudaEventRecord(stop_, stream_));
    CUDACHECK(cudaStreamSynchronize(stream_));
    CUDACHECK(cudaEventElapsedTime(&gpuTimeMs_, start_, stop_));
    gpuTimeMs_ /= numIter;

    return ncclSuccess;
  }

  double reportLatencyUs() {
    return gpuTimeMs_ * 1e3;
  }

  double reportBwGBPerSec() {
    // count both read and the write bytes
    return 2 * blockCount_ * blockLength_ * sizeof(T) / gpuTimeMs_ / 1e6;
  }

 private:
  // value for data block and unset unpackbuf;
  // use float so that can convert to all types including half and bfloat16
  const float dataVal_{9};
  const float unsetVal_{0};

  // cached user input arguments for each test
  size_t blockCount_{0};
  size_t blockLength_{0};
  size_t blockStride_{0};

  // internally created device buffers for each test
  T* unpackbuf_d_{nullptr};
  T* packbuf_d_{nullptr};
  int64_t* unpackIndices_d_{nullptr};

  // resources initialized once and reused for all tests
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;

  // Benchmark metrics
  float gpuTimeMs_{0};
};

int64_t blockCountStart = 4, blockCountEnd = 1024, blockLength = 4,
        blockStride = 4;
unsigned int numThreadBlocks = 0, threadBlockSize = 0;
int numIter = 100, numWarmup = 10;
int device = 0;
std::string dataTypeStr = "int32";

template <typename T>
ncclResult_t runBench() {
  double latency = 0, bandwidth = 0;
  CUDACHECK(cudaSetDevice(device));

  auto bench = std::make_unique<AllreduceSparseBlockUnpackBench<T>>();
  numThreadBlocks =
      numThreadBlocks > 0 ? numThreadBlocks : bench->bestNumThreadBlocks;
  threadBlockSize =
      threadBlockSize > 0 ? threadBlockSize : bench->bestThreadBlockSize;

  printf(
      "Device %d dtype %s numThreadBlocks %d threadBlockSize %d blockCountStart %ld blockCountEnd %ld\n",
      device,
      dataTypeStr.c_str(),
      numThreadBlocks,
      threadBlockSize,
      blockCountStart,
      blockCountEnd);
  for (size_t blockCount = blockCountStart; blockCount < blockCountEnd;
       blockCount *= 2) {
    NCCLCHECK(bench->initBuffers(blockCount, blockLength, blockStride));
    NCCLCHECK(bench->run(numThreadBlocks, threadBlockSize, numWarmup));
    NCCLCHECK(bench->run(numThreadBlocks, threadBlockSize, numIter));
    latency = bench->reportLatencyUs();
    bandwidth = bench->reportBwGBPerSec();
    NCCLCHECK(bench->freeBuffers());

    printf(
        "BlockCount %ld blockLength %ld blockStride %ld totalUnpackBytes %ld "
        "latency %.2f us bandwidth %.2f GB/s\n",
        blockCount,
        blockLength,
        blockStride,
        blockCount * blockLength * sizeof(T),
        latency,
        bandwidth);
  }
  return ncclSuccess;
}

int main(int argc, char** argv) {
  ncclResult_t ret = ncclSuccess;
  CLI::App app{"ncclKernel_AllReduceSparseBlock_Unpack microbenchmark"};

  app.add_option("--block-count-start", blockCountStart, "Starting block count")
      ->default_val(blockCountStart);
  app.add_option("--block-count-end", blockCountEnd, "End block count")
      ->default_val(blockCountEnd);
  app.add_option("--block-length", blockLength, "Block length")
      ->default_val(blockLength);
  app.add_option(
         "--block-stride",
         blockStride,
         "Block stride in elements to populate indices")
      ->default_val(blockStride);
  app.add_option(
         "--data-type", dataTypeStr, "Data type. Valid datatype includes")
      ->default_val(dataTypeStr);
  app.add_option("--num-iteration", numIter, "Number of iterations")
      ->default_val(numIter);
  app.add_option("--num-warmup", numWarmup, "Number of warmup")
      ->default_val(numWarmup);
  app.add_option("--device", device, "GPU device to run the benchmark")
      ->default_val(device);
  app.add_option(
      "--num-thread-blocks",
      numThreadBlocks,
      "Number of thread blocks to be used in unpack benchmark. If not set, use the best config recommended by cuda runtime.");
  app.add_option(
      "--thread-block-size",
      threadBlockSize,
      "Number of thread block size to be used in unpack benchmark. If not set, use the best config recommended by cuda runtime.");

  CLI11_PARSE(app, argc, argv);

  benchAbortSignalSetup();

  if (!dataTypeStr.compare("int8")) {
    NCCLCHECKGOTO(runBench<int8_t>(), ret, fail);
  } else if (!dataTypeStr.compare("uint8")) {
    NCCLCHECKGOTO(runBench<uint8_t>(), ret, fail);
  } else if (!dataTypeStr.compare("int32")) {
    NCCLCHECKGOTO(runBench<int32_t>(), ret, fail);
  } else if (!dataTypeStr.compare("uint32")) {
    NCCLCHECKGOTO(runBench<uint32_t>(), ret, fail);
  } else if (!dataTypeStr.compare("int64")) {
    NCCLCHECKGOTO(runBench<int64_t>(), ret, fail);
  } else if (!dataTypeStr.compare("uint64")) {
    NCCLCHECKGOTO(runBench<uint64_t>(), ret, fail);
  } else if (!dataTypeStr.compare("half")) {
    NCCLCHECKGOTO(runBench<half>(), ret, fail);
  } else if (!dataTypeStr.compare("float")) {
    NCCLCHECKGOTO(runBench<float>(), ret, fail);
  } else if (!dataTypeStr.compare("double")) {
    NCCLCHECKGOTO(runBench<double>(), ret, fail);
  }
#if defined(__CUDA_BF16_TYPES_EXIST__) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
  else if (!dataTypeStr.compare("bfloat16")) {
    NCCLCHECKGOTO(runBench<__nv_bfloat16>(), ret, fail);
  }
#endif
  else {
    BENCH_ERR("Invalid datataype %s\n", dataTypeStr.c_str());
    return ncclInvalidArgument;
  }

  return ncclSuccess;

fail:
  BENCH_ERR("Internal failure %d\n", ret);
  return ret;
}
