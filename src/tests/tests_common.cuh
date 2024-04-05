// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TESTS_COMMON_CUH_
#define TESTS_COMMON_CUH_

/**
 * Defines common utilities for tests that depends on CUDA or NCCL
 */

#include "cuda.h"
#include "mpi.h"
#include <tuple>

// Typed helper functions
template <typename T>
__device__ T floatToType(float val) {
  return (T)val;
}

template <typename T>
__device__ float toFloat(T val) {
  return (T)val;
}

template <>
__device__ half floatToType<half>(float val) {
  return __float2half(val);
}

template <>
__device__ float toFloat<half>(half val) {
  return __half2float(val);
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
__device__ __nv_bfloat16 floatToType<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
#endif

#define DECL_TYPED_KERNS(T)                        \
  template __device__ T floatToType<T>(float val); \
  template __device__ float toFloat<T>(T val);

DECL_TYPED_KERNS(int8_t);
DECL_TYPED_KERNS(uint8_t);
DECL_TYPED_KERNS(int32_t);
DECL_TYPED_KERNS(uint32_t);
DECL_TYPED_KERNS(int64_t);
DECL_TYPED_KERNS(uint64_t);
DECL_TYPED_KERNS(float);
DECL_TYPED_KERNS(double);
// Skip half and __nv_bfloat16 since already declared with specific type

#define MPICHECK_TEST(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK_TEST(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK_TEST(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Convenient RAII classes to simplify teardown of tests.

/**
 * Very thin RAII wrapper for ncclComm_t. The class is designed to be used in
 * the same way ncclComm_t to be used, but handle ncclCommDestroy automatically
 * in the destruction of the object. Sample usage:
 * NcclCommRAII comm{globalRank, numRanks, devId} // Init
 * func(comm)       // Pass to other functions as ncclComm_t
 * comm->proxyState // Dereference as if it is ncclComm_t
 */
ncclComm_t createNcclComm(int globalRank, int numRanks, int devId);
class NcclCommRAII {
  public:
    NcclCommRAII(int globalRank, int numRanks, int localRank) {
      comm_ = createNcclComm(globalRank, numRanks, localRank);
    }

    ncclComm& operator*() { return *comm_; }

    ncclComm_t operator->() { return comm_; }

    operator ncclComm_t() { return comm_; }

    ~NcclCommRAII() {
      NCCLCHECK_TEST(ncclCommDestroy(comm_));
    }

  private:
    ncclComm_t comm_;
    bool destroyed_ = false;
};

// Assign a new value for environment variable during one test, and change the
// value back at the end of the test.
template <typename T>
class EnvRAII {
 public:
  EnvRAII(T& envVar, T newValue) : oldValue_(envVar), envVar_(envVar) {
    envVar_ = newValue;
  }

  ~EnvRAII() {
    envVar_ = oldValue_;
  }

 private:
  T oldValue_;
  T& envVar_;
};

void initializeMpi(int argc, char **argv) {
  //initializing MPI
  MPICHECK_TEST(MPI_Init(&argc, &argv));
}

std::tuple<int, int, int> getMpiInfo() {
  int localRank, globalRank, numRanks = 0;

  MPICHECK_TEST(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  MPICHECK_TEST(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  MPI_Comm localComm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, globalRank, MPI_INFO_NULL, &localComm);
  MPI_Comm_rank(localComm, &localRank);
  MPI_Comm_free(&localComm);

  return std::make_tuple(localRank, globalRank, numRanks);
}

void finalizeMpi() {
  //finalizing MPI
  MPICHECK_TEST(MPI_Finalize());
}

// Convenience function to create a NCCL comm with all global ranks
// For creating comm with subset of ranks, would need call the internal steps directly.
ncclComm_t createNcclComm(int globalRank, int numRanks, int devId) {
  ncclUniqueId id;
  ncclComm_t comm;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0) ncclGetUniqueId(&id);
  MPICHECK_TEST(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK_TEST(cudaSetDevice(devId));

  //initializing NCCL
  NCCLCHECK_TEST(ncclCommInitRank(&comm, numRanks, id, globalRank));
  return comm;
}

std::tuple<int, int, int, ncclComm_t> setupNccl(int argc, char **argv) {
  int localRank, globalRank, numRanks = 0;
  ncclComm_t globalComm;

  initializeMpi(argc, argv);
  std::tie(localRank, globalRank, numRanks) = getMpiInfo();

  globalComm = createNcclComm(globalRank, numRanks, localRank);

  return std::make_tuple(localRank, globalRank, numRanks, globalComm);
}

void cleanupNccl(ncclComm_t globalComm) {
  //finalizing NCCL
  ncclCommDestroy(globalComm);

  finalizeMpi();
}

#endif
