// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <utils.h>

int main(int argc, char* argv[]) {
  int rank = 0, nproc = 1, localRank = 0;
  int count = 1024 * 1024;

  ncclComm_t comm;
  cudaStream_t stream;
  int* userBuff = NULL;

  ncclUniqueId ncclId;
  NCCLCHECK(ncclGetUniqueId(&ncclId));

  printf("Hello world. NCCL_VERSION %d-%s\n", NCCL_VERSION_CODE, NCCL_SUFFIX);

  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclCommInitRank(&comm, nproc, ncclId, rank));

  CUDACHECK(cudaMalloc(&userBuff, count * sizeof(int)));
  NCCLCHECK(ncclAllReduce(
      (const void*)userBuff, userBuff, count, ncclInt, ncclSum, comm, stream));

#ifdef NCCL_COMM_GET_UNIQUE_HASH
  uint64_t ncclCommHash;
  NCCLCHECK(ncclCommGetUniqueHash(comm, &ncclCommHash));
  printf("ncclCommHash %lu\n", ncclCommHash);
#endif

  CUDACHECK(cudaFree(userBuff));
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));

  return EXIT_SUCCESS;
}
