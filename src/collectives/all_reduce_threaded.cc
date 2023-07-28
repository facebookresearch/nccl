// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "enqueue.h"
#include "nccl.h"
#include "argcheck.h"
#include <cmath>
#include <assert.h>

NCCL_PARAM(ThreadedAllreduceMaxBlocks, "THREADED_ALLREDUCE_MAX_BLOCKS", 1);
NCCL_PARAM(ThreadedAllreduceTreeThresholdNVS, "THREADED_ALLREDUCE_TREE_THRESHOLD_NVS", 256 * 1024);
NCCL_PARAM(ThreadedAllreduceTreeThresholdHCM, "THREADED_ALLREDUCE_TREE_THRESHOLD_HCM", 256 * 1024);
NCCL_PARAM(ThreadedAllreduceLargeMessageHCM, "THREADED_ALLREDUCE_LARGE_MESSAGE_HCM", 0);

#define ASSIGN_FUNC(func, templ, nranks)        \
    do {                                        \
        switch ((nranks)) {                     \
        case 2:                                 \
            func = (const void *) &templ<T,2>;  \
            break;                              \
                                                \
        case 4:                                 \
            func = (const void *) &templ<T,4>;  \
            break;                              \
                                                \
        case 8:                                 \
            func = (const void *) &templ<T,8>;  \
            break;                              \
                                                \
        case 16:                                \
            func = (const void *) &templ<T,16>; \
            break;                              \
                                                \
        default:                                \
            return ncclNumResults;              \
        }                                       \
    } while (0)

static inline int getMaxBlocks(ncclComm* comm)
{
    int maxBlocks = ncclParamThreadedAllreduceMaxBlocks();

    if (maxBlocks > comm->threadedRanks.devProp.multiProcessorCount) {
        WARN("NCCL_THREADED_ALLREDUCE_MAX_BLOCKS cannot be larger than %d\n",
             comm->threadedRanks.devProp.multiProcessorCount);
        maxBlocks = comm->threadedRanks.devProp.multiProcessorCount;
    }

    return maxBlocks;
}

template <typename T>
static ncclResult_t launchKernel(ncclComm *comm, const void *sendbuff, void *recvbuff, size_t count,
                                  cudaStream_t stream)
{
    const void *func;

    if (comm->threadedRanks.md->topoType == NCCL_THREADED_TOPO_TYPE__NVS) {
        if (count * sizeof(T) < ncclParamThreadedAllreduceTreeThresholdNVS()) {
            ASSIGN_FUNC(func, ncclKernel_AllReduce_Threaded_Flat, comm->nRanks);
        } else {
            ASSIGN_FUNC(func, ncclKernel_AllReduce_Threaded_Tree, comm->nRanks);
        }
    } else if (comm->threadedRanks.md->topoType == NCCL_THREADED_TOPO_TYPE__HCM) {
        if (count * sizeof(T) < ncclParamThreadedAllreduceTreeThresholdHCM()) {
            ASSIGN_FUNC(func, ncclKernel_AllReduce_Threaded_HCM_Flat, comm->nRanks);
        } else if (ncclParamThreadedAllreduceLargeMessageHCM()) {
            ASSIGN_FUNC(func, ncclKernel_AllReduce_Threaded_HCM_Tree, comm->nRanks);
        } else {
            return ncclNumResults;
        }
    } else {
        return ncclNumResults;
    }

    cudaFuncAttributes attr;
    CUDACHECK(cudaFuncGetAttributes(&attr, func));

    int maxBlocks = getMaxBlocks(comm);
    int numBlocks[2] = { 1, maxBlocks };
    int numThreads[2] = { 32, attr.maxThreadsPerBlock };
    int eltsPerThread = 16 / sizeof(T);
    dim3 grid;
    dim3 blocks;

    if (comm->threadedRanks.md->topoType == NCCL_THREADED_TOPO_TYPE__NVS) {
        if (count * sizeof(T) < ncclParamThreadedAllreduceTreeThresholdNVS()) {
            if (count % eltsPerThread) {
                return ncclNumResults;
            }
            if (sendbuff == recvbuff) {
                if (count * sizeof(T) > ncclParamThreadedAllreduceMaxTmpbufSize()) {
                    return ncclNumResults;
                }
            }
        } else {
            if ((count % (comm->nRanks * eltsPerThread)) ||
                (count * sizeof(T) / comm->nRanks > ncclParamThreadedAllreduceMaxTmpbufSize())) {
                return ncclNumResults;
            }
        }
    } else {
        if ((count % eltsPerThread) ||
            (count * sizeof(T) > ncclParamThreadedAllreduceMaxTmpbufSize())) {
            return ncclNumResults;
        }
    }

    if (count <= numBlocks[0] * numThreads[0] * eltsPerThread) {
        /* for small counts, use the minimum number of blocks and
         * threads, while keeping eltsPerThread elements to be
         * computed by each thread. */
        grid.x = numBlocks[0];
        blocks.x = numThreads[0];
    } else if (count <= numBlocks[0] * numThreads[1] * eltsPerThread) {
        /* for slightly larger counts, increase the number of threads
         * per block to up to the maximum number of threads. */
        grid.x = numBlocks[0];
        blocks.x = (int) std::ceil(count / (numBlocks[0] * eltsPerThread));
    } else if (count <= numBlocks[1] * numThreads[1] * eltsPerThread) {
        /* for even larger counts, increase the number of blocks to up
         * to the maximum number of blocks. */
        grid.x = (int) std::ceil(count / (numThreads[1] * eltsPerThread));
        blocks.x = numThreads[1];
    } else {
        /* for even larger counts, use the maximum number of threads
         * and blocks, and let each thread compute more elements. */
        grid.x = numBlocks[1];
        blocks.x = numThreads[1];
    }
    grid.y = 1;
    grid.z = 1;
    blocks.y = 1;
    blocks.z = 1;

    /* mbox_id,barrierFlag: 1,0; 0,1; 1,1; 0,0 */
    /* We maintain two mboxes (0,1) and two flags (0,1) for each mbox.
     * We start with an mboxId of 1, and a barrierFlag of 0.  And each
     * mbox is set to all zeroes.
     *
     * At this point, the last bit in each element of the mbox is 0.
     *
     * The first time we call this collective, we switch the mboxId to
     * 0, and the barrierFlag to 1.  This way, the barrier operation
     * in our kernel has to wait for the barrierFlag to move to 1
     * before it can exit the barrier.
     *
     * The second time we call this collective, we switch the mboxId
     * to 1, and the barrierFlag to 1.  This way, the barrier
     * operation in our kernel has to wait for the barrierFlag to move
     * to 1 before it can exit the barrier.
     *
     * At this point, the last bit in each element of the mbox is 1.
     *
     * The third time we call this collective, we switch the mboxId to
     * 0, and the barrierFlag to 0.  This way, the barrier operation
     * in our kernel has to wait for the barrierFlag to move to 0
     * before it can exit the barrier.
     *
     * The fourth time we call this collective, we switch the mboxId
     * to 1, and the barrierFlag to 0.  This way, the barrier
     * operation in our kernel has to wait for the barrierFlag to move
     * to 0 before it can exit the barrier.
     *
     * At this point, the last bit in each element of the mbox is 0,
     * and we are back to our original state.
     *
     * We need two mboxes, so one rank can get started with the next
     * collective even if not all ranks have exited the previous
     * collective (and thus are still using the previous mbox).
     */
    if (comm->threadedRanks.barrierMboxId == 1 && comm->threadedRanks.barrierFlag == 0) {
        comm->threadedRanks.barrierMboxId = !comm->threadedRanks.barrierMboxId;
        comm->threadedRanks.barrierFlag = !comm->threadedRanks.barrierFlag;
    } else if (comm->threadedRanks.barrierMboxId == 0 && comm->threadedRanks.barrierFlag == 1) {
        comm->threadedRanks.barrierMboxId = !comm->threadedRanks.barrierMboxId;
    } else if (comm->threadedRanks.barrierMboxId == 1 && comm->threadedRanks.barrierFlag == 1) {
        comm->threadedRanks.barrierMboxId = !comm->threadedRanks.barrierMboxId;
        comm->threadedRanks.barrierFlag = !comm->threadedRanks.barrierFlag;
    } else if (comm->threadedRanks.barrierMboxId == 0 && comm->threadedRanks.barrierFlag == 0) {
        comm->threadedRanks.barrierMboxId = !comm->threadedRanks.barrierMboxId;
    }

    if (comm->threadedRanks.md->topoType == NCCL_THREADED_TOPO_TYPE__NVS) {
        threadedRanksClique *clique = comm->threadedRanks.md->cliques.front();

        if (count * sizeof(T) < ncclParamThreadedAllreduceTreeThresholdNVS()) {
            void *rbuf = (sendbuff == recvbuff) ? clique->rankToTmpbuf[comm->rank] : recvbuff;

            void *args[] = {
                &clique->barrierMbox[comm->threadedRanks.barrierMboxId],
                &comm->threadedRanks.barrierFlag, &comm->rank, &sendbuff, &rbuf, &count
            };

            CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));

            if (sendbuff == recvbuff) {
                CUDACHECK(cudaMemcpyAsync(recvbuff, clique->rankToTmpbuf[comm->rank],
                                          count * sizeof(T), cudaMemcpyDefault, stream));
            }
        } else {
            void *args[] = {
                &clique->barrierMbox[comm->threadedRanks.barrierMboxId],
                &comm->threadedRanks.barrierFlag, &comm->rank, &sendbuff, &clique->rankToTmpbuf[comm->rank],
                &recvbuff, &count
            };

            CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));
        }
    } else if (comm->threadedRanks.md->topoType == NCCL_THREADED_TOPO_TYPE__HCM) {
        threadedRanksClique *clique = comm->threadedRanks.md->cliques.front();
        threadedRanksClique *peerClique = comm->threadedRanks.md->cliques.back();

        if (clique->rankToGpu.find(comm->rank) == clique->rankToGpu.end()) {
            clique = comm->threadedRanks.md->cliques.back();
            peerClique = comm->threadedRanks.md->cliques.front();
        }
        assert(clique->rankToGpu.find(comm->rank) != clique->rankToGpu.end());

        int cliqueRank;
        for (cliqueRank = 0; cliqueRank < clique->gpus.size(); cliqueRank++) {
            if (clique->rankToGpu[comm->rank] == clique->gpus[cliqueRank]) {
                break;
            }
        }
        assert(cliqueRank < clique->gpus.size());

        int peerRank = -1;
        for (auto it : peerClique->rankToGpu) {
            if (it.second == peerClique->gpus[cliqueRank]) {
                peerRank = it.first;
            }
        }
        assert(peerRank != -1);

        comm->threadedRanks.localMboxId = !comm->threadedRanks.localMboxId;

        assert(peerClique->rankToLocalMbox[comm->threadedRanks.localMboxId][peerRank] != nullptr);

        void *args[] = {
            &clique->barrierMbox[comm->threadedRanks.barrierMboxId],
            &clique->rankToLocalMbox[comm->threadedRanks.localMboxId][comm->rank],
            &peerClique->rankToLocalMbox[comm->threadedRanks.localMboxId][peerRank],
            &comm->threadedRanks.barrierFlag, &cliqueRank, &sendbuff,
            &clique->rankToTmpbuf[comm->rank], &recvbuff, &count
        };

        CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));
    }

    return ncclSuccess;
}

ncclResult_t ncclAllReduceThreaded(const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                   cudaStream_t stream)
{
    threadedRanksClique *clique;
    int numThreadedRanks = 0;
    ncclResult_t res;

    NCCLCHECK(ncclCommEnsureReady(comm));
    if (datatype < 0 || datatype >= ncclNumTypes) {
        WARN("AllReduce : invalid type %d", datatype);
        return ncclInvalidArgument;
    }
    if (op < 0 || ncclMaxRedOp < op) {
        WARN("AllReduce : invalid reduction operation %d", op);
        return ncclInvalidArgument;
    }
    NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "AllReduce"));
    NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "AllReduce"));

    for (auto c : comm->threadedRanks.md->cliques) {
        numThreadedRanks += c->rankToGpu.size();
    }

    if ((numThreadedRanks != comm->nRanks) ||  /* collective must only contain threaded ranks */
        (numThreadedRanks & (numThreadedRanks - 1)) ||  /* power of two ranks */
        (numThreadedRanks == 1) ||  /* more than one rank */
        (numThreadedRanks > ncclParamMaxThreadedRanks()) ||  /* only small rank counts are supported */
        (op != ncclSum) ||  /* only sum is supported */
        ((uintptr_t) sendbuff % 16) ||  /* 16-byte alignment */
        ((uintptr_t) recvbuff % 16)) {  /* 16-byte alignment */
        goto not_supported;
    }

    switch (datatype) {
    case ncclInt8:
        NCCLCHECK(launchKernel<char>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclUint8:
        NCCLCHECK(launchKernel<uint8_t>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclInt32:
        NCCLCHECK(launchKernel<int32_t>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclUint32:
        NCCLCHECK(launchKernel<uint32_t>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclInt64:
        NCCLCHECK(launchKernel<int64_t>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclUint64:
        NCCLCHECK(launchKernel<uint64_t>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclFloat16:
        NCCLCHECK(launchKernel<half>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclFloat32:
        NCCLCHECK(launchKernel<float>(comm, sendbuff, recvbuff, count, stream));
        break;

    case ncclFloat64:
        NCCLCHECK(launchKernel<double>(comm, sendbuff, recvbuff, count, stream));
        break;

#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
        NCCLCHECK(launchKernel<__nv_bfloat16>(comm, sendbuff, recvbuff, count, stream));
        break;
#endif

    default:
        goto not_supported;
    }

    return ncclSuccess;

not_supported:
    return ncclNumResults;
}
