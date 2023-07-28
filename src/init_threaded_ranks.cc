// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "nccl.h"
#include "comm.h"
#include "topo.h"
#include "checks.h"
#include <mutex>
#include <assert.h>
#include <vector>
#include <string>
#include <iostream>

NCCL_PARAM(ThreadedAllreduceMaxTmpbufSize, "THREADED_ALLREDUCE_MAX_TMPBUF_SIZE", 8 * 1024 * 1024);
NCCL_PARAM(MaxThreadedRanks, "MAX_THREADED_RANKS", 16);
NCCL_PARAM(ForceP2pAccess, "FORCE_P2P_ACCESS", 0);

static std::vector<threadedRanksMd *> threadedRanksMdList;
static std::mutex threadedRanksMdListMutex;

bool operator==(const ncclUniqueId& lhs, const ncclUniqueId& rhs)
{
    for (int i = 0; i < sizeof(ncclUniqueId); i++) {
        if (lhs.internal[i] != rhs.internal[i]) {
            return false;
        }
    }

    return true;
}

bool operator==(const threadedRanksMd& lhs, const threadedRanksMd& rhs)
{
    return (lhs.commId == rhs.commId);
}

static void findNvsConnectedGpus(struct ncclTopoNode *node, std::vector<int> &gpus, std::vector<uint64_t> &nvs)
{
    nvs.push_back(node->id);
    for (int i = 0; i < node->nlinks; i++) {
        if (node->links[i].type == LINK_NVL) {
            struct ncclTopoNode *remNode = node->links[i].remNode;
            if (remNode->type == GPU) {
                gpus.push_back(i);
            } else if (remNode->type == NVS) {
                auto it = std::find(nvs.begin(), nvs.end(), remNode->id);
                if (it == nvs.end()) {
                    findNvsConnectedGpus(remNode, gpus, nvs);
                }
            }
        }
    }
}

/* This function only returns two types of cliques: fully connected
 * (NVSwitch) or HCM, to support typically GPU topologies.  In all
 * other cases, we return an empty vector.
 *
 * We do not currently account for the number of NVLinks connecting
 * two GPUs because our algorithm does not take advantage of that
 * (yet!).
 *
 * This also enables p2p access between all GPUs that are connected
 * using NVLink.  Unfortunately, we do not have a good way to detect
 * which GPUs are being used for this particular job, so we enable p2p
 * access for all connected GPUs.
 */
static ncclResult_t topoDetect(ncclComm_t comm, std::vector<std::vector<int>>& cliques)
{
    int nGPUs = comm->topo->nodes[GPU].count;
    uint8_t adjacencyMatrix[nGPUs][nGPUs];

    /* clear the cliques before we start */
    cliques.clear();

    /* set adjacency matrix for self as connected */
    for (int i = 0; i < nGPUs; i++) {
        for (int j = 0; j < nGPUs; j++) {
            if (i == j) {
                adjacencyMatrix[i][j] = 1;
            } else if (ncclParamForceP2pAccess()) {
                adjacencyMatrix[i][j] = 1;
            } else {
                adjacencyMatrix[i][j] = 0;
            }
        }
    }

    /* for each GPU in the system */
    for (int i = 0; i < nGPUs; i++) {
        /* for each NVLink connection on that GPU */
        for (int j = 0; j < comm->topo->nodes[GPU].nodes[i].nlinks; j++) {
            if (comm->topo->nodes[GPU].nodes[i].links[j].type == LINK_NVL) {
                struct ncclTopoNode *remNode = comm->topo->nodes[GPU].nodes[i].links[j].remNode;
                if (remNode->type == GPU) {  /* if it is connected to a GPU */
                    adjacencyMatrix[i][remNode->gpu.dev] = 1;
                } else if (remNode->type == NVS) {  /* if it is connected to an NVSwitch */
                    std::vector<uint64_t> nvs;
                    std::vector<int> gpus;
                    findNvsConnectedGpus(remNode, gpus, nvs);
                    for (auto it : gpus) {
                        adjacencyMatrix[i][it] = 1;
                    }
                }
            }
        }
    }


    /***** Detect fully connected (NVSwitch) topology *****/
    bool connected;
    connected = true;
    for (int i = 0; i < nGPUs; i++) {
        for (int j = 0; j < nGPUs; j++) {
            if (adjacencyMatrix[i][j] == 0) {
                connected = false;
                break;
            }
        }
        if (connected == false) {
            break;
        }
    }
    if (connected) {
        std::vector<int> v;
        for (int i = 0; i < nGPUs; i++) {
            v.push_back(i);
        }
        cliques.push_back(v);
        goto exit;
    }


    /***** Detect HCM topology *****/
    for (int i = 0; i < nGPUs; i++) {
        cliques.push_back(std::vector<int> { i });
    }

    /* find cliques of size nGPUs/2 */
    for (int k = 2; k <= nGPUs/2; k++) {
        std::vector<std::vector<int>> tmp;
        for (auto v : cliques) {
            for (int i = v.back() + 1; i < nGPUs; i++) {
                bool connected = true;
                for (auto j : v) {
                    if (adjacencyMatrix[i][j] == 0) {
                        connected = false;
                        break;
                    }
                }
                if (connected) {
                    std::vector<int> w = v;
                    w.push_back(i);
                    tmp.push_back(w);
                }
            }
        }

        cliques.clear();
        cliques = tmp;
        tmp.clear();
    }

    /* HCM has two cliques */
    if (cliques.size() != 2) {
        goto topo_not_found;
    }

    /* In HCM, each GPU is connected to (nGPUs/2 + 1) other GPUs */
    for (int i = 0; i < nGPUs; i++) {
        int count = 0;
        for (int j = 0; j < nGPUs; j++) {
            if (adjacencyMatrix[i][j]) {
                count++;
            }
        }
        if (count != (1 + nGPUs/2)) {
            goto topo_not_found;
        }
    }

    /* layout the cliques, so the two HCM cliques are ordered so that
     * the i'th GPU in one clique is connected to the i'th GPU in the
     * other clique.  If this is not possible, it's not HCM. */
    {
        std::vector<int> front = cliques.front();
        std::vector<int> back = cliques.back();
        std::vector<int> tmp;
        for (auto f : front) {
            /* each element in front should be connected to exactly
             * one element in back */
            for (auto b : back) {
                if (adjacencyMatrix[f][b]) {
                    tmp.push_back(b);
                    break;
                }
            }
        }
        assert(tmp.size() == nGPUs/2);
        cliques.pop_back();
        cliques.push_back(tmp);
    }

exit:
    for (int i = 0; i < nGPUs; i++) {
        int dev;
        CUDACHECK(cudaGetDevice(&dev));
        if (i == dev) {
            continue;
        } else if (adjacencyMatrix[comm->cudaDev][i] == 1) {
            cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
            if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
                CUDACHECK(e);
            }
        }
    }
    return ncclSuccess;

topo_not_found:
    cliques.clear();
    return ncclSuccess;
}

/*
 * This is our core function to detect the number of ranks in the same
 * virtual address space (i.e., threads).  The first communicator
 * handle that is created with a particular context ID creates and
 * enqueues an threadedRanksMd object in the threadedRanksMdList
 * queue.  If a new communicator handle is created with the same
 * context ID, it would point to the same threadedRanksMd object.  The
 * number of communicator handles pointing to the threadedRanksMd
 * object determines the number of threaded ranks in this address
 * space.
 */
ncclResult_t allocThreadedRanksMd(ncclComm_t comm, ncclUniqueId commId)
{
    threadedRanksMd *md;
    ncclResult_t ret = ncclSuccess;
    std::vector<std::vector<int>> gpuCliques;

    threadedRanksMdListMutex.lock();

    NCCLCHECKGOTO(topoDetect(comm, gpuCliques), ret, exit);

    /* allocate the threadedRanksMd structure or find an existing
     * one for this commId */
    md = nullptr;
    for (auto t : threadedRanksMdList) {
        if (t->commId == commId) {
            md = t;
            break;
        }
    }
    if (md == nullptr) {
        md = new threadedRanksMd(commId, gpuCliques);
        threadedRanksMdList.push_back(md);
    }

    md->insertRank(comm->rank, comm->cudaDev);

    comm->threadedRanks.md = md;
    comm->threadedRanks.barrierFlag = 0;
    comm->threadedRanks.barrierMboxId = 1;
    comm->threadedRanks.localMboxId = 1;
    CUDACHECK(cudaGetDeviceProperties(&comm->threadedRanks.devProp, comm->cudaDev));

    md->refCount++;

exit:
    threadedRanksMdListMutex.unlock();
    return ret;
}

/* This function decreases the refCount for the threadedRanksMd object
 * (one of the communicator pointing to it is getting freed).  If the
 * refCount reaches zero, that means no communicators are pointing to
 * it -- in that case, we can remove it from the
 * threadedRanksMdList. */
ncclResult_t freeThreadedRanksMd(threadedRanksMd *md, int rank)
{
    threadedRanksMdListMutex.lock();

    md->refCount--;

    if (md->refCount == 0) {
        auto mdIdx = std::remove(threadedRanksMdList.begin(), threadedRanksMdList.end(), md);
        threadedRanksMdList.erase(mdIdx, threadedRanksMdList.end());
        delete md;
    }

    threadedRanksMdListMutex.unlock();

    return ncclSuccess;
}
