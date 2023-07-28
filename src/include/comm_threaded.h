// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_COMM_THREADED_H_
#define NCCL_COMM_THREADED_H_

#include <vector>
#include <unordered_map>
#include <stdexcept>

int64_t ncclParamMaxThreadedRanks(void);
int64_t ncclParamThreadedAllreduceMaxTmpbufSize(void);

typedef enum {
    NCCL_THREADED_TOPO_TYPE__NVS,
    NCCL_THREADED_TOPO_TYPE__HCM,
    NCCL_THREADED_TOPO_TYPE__UNKNOWN,
} ncclThreadedTopoType_t;

/* each clique (direct NVLink connected group) of ranks */
class threadedRanksClique {
public:
    threadedRanksClique(std::vector<int> gpuClique) {
        this->gpus = gpuClique;

        /* mailbox for ranks to exchange their source buffer
         * information.  We create two copies and swap between the two
         * each time the collective is called.  This way, if one rank
         * is delayed in completing its work, we don't overwrite this
         * data in the next iteration. */
        for (int i = 0; i < 2; i++) {
            this->barrierMbox[i] = nullptr;
        }
    }

    ~threadedRanksClique() {
        for (auto it : this->rankToTmpbuf) {
            void *buf = it.second;
            CUDACHECKIGNORE(cudaFree(buf));
        }
        this->rankToTmpbuf.clear();

        for (int i = 0; i < 2; i++) {
            if (this->barrierMbox[i] != nullptr) {
                CUDACHECKIGNORE(cudaFree(this->barrierMbox[i]));
            }

            for (auto it : this->rankToLocalMbox[i]) {
                void *buf = it.second;
                CUDACHECKIGNORE(cudaFree(buf));
            }
            this->rankToLocalMbox[i].clear();
        }
    }

    void insertRank(int rank, int cudaDev) {
        auto it = std::find(this->gpus.begin(), this->gpus.end(), cudaDev);
        if (it == this->gpus.end()) {
            return;
        }

        if (this->rankToGpu.empty()) {
            int numBarrierPtrs = 2 * this->gpus.size() * this->gpus.size();

            for (int i = 0; i < 2; i++) {
                CUDACHECKIGNORE(cudaMalloc(&this->barrierMbox[i], numBarrierPtrs * sizeof(uintptr_t)));
                CUDACHECKIGNORE(cudaMemset(this->barrierMbox[i], 0, numBarrierPtrs * sizeof(uintptr_t)));
            }
        }

        this->rankToGpu[rank] = cudaDev;

        /* an extra temporary buffer for each rank; this would be used
         * if, for example, with IN_PLACE operations so we can perform
         * the reduction into the temporary buffer and then copy it
         * back to the user buffer. */
        void *buf;

        for (int i = 0; i < 2; i++) {
            CUDACHECKIGNORE(cudaMalloc(&buf, sizeof(uintptr_t)));
            CUDACHECKIGNORE(cudaMemset(buf, 0, sizeof(uintptr_t)));

            this->rankToLocalMbox[i][rank] = reinterpret_cast<uintptr_t *>(buf);
        }

        CUDACHECKIGNORE(cudaMalloc(&buf, ncclParamThreadedAllreduceMaxTmpbufSize()));
        this->rankToTmpbuf[rank] = buf;
    }

    /* mapping from rank to the GPU ID, temporary buffer, and local mbox */
    std::vector<int> gpus;
    std::unordered_map<int, int> rankToGpu;
    std::unordered_map<int, void *> rankToTmpbuf;
    std::unordered_map<int, uintptr_t *> rankToLocalMbox[2];
    uintptr_t *barrierMbox[2];
};

/* metadata for threaded ranks: contains the clique of GPUs (currently
 * all of the GPUs in the system), and a refcount of the number of
 * communicator handles in this address space that point to the same
 * commId */
class threadedRanksMd {
public:
    threadedRanksMd(ncclUniqueId commId, std::vector<std::vector<int>> gpuCliques) {
        this->commId = commId;

        /* add topology information */
        this->topoType = NCCL_THREADED_TOPO_TYPE__UNKNOWN;
        if (gpuCliques.size() == 1) {
            this->topoType = NCCL_THREADED_TOPO_TYPE__NVS;
        } else if (gpuCliques.size() == 2) {
            this->topoType = NCCL_THREADED_TOPO_TYPE__HCM;
        }

        for (auto it : gpuCliques) {
            threadedRanksClique *clique = new threadedRanksClique(it);
            this->cliques.push_back(clique);
        }

        this->refCount = 0;
    }

    ~threadedRanksMd() {
        for (auto c : cliques) {
            delete c;
        }
    }

    void insertRank(int rank, int cudaDev) {
        for (auto it : cliques) {
            it->insertRank(rank, cudaDev);
        }
    }

    ncclUniqueId commId;
    ncclThreadedTopoType_t topoType;
    std::vector<threadedRanksClique *> cliques;
    int refCount;
};

ncclResult_t allocThreadedRanksMd(ncclComm_t comm, ncclUniqueId commId);
ncclResult_t freeThreadedRanksMd(threadedRanksMd *md, int rank);

#endif
