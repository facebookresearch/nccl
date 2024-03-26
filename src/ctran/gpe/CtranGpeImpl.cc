// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranGpeImpl.h"
#include <nccl.h>
#include <cassert>
#include <iostream>
#include <new>
#include <stdexcept>
#include "CtranChecks.h"
#include "CtranGpe.h"
#include "CtranGpeKernel.h"
#include "checks.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===
 - name        : NCCL_CTRAN_NUM_KERNEL_P2PELEMS
   type        : int
   default     : 65536
   description : |-
     Size of kernel p2p elements pre-allocated for each communicator.
     Used to pass variable number of p2p operations to the kernel.
     Each p2p element is allocated from page-locked memory on the host.
=== END_NCCL_CVAR_INFO_BLOCK ===
*/

CtranGpe::Impl::Impl() {
  CUDACHECKTHROW(
      cudaHostAlloc(&this->kernelFlag, sizeof(int), cudaHostAllocDefault));
  *(this->kernelFlag) = UNSET;

  this->kernelP2pElemPool = std::unique_ptr<KernelP2pElemPool>(
      new KernelP2pElemPool(NCCL_CTRAN_NUM_KERNEL_P2PELEMS));
  return;
}

CtranGpe::Impl::~Impl() {
  CUDACHECKIGNORE(cudaFreeHost(this->kernelFlag));

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

ncclResult_t CtranGpe::Impl::submit(
    CtranGpeCmd::TypeEnum type,
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    KernelConfig& kernelConfig,
    const void* ncclKernel) {
  // Set first kernel argument as kernelFlag if GPE op is not empty.
  // Check it before passing opGroup to cmd
  std::vector<void*> kernelArgs;
  void* kernelFlag = opGroup.size() ? this->kernelFlag : nullptr;
  kernelArgs.push_back(&kernelFlag);

  // Enqueue op to gpeThread if any op is appended
  if (!opGroup.empty()) {
    struct CtranGpeCmd* cmd = new struct CtranGpeCmd;
    cmd->type = type;

    if (type == CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE) {
      cmd->coll.opGroup = std::move(opGroup);
      cmd->coll.func = func;
    }

    this->m.lock();
    cmdQueue.push(cmd);
    this->m.unlock();
    c.notify_one();
  }

  // Enqueue the kernel with arguments.  It will not start till all other
  // operations on this stream have completed
  dim3 grid = {kernelConfig.numBlocks, 1, 1};
  dim3 blocks = {kernelConfig.numThreads, 1, 1};

  // Specify collective arguments
  kernelArgs.push_back((void*)&kernelConfig.args.devState_d);
  kernelArgs.push_back((void*)&kernelConfig.args.collective);

  CUDACHECK(cudaLaunchKernel(
      ncclKernel, grid, blocks, kernelArgs.data(), 0, kernelConfig.stream));

  return ncclSuccess;
}

ncclResult_t CtranGpe::Impl::terminate() {
  struct CtranGpeCmd* cmd = new struct CtranGpeCmd;
  cmd->type = CtranGpeCmd::TypeEnum::TERMINATE;

  this->m.lock();
  cmdQueue.push(cmd);
  this->m.unlock();
  c.notify_one();

  return ncclSuccess;
}

void CtranGpe::Impl::gpeThreadFn(CtranGpe::Impl* pimpl, int cudaDev) {
  NCCL_NAMED_THREAD_START("CTranGPE");

  CUDACHECKTHROW(cudaSetDevice(cudaDev));

  while (1) {
    CtranGpeCmd* cmd;

    {
      std::unique_lock<std::mutex> lk(pimpl->m);
      pimpl->c.wait(lk, [&] { return !pimpl->cmdQueue.empty(); });

      cmd = pimpl->cmdQueue.front();
      pimpl->cmdQueue.pop();
    }

    if (cmd->type == CtranGpeCmd::TypeEnum::TERMINATE) {
      return;
    }

    /* wait for the kernel to launch */
    volatile int* flag_d = pimpl->kernelFlag;
    while (*flag_d != KERNEL_STARTED)
      ;

    /* run collective */
    NCCLCHECKTHROW(cmd->coll.func(std::move(cmd->coll.opGroup)));

    /* stop kernel */
    *flag_d = KERNEL_TERMINATE;

    delete cmd;
  }
  return;
}

KernelP2pElemPool::KernelP2pElemPool(size_t capacity) : capacity_(capacity) {
  CUDACHECKTHROW(cudaHostAlloc(
      &this->memPtr_,
      this->capacity_ * sizeof(struct KernelP2pElem),
      cudaHostAllocDefault));

  for (int i = 0; i < capacity_; ++i) {
    KernelP2pElem* workElem =
        reinterpret_cast<KernelP2pElem*>(this->memPtr_) + i;
    this->resetWorkElem(workElem);
    this->freeWorkElems_.push(workElem);
  }
  return;
}

KernelP2pElemPool::~KernelP2pElemPool() {
  this->reclaim();
  if (this->inuseWorkElems_.size()) {
    WARN(
        "CTRAN-GPE: Internal KernelP2pElem pool has %ld inuse elements",
        this->inuseWorkElems_.size());
  }
  CUDACHECKIGNORE(cudaFreeHost(this->memPtr_));

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

void KernelP2pElemPool::resetWorkElem(KernelP2pElem* workElem) {
  workElem->next = nullptr;
  workElem->displ = 0;
  workElem->count = 0;
  workElem->peerRank = -1;
  workElem->ngroups = 0;
  for (int i = 0; i < CTRAN_ALGO_MAX_THREAD_BLOCKS; ++i) {
    workElem->inuse[i] = false;
  }
}

size_t KernelP2pElemPool::size() {
  return this->freeWorkElems_.size();
}

KernelP2pElem* KernelP2pElemPool::pop(int ngroups) {
  if (ngroups > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    WARN(
        "CTRAN-GPE: ngroups %d exceeds max thread blocks %d",
        ngroups,
        CTRAN_ALGO_MAX_THREAD_BLOCKS);
    return nullptr;
  }

  KernelP2pElem* workElem = this->freeWorkElems_.top();
  this->freeWorkElems_.pop();
  workElem->ngroups = ngroups;
  for (int i = 0; i < ngroups; i++) {
    workElem->inuse[i] = true;
  }

  this->inuseWorkElems_.push_back(workElem);
  return workElem;
}

void KernelP2pElemPool::reclaim() {
  // Iterate inuseWorkElems_ and reclaim any unused workElems
  auto it = this->inuseWorkElems_.begin();
  while (it != this->inuseWorkElems_.end()) {
    auto workElem = *it;

    // Each kernel thread block resets inuse flag when finished
    assert(workElem->ngroups > 0);
    bool anyInuse = false;
    for (int i = 0; i < workElem->ngroups && !anyInuse; i++) {
      anyInuse |= workElem->inuse[i];
    }

    // If no more thread block is using, reclaim the workElem
    if (!anyInuse) {
      it = this->inuseWorkElems_.erase(it);

      this->resetWorkElem(workElem);
      this->freeWorkElems_.push(workElem);
    } else {
      it++;
    }
  }
}
