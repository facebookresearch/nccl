// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef COLL_TRACE_H
#define COLL_TRACE_H

#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include "FbInternal.h"
#include "checks.h"
#include "debug.h"
#include "info.h"
#include "nccl.h"
#include "nccl_common.h"

// CUDA event pointer w/ deleter
struct CudaEventDeleter {
  void operator()(cudaEvent_t e) {
    CUDACHECKIGNORE(cudaEventDestroy(e));
  }
};
using CudaEventPtr = std::unique_ptr<
    std::pointer_traits<cudaEvent_t>::element_type,
    CudaEventDeleter>;

enum class EventState {
  PENDING,
  RUNNING,
  FINISHED,
};

// Event data structure
struct EventInfo {
  uint64_t opCount;
  ncclInfo info;
  int64_t iteration;
  CudaEventPtr start;
  CudaEventPtr stop;
  cudaStream_t stream;

  EventInfo() = default;
  EventInfo(const EventInfo&) = delete;
  EventInfo& operator=(const EventInfo&) = delete;
};

// Result data structure
struct ResultInfo {
  uint64_t opCount;
  ncclInfo info;
  cudaStream_t stream;
  int64_t iteration;
  float latency;
};

// event pool
class SharedPool {
 public:
  ~SharedPool(){};

  void add(CudaEventPtr item) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push(std::move(item));
  }

  CudaEventPtr takeOne() {
    std::lock_guard<std::mutex> lock(mutex_);

    // no event available, create new one
    if (pool_.empty()) {
      cudaEvent_t newEvent = nullptr;
      CUDACHECKIGNORE(cudaEventCreate(&newEvent));
      CudaEventPtr item(newEvent);
      return item;
    }

    // reuse existing event
    CudaEventPtr tmp = std::move(pool_.front());
    pool_.pop();
    return tmp;
  }

 private:
  std::queue<CudaEventPtr> pool_;
  mutable std::mutex mutex_;
};

struct ncclComm;

// Class for colltrace
class CollTrace {
 public:
  CollTrace(ncclComm* comm);
  ~CollTrace();

  struct CollTraceDump {
    std::list<ResultInfo> pastResults;
    std::queue<std::unique_ptr<EventInfo>> pendingEvents;
    std::shared_ptr<EventInfo> currentEvent;
    EventState currentEventState {EventState::PENDING};
  };

 private:
  // Work queue data structure
  class EventQueue {
   private:
    std::queue<std::unique_ptr<EventInfo>> queue_;
    std::mutex mutex_;

   public:
    std::queue<std::unique_ptr<EventInfo>> dumpQueue() {
      std::queue<std::unique_ptr<EventInfo>> tmp {};
      {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.swap(tmp);
      }
      return tmp;
    }
    void push(std::unique_ptr<EventInfo> item) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(item));
    }
    bool isEmpty() {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }
    std::unique_ptr<EventInfo> tryPop() {
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock.owns_lock()) {
        return NULL;
      }
      if (queue_.empty()) {
        return NULL;
      }
      std::unique_ptr<EventInfo> item = std::move(queue_.front());
      queue_.pop();

      return item;
    }
  };

  void outputResults();

 private:
  // cudaEvent pool to avoid cudaEvent destory during run and enable reuse.
  SharedPool eventPool_;
  EventQueue eventQueue_;
  // Using shared ptr to avoid race condition when worker thread is exiting
  // while we are trying to dump results in collDump.
  std::shared_ptr<EventInfo> curEvent_;
  std::atomic<EventState> curEventState_{EventState::PENDING};
  std::list<ResultInfo> results_;
  std::atomic<bool> workerThreadExitSignal_{false};

  struct ncclComm* comm_{nullptr};
  std::thread profilingWorkerThread_;

 public:
  enum Features {
    VERBOSE = 1,
    FILE = 2,
    FB_IO_DURING_RUN = 4,
    ONLINE_TUNING = 8,
    TRACE = 16,
  };
  int features{0}; // bitwise OR of Features

  CollTraceDump dumpTrace();

  // Internal function called in collTraceThreadFn for worker thread to access
  // private members
  void* collTraceThreadFnImpl();
  // Wrapper function called by worker thread
  static void* collTraceThreadFn(CollTrace* collTrace);

  // Get free EventInfo object from pool
  std::unique_ptr<EventInfo> getEventFromPool();

  void enqueueEvent(std::unique_ptr<EventInfo> eventInfo);
};

ncclResult_t collTraceInit(ncclComm* comm);
ncclResult_t collTraceDestroy(ncclComm* comm);

#define COLLTRACE_INFO_COPY(comm, plan, aggInfo)          \
  do {                                                    \
    if (comm->collTrace && aggInfo.count > 0) {           \
      memcpy(&plan->aggInfo, &aggInfo, sizeof(ncclInfo)); \
    }                                                     \
  } while (0)

#define COLLTRACE_P2P_APPEND(comm, plan, info)           \
  do {                                                   \
    if (comm->collTrace) {                               \
      if (info.coll == ncclFuncSend && info.count > 0) { \
        plan->nSendBytes += info.count;                  \
      } else {                                           \
        plan->nRecvBytes += info.count;                  \
      }                                                  \
    }                                                    \
  } while (0)

#define COLLTRACE_ACQUIRE_EVENT(comm, plan)                                           \
  std::unique_ptr<EventInfo> eventInfo = nullptr;                                     \
  do {                                                                                \
    if (comm->collTrace) {                                                            \
      if (plan->aggInfo.count > 0 && (plan->nSendBytes || plan->nRecvBytes)) {        \
        WARN(                                                                         \
            "COLLTRACE: do not support grouped collective and p2p. Skip this plan."); \
      } else {                                                                        \
        eventInfo = comm->collTrace->getEventFromPool();                              \
        if (!eventInfo) {                                                             \
          return ncclInternalError; /*Event init failed*/                             \
        }                                                                             \
        eventInfo->iteration = ncclFbGetTrainerIteration();                           \
      }                                                                               \
    }                                                                                 \
  } while (0)

#define COLLTRACE_RECORD_START_EVENT(comm, launchStream)                \
  do {                                                                  \
    if (comm->collTrace && eventInfo) {                                 \
      CUDACHECK(cudaEventRecord(eventInfo->start.get(), launchStream)); \
    }                                                                   \
  } while (0)

#define COLLTRACE_RECORD_END_EVENT(comm, plan, launchStream)           \
  do {                                                                 \
    if (comm->collTrace && eventInfo) {                                \
      CUDACHECK(cudaEventRecord(eventInfo->stop.get(), launchStream)); \
      eventInfo->opCount = comm->opCount;                              \
      /* single or grouped collective */                               \
      if (plan->aggInfo.count > 0) {                                   \
        eventInfo->info = plan->aggInfo;                               \
      } else { /*groupd p2p */                                         \
        if (plan->nSendBytes && plan->nRecvBytes) {                    \
          eventInfo->info.opName = "SendRecv";                         \
          eventInfo->info.coll = ncclFuncSendRecv;                     \
        } else if (plan->nSendBytes) {                                 \
          eventInfo->info.opName = "Send";                             \
          eventInfo->info.coll = ncclFuncSend;                         \
        } else if (plan->nRecvBytes) {                                 \
          eventInfo->info.opName = "Recv";                             \
          eventInfo->info.coll = ncclFuncRecv;                         \
        }                                                              \
        eventInfo->info.sendbuff = eventInfo->info.recvbuff = nullptr; \
        eventInfo->info.count = plan->nSendBytes + plan->nRecvBytes;   \
        eventInfo->info.datatype = ncclInt8;                           \
        eventInfo->info.root = -1;                                     \
        eventInfo->info.op = ncclSum;                                  \
        /* FIXME: cannot record protocol for sendrecvs since a grouped \
         * sendrecv may contain multiple protocols */                  \
        eventInfo->info.algorithm = -1;                                \
        eventInfo->info.protocol = -1;                                 \
        eventInfo->info.nChannels = plan->channelCount;                \
        eventInfo->info.nThreads = plan->threadPerBlock;               \
      }                                                                \
      eventInfo->stream = launchStream;                                \
      comm->collTrace->enqueueEvent(std::move(eventInfo));             \
    }                                                                  \
  } while (0)

#endif
