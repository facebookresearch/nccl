// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef COLL_TRACE_H
#define COLL_TRACE_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
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

// Result data structure
struct CollTraceColl {
  uint64_t opCount;
  ncclInfo info;
  int64_t iteration;
  cudaStream_t stream;
  float latency{-1};
  // This is achieved by waiting for the start event. We can only guarantee
  // before this time point kernel has already started, but we cannot guarantee
  // kernel started exactly at this time point.
  std::chrono::time_point<std::chrono::high_resolution_clock> startTs{};

  // serialize the entry to a json format string
  std::string serialize(bool quoted = false);
  // flatten the entry to a plain string
  std::string toString();

 private:
  // internal helper function to retrieve the struct to a string map
  std::unordered_map<std::string, std::string> retrieveMap(bool quoted);
};

// Event data structure
struct CollTraceEvent {
  enum class EventType {
    COMM,
    // Wake up the worker thread. Currently used to wake up the worker thread
    // to dump information.
    WAKE_UP,
    TERMINATE
  };

  CollTraceColl coll;
  CudaEventPtr start;
  CudaEventPtr stop;
  EventType eventType = EventType::COMM;

  CollTraceEvent(EventType type) : eventType(type) {}
  CollTraceEvent() = default;
};

struct ncclComm;

// Class for colltrace
class CollTrace {
 public:
  CollTrace(ncclComm* comm);
  ~CollTrace();

  enum class CurrentCollState {
    PENDING,
    WAIT_START,
    IN_PROGRESS,
    DONE,
  };

  struct Dump {
    std::deque<CollTraceColl> pastColls;
    std::deque<CollTraceColl> pendingColls;
    std::unique_ptr<CollTraceColl> currentColl;
  };

 private:
  // Work queue data structure
  class EventQueue {
   private:
    std::deque<std::unique_ptr<CollTraceEvent>> queue_;
    std::condition_variable cv_;
    std::mutex mutex_;

   public:
    std::deque<CollTraceColl> dumpQueue() {
      std::deque<CollTraceColl> tmp{};
      {
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto& item : queue_) {
          // copy content of coll within each event
          tmp.emplace_back(item->coll);
        }
      }
      return tmp;
    }

    void push(std::unique_ptr<CollTraceEvent> item) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(item));
      }
      cv_.notify_one();
    }

    bool isEmpty() {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }

    std::unique_ptr<CollTraceEvent> waitPop() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (queue_.empty()) {
        cv_.wait(lock, [this] { return !queue_.empty(); });
      }
      std::unique_ptr<CollTraceEvent> item = std::move(queue_.front());
      queue_.pop_front();

      return item;
    }
  };

  // event pool
  class SharedPool {
   public:
    ~SharedPool(){};

    void add(CudaEventPtr item) {
      std::lock_guard<std::mutex> lock(mutex_);
      pool_.push_back(std::move(item));
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
      pool_.pop_front();
      return tmp;
    }

   private:
    std::deque<CudaEventPtr> pool_;
    mutable std::mutex mutex_;
  };

 private:
  // cudaEvent pool to avoid cudaEvent destory during run and enable reuse.
  SharedPool cudaEventPool_;
  EventQueue eventQueue_;
  // Using shared ptr to avoid race condition when worker thread is exiting
  // while we are trying to dump results in collDump.
  std::shared_ptr<CollTraceEvent> curEvent_;
  std::atomic<CurrentCollState> curCollState_{CurrentCollState::PENDING};
  std::deque<std::unique_ptr<CollTraceColl>> pastColls_;
  // Lock changes from worker thread to curEvent_, eventQueue_ and pastColls_
  std::mutex workerMutex_;

  // For testing purpose
  std::atomic<bool> waitingForQueueEmpty_;
  std::mutex waitQueueEmptyMutex_;
  std::condition_variable waitQueueEmptyCv_;

  struct ncclComm* comm_{nullptr};
  std::thread profilingWorkerThread_;

  bool logCollSample(CollTraceColl& coll);

 public:
  enum Features {
    VERBOSE = 1,
    FILE = 2,
    FB_IO_DURING_RUN = 4,
    ONLINE_TUNING = 8,
    TRACE = 16,
  };
  int features{0}; // bitwise OR of Features

  CollTrace::Dump dump();

  // Internal function called in collTraceThreadFn for worker thread to access
  // private members
  void* collTraceThreadFnImpl();
  // Wrapper function called by worker thread
  static void* collTraceThreadFn(CollTrace* collTrace);

  // Create a CollTraceEvent object and assign cuda events from pool
  std::unique_ptr<CollTraceEvent> createEvent();

  void enqueueEvent(std::unique_ptr<CollTraceEvent> event);

  void waitForWorkerFinishQueue();

  // Dump results to file. File path is specified by NCCL_COLLTRACE_DIR
  // Return true if dumping is successful, otherwise false.
  bool dumpResultsToFile();
};

ncclResult_t collTraceInit(ncclComm* comm);
ncclResult_t collTraceDestroy(ncclComm* comm);

#define COLLTRACE_INFO_COPY(comm, plan, aggInfoRef)          \
  do {                                                    \
    if (comm->collTrace && aggInfoRef.count > 0) {           \
      memcpy(&plan->aggInfo, &aggInfoRef, sizeof(ncclInfo)); \
    }                                                     \
  } while (0)

#define COLLTRACE_P2P_APPEND(comm, plan, info)                                \
  do {                                                                        \
    if (comm->collTrace) {                                                    \
      if (info.coll == ncclFuncSend && info.count > 0) {                      \
        /* addP2pToPlan already converts info.count to bytes with ncclInt8 */ \
        plan->nSendBytes += info.count;                                       \
      } else {                                                                \
        plan->nRecvBytes += info.count;                                       \
      }                                                                       \
    }                                                                         \
  } while (0)

#define COLLTRACE_ACQUIRE_EVENT(comm, plan)                                           \
  std::unique_ptr<CollTraceEvent> event = nullptr;                                    \
  do {                                                                                \
    if (comm->collTrace) {                                                            \
      if (plan->aggInfo.count > 0 && (plan->nSendBytes || plan->nRecvBytes)) {        \
        WARN(                                                                         \
            "COLLTRACE: do not support grouped collective and p2p. Skip this plan."); \
      } else {                                                                        \
        event = comm->collTrace->createEvent();                                       \
        if (!event) {                                                                 \
          return ncclInternalError; /*Event init failed*/                             \
        }                                                                             \
        event->coll.iteration = ncclFbGetTrainerIteration();                          \
      }                                                                               \
    }                                                                                 \
  } while (0)

#define COLLTRACE_RECORD_START_EVENT(comm, launchStream)            \
  do {                                                              \
    if (comm->collTrace && event) {                                 \
      CUDACHECK(cudaEventRecord(event->start.get(), launchStream)); \
    }                                                               \
  } while (0)

#define COLLTRACE_RECORD_END_EVENT(comm, plan, launchStream)             \
  do {                                                                   \
    if (comm->collTrace && event) {                                      \
      CUDACHECK(cudaEventRecord(event->stop.get(), launchStream));       \
      event->coll.opCount = comm->opCount;                               \
      /* single or grouped collective */                                 \
      if (plan->aggInfo.count > 0) {                                     \
        event->coll.info = plan->aggInfo;                                \
      } else { /*groupd p2p */                                           \
        if (plan->nSendBytes && plan->nRecvBytes) {                      \
          event->coll.info.opName = "SendRecv";                          \
          event->coll.info.coll = ncclFuncSendRecv;                      \
        } else if (plan->nSendBytes) {                                   \
          event->coll.info.opName = "Send";                              \
          event->coll.info.coll = ncclFuncSend;                          \
        } else if (plan->nRecvBytes) {                                   \
          event->coll.info.opName = "Recv";                              \
          event->coll.info.coll = ncclFuncRecv;                          \
        }                                                                \
        event->coll.info.sendbuff = event->coll.info.recvbuff = nullptr; \
        event->coll.info.count = plan->nSendBytes + plan->nRecvBytes;    \
        event->coll.info.datatype = ncclInt8;                            \
        event->coll.info.root = -1;                                      \
        event->coll.info.op = ncclSum;                                   \
        /* FIXME: cannot record protocol for sendrecvs since a grouped   \
         * sendrecv may contain multiple protocols */                    \
        event->coll.info.algorithm = -1;                                 \
        event->coll.info.protocol = -1;                                  \
        event->coll.info.nChannels = plan->channelCount;                 \
        event->coll.info.nThreads = plan->threadPerBlock;                \
      }                                                                  \
      event->coll.stream = launchStream;                                 \
      comm->collTrace->enqueueEvent(std::move(event));                   \
    }                                                                    \
  } while (0)

#endif
