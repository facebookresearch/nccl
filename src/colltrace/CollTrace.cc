// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "CollTrace.h"
#include "ExtChecks.h"
#include "ExtUtils.h"
#include "FbInternal.h"
#include "bootstrap.h"
#include "comm.h"
#include "nccl.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_COLLTRACE
   type        : stringlist
   default     :
   description : |-
     Enable collective trace collection by leveraging cuda events
     and a background thread. Valid options are comma separated list
     of the following features. Leave empty to disable all features.
     verbose - print every completed event as NCCL INFO log. Mostly for debug.
     trace - Just enable collective trace.
     file    - dump traced events to file at communicator destory. Also see
            NCCL_COLLTRACE_DIR.
     online_tuning - enable online tuning
     (Other FB internal featuers are not listed)

 - name        : NCCL_COLLTRACE_DIR
   type        : string
   default     : ""
   description : |-
     Directory for CollTrace to dump.
     Can be either local or FB internal remote URL.

 - name        : NCCL_COLLTRACE_RECORD_MAX
   type        : int
   default     : 20
   description : |-
     Maximum amount of past collectives CollTrace will record.
     If the amount of collective exceeds this value, the oldest one will be
     dropped. Set the value to -1 will make CollTrace record all collectives.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

static std::unordered_map<ncclPattern_t, std::string> ncclPatternStr = {
    {ncclPatternRing, "Ring"},
    {ncclPatternRingTwice, "RingTwice"},
    {ncclPatternPipelineFrom, "PipelineFrom"},
    {ncclPatternPipelineTo, "PipelineTo"},
    {ncclPatternTreeUp, "TreeUp"},
    {ncclPatternTreeDown, "TreeDown"},
    {ncclPatternTreeUpDown, "TreeUpDown"},
    {ncclPatternCollnetChain, "CollnetChain"},
    {ncclPatternCollnetDirect, "CollnetDirect"},
    {ncclPatternNvls, "Nvls"},
    {ncclPatternNvlsTree, "NvlsTree"},
    {ncclPatternSend, "Send"},
    {ncclPatternRecv, "Recv"}};

CollTrace::CollTrace(ncclComm* comm) : comm_(comm) {
  std::vector<std::string> enabledFeatures;
  if (!NCCL_COLLTRACE.empty()) {
    for (auto& f : NCCL_COLLTRACE) {
      if (f == "verbose") {
        features |= CollTrace::Features::VERBOSE;
        enabledFeatures.push_back(f);
      } else if (f == "file") {
        features |= CollTrace::Features::FILE;
        enabledFeatures.push_back(f);
      } else if (f == "online_tuning") {
        features |= CollTrace::Features::ONLINE_TUNING;
        enabledFeatures.push_back(f);
      } else if (f == "fb") {
        features |= CollTrace::Features::FB_IO_DURING_RUN;
        enabledFeatures.push_back(f);
      } else if (f == "trace") {
        features |= CollTrace::Features::TRACE;
        enabledFeatures.push_back(f);
      }
    }
  }

  // create worker thread
  profilingWorkerThread_ = std::thread{collTraceThreadFn, this};

  std::string enabledFeaturesStr = vecToStr(enabledFeatures);
  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx rank %d enabled features: %s - Init COMPLETE",
      comm,
      comm->commHash,
      comm->rank,
      enabledFeaturesStr.c_str());
}

CollTrace::~CollTrace() {
  try {
    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lx rank %d - Destroy START",
        comm_,
        comm_->commHash,
        comm_->rank);

    eventQueue_.push(std::unique_ptr<CollTraceEvent>(
        new CollTraceEvent(CollTraceEvent::EventType::TERMINATE)));
    if (profilingWorkerThread_.joinable()) {
      profilingWorkerThread_.join();
    }

    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lx rank %d - Destroy COMPLETE",
        comm_,
        comm_->commHash,
        comm_->rank);
  } catch (const std::exception& e) {
    WARN(
        "COLLTRACE: comm %p commHash %lx rank %d - Destroy FAILED: %s",
        comm_,
        comm_->commHash,
        comm_->rank,
        e.what());
  } catch (...) {
    WARN(
        "COLLTRACE: comm %p commHash %lx rank %d - Destroy FAILED: Unkown exception",
        comm_,
        comm_->commHash,
        comm_->rank);
  }
}

bool CollTrace::dumpResultsToFile() {
  if (features & CollTrace::Features::FILE && !NCCL_COLLTRACE_DIR.empty()) {
    // In case outputResults is called by user when worker thread is still
    // running
    std::lock_guard<std::mutex> lock(workerMutex_);

    std::vector<std::string> serializedResults(pastColls_.size());

    for (int i = 0; i < pastColls_.size(); i++) {
      auto& result = pastColls_[i];
      serializedResults[i] = result->serialize(true);
    }
    std::string contents = serializeVec(serializedResults);

    const std::string fileName = NCCL_COLLTRACE_DIR + "/comm" +
        hashToHexStr(comm_->commHash) + "_rank" + std::to_string(comm_->rank) +
        "_online.json";
    INFO(
        NCCL_ALL,
        "COLLTRACE: rank %d writing %lu online profiler data to : %s",
        comm_->rank,
        pastColls_.size(),
        fileName.c_str());

    if (ncclIsFbPath(fileName)) {
      ncclFbUpload(contents, fileName);
    } else {
      std::ofstream f(fileName);
      f << contents;
      f.close();
    }
    return true;
  }
  return false;
}

CollTrace::Dump CollTrace::dump() {
  std::lock_guard<std::mutex> lock(workerMutex_);
  CollTrace::Dump dump{};

  if (curCollState_ == CurrentCollState::IN_PROGRESS ||
      curCollState_ == CurrentCollState::WAIT_START) {
    // copy contents
    dump.currentColl =
        std::unique_ptr<CollTraceColl>(new CollTraceColl(curEvent_->coll));
  }

  dump.pendingColls = eventQueue_.dumpQueue();

  for (auto& result : pastColls_) {
    // copy contents
    dump.pastColls.emplace_back(*result);
  }
  return dump;
}

void* CollTrace::collTraceThreadFn(CollTrace* ct) {
  NCCL_NAMED_THREAD_START("CollTrace");
  return ct->collTraceThreadFnImpl();
}

void* CollTrace::collTraceThreadFnImpl() {
  CUDACHECKTHROW(cudaSetDevice(comm_->cudaDev));

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx rank %d - worker thread STARTED",
      comm_,
      comm_->commHash,
      comm_->rank);

  while (true) {
    curCollState_ = CurrentCollState::PENDING;
    curEvent_ = nullptr;

    // For testing purpose only. During testing, we want to ensure the worker
    // thread reached a steady state before dumping so that the trace dump
    // result is predictable. Otherwise the test can be flaky.
    if (waitingForQueueEmpty_ && eventQueue_.isEmpty()) {
      {
        std::unique_lock<std::mutex> lock(waitQueueEmptyMutex_);
        waitingForQueueEmpty_ = false;
      }
      waitQueueEmptyCv_.notify_all();
    }

    // We intentionally didn't hold the event queue lock till curEvent is
    // updated. That will potentially create deadlock.
    // Downside of current approach is we might miss one pending event in the
    // dump in very rare occasion. But since the worker thread haven't started
    // to wait for the event, it should be fine.
    {
      auto tmp_event = eventQueue_.waitPop();
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_ = std::move(tmp_event);
    }

    if (curEvent_->eventType == CollTraceEvent::EventType::TERMINATE) {
      break;
    } else if (curEvent_->eventType == CollTraceEvent::EventType::WAKE_UP) {
      continue;
    }
    curCollState_ = CurrentCollState::WAIT_START;
    cudaError_t res = cudaEventSynchronize(curEvent_->start.get());
    {
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_->coll.startTs = std::chrono::high_resolution_clock::now();
    }
    curCollState_ = CurrentCollState::IN_PROGRESS;
    res = cudaEventSynchronize(curEvent_->stop.get());
    curCollState_ = CurrentCollState::DONE;
    float latency = -1;

    if (res == cudaSuccess) {
      res = cudaEventElapsedTime(
          &latency, curEvent_->start.get(), curEvent_->stop.get());
    }

    {
      // Bracket to ensure result not getting accessed after it is moved.
      // Also for release lock_guard.
      auto result = std::unique_ptr<CollTraceColl>(new CollTraceColl());
      *result = curEvent_->coll;
      result->latency = (res == cudaSuccess) ? latency : -1;

      if (features & CollTrace::Features::VERBOSE) {
        INFO(NCCL_COLL, "COLLTRACE: %s", result->toString().c_str());
      }

      // FIXME: cannot record protocol for sendrecvs since a grouped sendrecv
      // may contain multiple protocols
      if (features & CollTrace::Features::FB_IO_DURING_RUN) {
        logCollSample(*result);
      }

      std::lock_guard<std::mutex> lock(workerMutex_);
      pastColls_.push_back(std::move(result));
      if (NCCL_COLLTRACE_RECORD_MAX >= 0 &&
          pastColls_.size() > NCCL_COLLTRACE_RECORD_MAX) {
        pastColls_.pop_front();
      }
    }

    // Free the event objects
    cudaEventPool_.add(std::move(curEvent_->start));
    cudaEventPool_.add(std::move(curEvent_->stop));

    // FIXME: we should revisit bootstrapAllGather() here since commAbort
    // may be called either on local rank or a remote rank causing socket
    // failure
    if (comm_->tuner != NULL && features & CollTrace::Features::ONLINE_TUNING) {
      // Online tuning - average latencies across ranks & send to tuner
      float* latencies = NULL;
      NCCLCHECKIGNORE(
          ncclCalloc(&latencies, curEvent_->coll.info.comm->nRanks));
      latencies[curEvent_->coll.info.comm->rank] = latency;
      NCCLCHECKIGNORE(bootstrapAllGather(
          curEvent_->coll.info.comm->bootstrap, latencies, sizeof(float)));
      float sum = 0.0;
      for (int i = 0; i < curEvent_->coll.info.comm->nRanks; i++) {
        sum += latencies[i];
      }

      free(latencies);
      sum /= (float)curEvent_->coll.info.comm->nRanks;

      curEvent_->coll.info.comm->tuner->addOnlineResult(
          curEvent_->coll.info.coll,
          curEvent_->coll.info.count *
              ncclTypeSize(curEvent_->coll.info.datatype),
          curEvent_->coll.iteration,
          sum,
          curEvent_->coll.info.algorithm,
          curEvent_->coll.info.protocol,
          curEvent_->coll.info.nChannels,
          curEvent_->coll.info.nThreads);
    }
  }

  dumpResultsToFile();

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx rank %d - worker thread TERMINATE",
      comm_,
      comm_->commHash,
      comm_->rank);
  return nullptr;
}

std::unique_ptr<CollTraceEvent> CollTrace::createEvent() {
  std::unique_ptr<CollTraceEvent> eventInfo(new CollTraceEvent);
  eventInfo->start = cudaEventPool_.takeOne();
  eventInfo->stop = cudaEventPool_.takeOne();
  if (!eventInfo->start || !eventInfo->stop) {
    std::unique_ptr<CollTraceEvent> nullCollTraceEvent(nullptr);
    return nullCollTraceEvent;
  }
  return eventInfo;
}

void CollTrace::enqueueEvent(std::unique_ptr<CollTraceEvent> event) {
  eventQueue_.push(std::move(event));
}

void CollTrace::waitForWorkerFinishQueue() {
  std::unique_lock<std::mutex> waitLock(waitQueueEmptyMutex_);
  waitingForQueueEmpty_ = true;
  eventQueue_.push(std::unique_ptr<CollTraceEvent>(
      new CollTraceEvent(CollTraceEvent::EventType::WAKE_UP)));
  waitQueueEmptyCv_.wait(waitLock, [this] { return !waitingForQueueEmpty_; });
}

bool CollTrace::logCollSample(CollTraceColl& coll) {
  std::unordered_map<std::string, std::string> normalMap;
  std::unordered_map<std::string, int64_t> intMap;

  intMap["rank"] = coll.info.comm->rank;
  intMap["commHash"] = coll.info.comm->commHash;
  intMap["opCount"] = coll.opCount;
  intMap["stream"] = reinterpret_cast<int64_t>(coll.stream);
  intMap["iteration"] = coll.iteration;
  normalMap["opName"] = coll.info.opName;
  intMap["sendbuff"] = reinterpret_cast<int64_t>(coll.info.sendbuff);
  intMap["recvbuff"] = reinterpret_cast<int64_t>(coll.info.recvbuff);
  intMap["count"] = coll.info.count;
  normalMap["dataType"] = getDatatypeStr(coll.info.datatype);
  normalMap["redOp"] = getRedOpStr(coll.info.op);
  intMap["root"] = coll.info.root;
  normalMap["algorithm"] = ncclAlgoStr[coll.info.algorithm];
  normalMap["protocol"] = ncclProtoStr[coll.info.protocol];
  intMap["nChannels"] = coll.info.nChannels;
  intMap["nThreads"] = coll.info.nThreads;
  intMap["latency (microseconds)"] = 1000 * coll.latency;
  intMap["startTs"] = std::chrono::duration_cast<std::chrono::microseconds>(
                          coll.startTs.time_since_epoch())
                          .count();
  ncclFbLogSample("nccl_coll_trace", normalMap, intMap);
  return true;
}

static std::vector<std::string> collKeys = {
    "opCount",
    "opName",
    "sendbuff",
    "recvbuff",
    "count",
    "datatype",
    "redOp",
    "root",
    "algorithm",
    "protocol",
    "pattern",
    "channelId",
    "nChannels",
    "nThreads",
    "latencyUs",
    "startTs"};

std::unordered_map<std::string, std::string> CollTraceColl::retrieveMap(
    bool quoted) {
  std::unordered_map<std::string, std::string> infoMap;
  std::string algoStr =
      info.algorithm >= 0 ? ncclAlgoStr[info.algorithm] : "N/A";
  std::string protoStr =
      info.protocol >= 0 ? ncclProtoStr[info.protocol] : "N/A";
  std::string patternStr =
      ncclPatternStr.count(info.pattern) ? ncclPatternStr[info.pattern] : "N/A";
  std::string datatypeStr = getDatatypeStr(info.datatype);
  std::string redOpStr = getRedOpStr(info.op);

  infoMap["opCount"] = std::to_string(opCount);
  infoMap["opName"] = quoted ? toQuotedString(info.opName) : info.opName;
  infoMap["sendbuff"] =
      std::to_string(reinterpret_cast<uint64_t>(info.sendbuff));
  infoMap["recvbuff"] =
      std::to_string(reinterpret_cast<uint64_t>(info.recvbuff));
  infoMap["count"] = std::to_string(info.count);
  infoMap["datatype"] = quoted ? toQuotedString(datatypeStr) : datatypeStr;
  infoMap["redOp"] = quoted ? toQuotedString(redOpStr) : redOpStr;
  infoMap["root"] = std::to_string(info.root);
  infoMap["algorithm"] = quoted ? toQuotedString(algoStr) : algoStr;
  infoMap["protocol"] = quoted ? toQuotedString(protoStr) : protoStr;
  infoMap["pattern"] = quoted ? toQuotedString(patternStr) : patternStr;
  infoMap["channelId"] = std::to_string(info.channelId);
  infoMap["nChannels"] = std::to_string(info.nChannels);
  infoMap["nThreads"] = std::to_string(info.nThreads);
  infoMap["latencyUs"] = std::to_string(latency < 0 ? -1 : latency * 1000);
  infoMap["startTs"] =
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                          startTs.time_since_epoch())
                          .count());
  return infoMap;
}

std::string CollTraceColl::serialize(bool quoted) {
  std::unordered_map<std::string, std::string> infoMap = retrieveMap(quoted);
  return serializeMap(collKeys, infoMap, quoted);
}

std::string CollTraceColl::toString() {
  std::unordered_map<std::string, std::string> infoMap = retrieveMap(false);
  // Convert integer sendbuff and recvbuff to hexadecimal only for display
  infoMap["sendbuff"] =
      uint64ToHexStr(reinterpret_cast<uint64_t>(info.sendbuff), "0x");
  infoMap["recvbuff"] =
      uint64ToHexStr(reinterpret_cast<uint64_t>(info.sendbuff), "0x");
  return mapToString(collKeys, infoMap);
}

ncclResult_t collTraceInit(ncclComm* comm) {
  try {
    if (!NCCL_COLLTRACE.empty()) {
      comm->collTrace = std::unique_ptr<CollTrace>(new CollTrace(comm));
    }
  } catch (const std::exception& e) {
    WARN("COLLTRACE initialization failed: %s\n", e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  if (comm->collTrace) {
    comm->collTrace.reset();
  }
  // Try catch clause here is not going to be useful as destructors are noexcept
  // by default. Instead of throwing an exception it will just crash the
  // program. We need to think about a better way to handle this.
  return ncclSuccess;
}
