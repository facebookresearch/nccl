// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iterator>
#include <string>
#include <unordered_map>
#include "CollTrace.h"
#include "ExtUtils.h"
#include "ProxyTrace.h"
#include "TraceUtils.h"
#include "comm.h"
#include "nccl.h"

static std::string dumpRing(int* userRanks, int nRank) {
  std::vector<std::string> ringVec;
  ringVec.reserve(nRank);
  for (int i = 0; i < nRank; i++) {
    ringVec.emplace_back(std::to_string(userRanks[i]));
  }
  return serializeVec(ringVec);
}

static std::string dumpRings(ncclComm_t comm) {
  std::vector<std::string> ringsVec;
  ringsVec.reserve(comm->nChannels);
  for (int i = 0; i < comm->nChannels; i++) {
    ringsVec.emplace_back(
        dumpRing(comm->channels[i].ring.userRanks, comm->nRanks));
  }
  return serializeVec(ringsVec);
}

static void dumpCommInfo(
    ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  map["commHash"] = toQuotedString(hashToHexStr(comm->commHash));
  map["rank"] = std::to_string(comm->rank);
  map["localRank"] = std::to_string(comm->localRank);
  map["node"] = std::to_string(comm->node);

  map["nRanks"] = std::to_string(comm->nRanks);
  map["localRanks"] = std::to_string(comm->localRanks);
  map["nNodes"] = std::to_string(comm->nNodes);
  map["commDesc"] = comm->config.commDesc;

  // Still only dump rings on rank 0 because it is a large amount of data
  if (comm->rank == 0) {
    // TODO: dump tree topology
    map["rings"] = dumpRings(comm);
  }
}

static void dumpCollTrace(
    ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  if (comm->collTrace != nullptr) {
    auto dump = comm->collTrace->dump();

    INFO(
        NCCL_ALL,
        "CommDump: COLLTRACE dump from rank %d comm %p commHash %lx commDesc %s : %zu past, %zu pending, %d current collective records",
        comm->rank,
        comm,
        comm->commHash,
        comm->config.commDesc,
        dump.pastColls.size(),
        dump.pendingColls.size(),
        dump.currentColl == nullptr ? 0 : 1);

    map["CT_pastColls"] = serializeObjects(dump.pastColls);
    map["CT_pendingColls"] = serializeObjects(dump.pendingColls);

    if (dump.currentColl != nullptr) {
      map["CT_currentColl"] = dump.currentColl->serialize(true);

      auto algorithm = dump.currentColl->info.algorithm;
      auto channelId = dump.currentColl->info.channelId;
    } else {
      map["CT_currentColl"] = "null";
    }
  } else {
    INFO(NCCL_ALL, "CommDump: COLLTRACE is disabled. No trace to dump");
  }
}

static void dumpProxyTrace(
    ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  if (comm->proxyState != nullptr && comm->proxyState->trace) {
    auto dump = comm->proxyState->trace->dump(comm->commHash);

    INFO(
        NCCL_ALL,
        "CommDump: PROXYTRACE dump from rank %d comm %p commHash %lx commDesc %s: %zu past collectives, %zu active network operations",
        comm->rank,
        comm,
        comm->commHash,
        comm->config.commDesc,
        dump.pastColls.size(),
        dump.activeOps.size());

    map["PT_pastColls"] = serializeObjects(dump.pastColls);
    map["PT_activeOps"] = serializeObjects(dump.activeOps);
    map["PT_activeColls"] = serializeObjects(dump.activeColls);
  } else {
    INFO(NCCL_ALL, "CommDump: PROXYTRACE is disabled. No trace to dump");
  }
}

__attribute__((visibility("default"))) ncclResult_t ncclCommDump(
    ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  dumpCommInfo(comm, map);
  dumpCollTrace(comm, map);
  dumpProxyTrace(comm, map);

  return ncclSuccess;
}
