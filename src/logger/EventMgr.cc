#include "EventMgr.h"
#include <cstring>

#include "ExtUtils.h"
#include "FbInternal.h"
#include "TraceUtils.h"
#include "nccl_cvars.h"

static std::vector<std::string> commkeys = {
    "commId",
    "jobname",
    "version",
    "attempt",
    "time",
    "commHash",
    "commDesc",
    "rank",
    "nranks",
    "stage",
    "split",
    "timerDeltaMs"};

static inline std::string getJobName() {
  char* hpcJobName = getenv("NCCL_HPC_JOB_NAME");
  if (hpcJobName && strlen(hpcJobName)) {
    const auto hpcJobIdStr = std::string(hpcJobName);
    const auto lastSlash = hpcJobIdStr.find_last_of('/');
    const auto jobName = lastSlash != hpcJobIdStr.npos
        ? hpcJobIdStr.substr(lastSlash + 1)
        : hpcJobIdStr;
    return jobName;
  }

  return "unknown";
};

static inline int getJobVersion() {
  char* hpcJobVersion = getenv("NCCL_HPC_JOB_VERSION");
  if (hpcJobVersion && strlen(hpcJobVersion)) {
    // char* to int convertion
    return std::stoi(hpcJobVersion);
  }

  return -1; // unknown job version
};

static inline int getJobAttempt() {
  char* hpcJobAttempt = getenv("NCCL_HPC_JOB_ATTEMPT");
  if (hpcJobAttempt && strlen(hpcJobAttempt)) {
    // char* to int convertion
    return std::stoi(hpcJobAttempt);
  }

  return -1; // unknown job attempt
};

void ScubaEntry::addCommonFields() {
  addNormalValue("jobname", getJobName());
  addIntValue("version", getJobVersion());
  addIntValue("attempt", getJobAttempt());
  addIntValue("time", getTimestamp());
}

std::string CommEvent::serialize() {
  ScubaEntry entry;
  entry.addCommonFields();

  entry.addIntValue(
      "commId",
      static_cast<int64_t>(
          commId % INT64_MAX)); // scuba requires bigint for integer
  entry.addIntValue(
      "commHash",
      static_cast<int64_t>(
          commHash % INT64_MAX)); // scuba requires bigint for integer
  entry.addNormalValue("commDesc", commDesc);
  entry.addIntValue("rank", rank);
  entry.addIntValue("nranks", nRanks);

  entry.addNormalValue("stage", stage);
  entry.addNormalValue("split", split);

  entry.addDoubleValue("timerDeltaMs", timerDeltaMs);

  std::unordered_map<std::string, std::string> tmpMap = {
      {"int", serializeMap(commkeys, entry.getIntMap(), true)},
      {"normal", serializeMap(commkeys, entry.getNormalMap(), true, true)},
      {"double", serializeMap(commkeys, entry.getDoubleMap(), true)}};

  std::vector<std::string> tmpkeys{"int", "normal", "double"};

  return serializeMap(tmpkeys, tmpMap, true) + "\n";
}

#ifdef ENABLE_FB_DATA_EXPORT

bool CommEvent::toScuba(std::string& tableName) {
  ScubaEntry entry;
  entry.addCommonFields();

  entry.addIntValue("commId", commId);
  entry.addIntValue("commHash", commHash);
  entry.addNormalValue("commDesc", commDesc);
  entry.addIntValue("rank", rank);
  entry.addIntValue("nranks", nRanks);

  entry.addNormalValue("stage", stage);
  entry.addNormalValue("split", split);

  entry.addDoubleValue("timerDeltaMs", timerDeltaMs);

  return logToScuba(
      tableName, entry.getNormalMap(), entry.getIntMap(), entry.getDoubleMap());
}
#else
bool CommEvent::toScuba(std::string& tableName) {
  return false;
}
#endif // ENABLE_FB_DATA_EXPORT
