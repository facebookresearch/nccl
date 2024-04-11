#ifndef _NCCL_EVENT_MGR_H
#define _NCCL_EVENT_MGR_H

#include <sys/types.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

enum class LoggerEventType {
  DebugEventType,
  CommEventType,
  CollEventType,
};

class ScubaEntry {
 public:
  ScubaEntry() = default;
  void addCommonFields();
  void addNormalValue(const std::string& key, const std::string& value) {
    normalMap_[key] = value;
  };

  void addIntValue(const std::string& key, const int64_t value) {
    intMap_[key] = value;
  };
  void addDoubleValue(const std::string& key, const double value) {
    doubleMap_[key] = value;
  }

  std::unordered_map<std::string, std::string>& getNormalMap() {
    return normalMap_;
  }
  std::unordered_map<std::string, int64_t>& getIntMap() {
    return intMap_;
  }
  std::unordered_map<std::string, double>& getDoubleMap() {
    return doubleMap_;
  }

 private:
  std::unordered_map<std::string, std::string> normalMap_;
  std::unordered_map<std::string, int64_t> intMap_;
  std::unordered_map<std::string, double> doubleMap_;
};

class LoggerEvent {
 public:
  virtual void setTimerDelta(double delta) = 0;
  virtual std::string serialize() = 0;
  virtual bool toScuba(std::string& tableName) = 0;
  virtual LoggerEventType getEventType() = 0;
  virtual ~LoggerEvent() = default;
};

class DebugEvent : public LoggerEvent {
 public:
  DebugEvent() = default;
  DebugEvent(std::string msg) : msg(msg) {}

  void setTimerDelta(double delta) override {
    // FIXME: this is a placeholder for now
  }

  bool toScuba(std::string& tableName) override {
    // FIXME: this is a placeholder for now
    return false;
  };

  std::string serialize() override {
    return msg;
  };

  ~DebugEvent() override{};
  LoggerEventType getEventType() override {
    return LoggerEventType::DebugEventType;
  }

 private:
  std::string msg;
};

struct CommEvent : public LoggerEvent {
 public:
  CommEvent() = default;
  CommEvent(
      unsigned long long commId,
      uint64_t commHash,
      int rank,
      int nRanks,
      std::string stage,
      std::string split,
      double delta = 0.0)
      : commId(commId),
        commHash(commHash),
        rank(rank),
        nRanks(nRanks),
        stage(stage),
        split(split),
        timerDeltaMs(delta) {}

  ~CommEvent() override = default;

  void setTimerDelta(double delta) override {
    timerDeltaMs = delta;
  }

  std::string serialize() override;
  bool toScuba(std::string& tableName) override;
  LoggerEventType getEventType() override {
    return LoggerEventType::CommEventType;
  }

 private:
  unsigned long long commId;
  uint64_t commHash;
  int rank;
  int nRanks;
  std::string stage;
  std::string split;
  double timerDeltaMs;
};

#endif
