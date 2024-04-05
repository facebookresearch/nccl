// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef TRACE_UTILES_H
#define TRACE_UTILES_H

#include <chrono>
#include <deque>
#include <iomanip>
#include <list>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// TODO: should this be util function?
static inline std::string timePointToStr(
    std::chrono::time_point<std::chrono::high_resolution_clock> ts) {
  std::time_t ts_c = std::chrono::system_clock::to_time_t(ts);
  auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                   ts.time_since_epoch()) %
      1000000;
  std::stringstream ts_ss;
  ts_ss << std::put_time(std::localtime(&ts_c), "%T.") << std::setfill('0')
        << std::setw(6) << ts_us.count();
  return ts_ss.str();
}

// Overwrite std::toString to support std::string type
template <typename T>
static inline std::string toString(const T& obj) {
  std::ostringstream oss{};
  oss << obj;
  return oss.str();
}

template <typename T>
static inline std::string toQuotedString(const T& obj) {
  return "\"" + toString(obj) + "\"";
}

/**
 * Serialize a map object to json string
 * Input arguments:
 *   keys: a vector of keys in the order of insertion
 *   map: the map object to be serialized
 *   quoted: whether to quote string key or value
 */
template <typename T>
static inline std::string serializeMap(
    std::vector<std::string>& keys,
    std::unordered_map<std::string, T>& map,
    bool quoted = false) {
  std::string final_string = "{";
  // unordered_map doesn't maintain insertion order. Use keys to ensure
  // serialize in the same order as program defined
  for (auto& key : keys) {
    // skip if key doesn't exist in map
    if (map.find(key) == map.end()) {
      continue;
    }
    T& val = map[key];
    final_string += quoted ? toQuotedString(key) : key; // always quote key
    final_string += ": ";
    // only string value needs to be quoted; set when creating map
    final_string += toString(val);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "}";
  return final_string;
}

/**
 * Serialize a unordered set object to json string
 * Input arguments:
 *   set: the unordered set object to be serialized
 */
template <typename T>
static inline std::string serializeSet(std::unordered_set<T>& set) {
  std::string final_string = "[";
  for (auto& it : set) {
    final_string += toString(it);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}

/**
 * Serialize a vector object to json string
 * Input arguments:
 *   vec: the vector object to be serialized
 */
template <typename T>
static inline std::string serializeVec(std::vector<T>& vec) {
  std::string final_string = "[";
  for (auto& it : vec) {
    final_string += toString(it);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}

/**
 * Serialize a list object to json string
 * Input arguments:
 *   list: the list object to be serialized
 */
template <typename T>
static inline std::string serializeList(std::list<T>& list) {
  std::string final_string = "[";
  for (auto& it : list) {
    final_string += toString(it);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}
/**
 * Serialize a deque of objects to json string. Require the object type has
 * serialize function. Input arguments: deque: the deque object to be serialized
 */
template <typename T>
static inline std::string serializeObjects(std::deque<T>& objs) {
  std::string final_string = "[";
  for (auto& obj : objs) {
    final_string += obj.serialize(true /*quote*/);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}

/**
 * Flatten a map object to plain string
 * Input arguments:
 *   keys: a vector of keys in the order of insertion
 *   map: the map object to be serialized
 *   kvdelim: delimiter between key and value
 *   kdelim: delimiter between key-value pairs
 */
template <typename T>
static inline std::string mapToString(
    std::vector<std::string>& keys,
    std::unordered_map<std::string, T>& map,
    const std::string& kvdelim = " ",
    const std::string& kdelim = " ") {
  std::string final_string = "";
  // unordered_map doesn't maintain insertion order. Use keys to ensure
  // serialize in the same order as program defined
  for (auto& key : keys) {
    // skip if key doesn't exist in map
    if (map.find(key) == map.end()) {
      continue;
    }
    T& val = map[key];
    final_string += key; // always quote key
    final_string += kvdelim;
    final_string += toString(val);
    final_string += kdelim;
  }
  if (final_string.size() > 1) {
    final_string = final_string.substr(
        0, final_string.size() - std::string(kdelim).size());
  }
  return final_string;
}
#endif
