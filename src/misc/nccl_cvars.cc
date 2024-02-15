// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <limits>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <strings.h>
#include <string.h>
#include <cuda_runtime.h>
#include "nccl_cvars.h"
#include "debug.h"
#include "checks.h"

// Cvar internal logger
// We need avoid calling into default logger because it may call ncclGetEnv() on
// demand in ncclDebugInit() and cause circular call & deadlock. Since CVAR_WARN
// happens usually only at initialization time and is for warning only, we might
// be OK to use separate logger here and always print to stdout.
static int pid = getpid();
static thread_local int tid = syscall(SYS_gettid);
static char hostname[HOST_NAME_MAX];
static bool enableCvarWarn = true;
static int cudaDev = -1;

void initEnvSet(std::unordered_set<std::string>& env);
void readCvarEnv();

#define CVAR_WARN(fmt, ...)                                 \
  if (enableCvarWarn) {                                     \
    printf(                                                 \
        "%s %s:%d:%d [%d] %s:%d NCCL WARN CVAR: " fmt "\n", \
        getTime().c_str(),                                  \
        hostname,                                           \
        pid,                                                \
        tid,                                                \
        cudaDev,                                            \
        __FUNCTION__,                                       \
        __LINE__,                                           \
        ##__VA_ARGS__);                                     \
  }

#define CVAR_WARN_UNKNOWN_VALUE(name, value)               \
  do {                                                     \
    CVAR_WARN("Unknown value %s for env %s", value, name); \
  } while (0)

static void initCvarLogger() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL || strcasecmp(nccl_debug, "VERSION") == 0) {
    enableCvarWarn = false;
  }
  getHostName(hostname, HOST_NAME_MAX, '.');

  // Used for ncclCvarInit time warning only
  CUDACHECKIGNORE(cudaGetDevice(&cudaDev));
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), s.end());
}

static std::vector<std::string> tokenizer(std::string str) {
  std::string delim = ",";
  std::vector<std::string> tokens;

  while (auto pos = str.find(",")) {
    std::string newstr = str.substr(0, pos);
    ltrim(newstr);
    rtrim(newstr);
    // Skip empty string
    if(!newstr.empty()) {
      if(std::find(tokens.begin(), tokens.end(), newstr) != tokens.end()) {
        CVAR_WARN("Duplicate token %s found in the value of %s", newstr.c_str(), str.c_str());
      }
      tokens.push_back(newstr);
    }
    str.erase(0, pos + delim.length());
    if (pos == std::string::npos) {
      break;
    }
  }
  return tokens;
}

static bool env2bool(const char *str_, const char *def) {
  std::string str(getenv(str_) ? getenv(str_) : def);
  std::transform(str.cbegin(), str.cend(), str.begin(), [](unsigned char c) { return std::tolower(c); });
  if (str == "y") return true;
  else if (str == "n") return false;
  else if (str == "yes") return true;
  else if (str == "no") return false;
  else if (str == "t") return true;
  else if (str == "f") return false;
  else if (str == "true") return true;
  else if (str == "false") return false;
  else if (str == "1") return true;
  else if (str == "0") return false;
  else CVAR_WARN_UNKNOWN_VALUE(str_, str.c_str());
  return true;
}

template <typename T>
static T env2num(const char *str, const char *def) {
  std::string s(getenv(str) ? getenv(str) : def);

  if (std::find_if(s.begin(), s.end(), ::isdigit) != s.end()) {
    /* if the string contains a digit, try converting it normally */
    std::stringstream sstream(s);
    T ret;
    sstream >> ret;
    return ret;
  } else {
    /* if there are no digits, see if its a special string such as
     * "MAX" or "MIN". */
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    if (s == "MAX") {
      return std::numeric_limits<T>::max();
    } else if (s == "MIN") {
      return std::numeric_limits<T>::min();
    } else {
      CVAR_WARN("Unrecognized numeral %s\n", s.c_str());
      return 0;
    }
  }
}

static std::string env2str(const char *str, const char *def_) {
  const char *def = def_ ? def_ : "";
  std::string str_s = getenv(str) ? std::string(getenv(str)) : std::string(def);
  ltrim(str_s);
  rtrim(str_s);
  return str_s;
}

static std::vector<std::string> env2strlist(const char* str, const char* def_) {
  const char* def = def_ ? def_ : "";
  std::string str_s(getenv(str) ? getenv(str) : def);
  return tokenizer(str_s);
}

static std::tuple<std::string, std::vector<std::string>> env2prefixedStrlist(
    const char* str,
    const char* def_,
    const std::vector<std::string>& prefixes) {
  const char* def = def_ ? def_ : "";
  std::string str_s(getenv(str) ? getenv(str) : def);

  // search if any prefix is specified
  for (auto prefix : prefixes) {
    if (!str_s.compare(0, prefix.size(), prefix)) {
      // if prefix is found, convert the remaining string to stringList
      std::string slist_s = str_s.substr(prefix.size());
      return std::make_tuple(prefix, tokenizer(slist_s));
    }
  }
  // if no prefix is found, convert entire string to stringList
  return std::make_tuple("", tokenizer(str_s));
}

extern char **environ;
void ncclCvarInit() {
  std::unordered_set<std::string> env;
  initEnvSet(env);

  initCvarLogger();

  // Check if any NCCL_ env var is not in allow list
  char **s = environ;
  for (; *s; s++) {
    if (!strncmp(*s, "NCCL_", strlen("NCCL_"))) {
      std::string str(*s);
      str = str.substr(0, str.find("="));
      if (env.find(str) == env.end()) {
        CVAR_WARN("Unknown env %s in the NCCL namespace", str.c_str());
      }
    }
  }

  readCvarEnv();
}

// Automatically generated by ./maint/extractcvars.py --- START
// DO NOT EDIT!!!
std::string CUDA_LAUNCH_BLOCKING;
std::string CUDA_LAUNCH_BLOCKING_DEFAULT;
int64_t NCCL_AGG_CHANNEL_SIZE;
int64_t NCCL_AGG_CHANNEL_SIZE_DEFAULT;
std::string NCCL_ALGO;
std::string NCCL_ALGO_DEFAULT;
enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO;
enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO_DEFAULT;
int64_t NCCL_ALLOC_P2P_NET_LL_BUFFERS;
int64_t NCCL_ALLOC_P2P_NET_LL_BUFFERS_DEFAULT;
enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO;
enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO_DEFAULT;
enum NCCL_ALLTOALLV_ALGO NCCL_ALLTOALLV_ALGO;
enum NCCL_ALLTOALLV_ALGO NCCL_ALLTOALLV_ALGO_DEFAULT;
enum NCCL_ALLTOALL_ALGO NCCL_ALLTOALL_ALGO;
enum NCCL_ALLTOALL_ALGO NCCL_ALLTOALL_ALGO_DEFAULT;
int64_t NCCL_BUFFSIZE;
int64_t NCCL_BUFFSIZE_DEFAULT;
int64_t NCCL_CGA_CLUSTER_SIZE;
int64_t NCCL_CGA_CLUSTER_SIZE_DEFAULT;
int64_t NCCL_CHECK_POINTERS;
int64_t NCCL_CHECK_POINTERS_DEFAULT;
int64_t NCCL_CHUNK_SIZE;
int64_t NCCL_CHUNK_SIZE_DEFAULT;
std::string NCCL_COLLNET_ENABLE;
std::string NCCL_COLLNET_ENABLE_DEFAULT;
int64_t NCCL_COLLNET_NODE_THRESHOLD;
int64_t NCCL_COLLNET_NODE_THRESHOLD_DEFAULT;
int64_t NCCL_COMM_BLOCKING;
int64_t NCCL_COMM_BLOCKING_DEFAULT;
std::string NCCL_COMM_ID;
std::string NCCL_COMM_ID_DEFAULT;
int NCCL_COMM_SPLIT_SHARE_RESOURCES;
int NCCL_COMM_SPLIT_SHARE_RESOURCES_DEFAULT;
int64_t NCCL_CONNECT_ROUND_MAX_PEERS;
int64_t NCCL_CONNECT_ROUND_MAX_PEERS_DEFAULT;
int64_t NCCL_CREATE_THREAD_CONTEXT;
int64_t NCCL_CREATE_THREAD_CONTEXT_DEFAULT;
int64_t NCCL_CROSS_NIC;
int64_t NCCL_CROSS_NIC_DEFAULT;
bool NCCL_CTRAN_AG_RD_RTR;
bool NCCL_CTRAN_AG_RD_RTR_DEFAULT;
int NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS;
int NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS_DEFAULT;
int NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE;
int NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE_DEFAULT;
int NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS;
int NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS_DEFAULT;
int NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE;
int NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE_DEFAULT;
uint64_t NCCL_CTRAN_ALLTOALL_THRESHOLD;
uint64_t NCCL_CTRAN_ALLTOALL_THRESHOLD_DEFAULT;
std::vector<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS;
std::vector<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS_DEFAULT;
uint64_t NCCL_CTRAN_IB_CTRL_TC;
uint64_t NCCL_CTRAN_IB_CTRL_TC_DEFAULT;
int NCCL_CTRAN_IB_MAX_QPS;
int NCCL_CTRAN_IB_MAX_QPS_DEFAULT;
uint64_t NCCL_CTRAN_IB_QP_SCALING_THRESHOLD;
uint64_t NCCL_CTRAN_IB_QP_SCALING_THRESHOLD_DEFAULT;
bool NCCL_CTRAN_IB_TRAFFIC_PROFILNG;
bool NCCL_CTRAN_IB_TRAFFIC_PROFILNG_DEFAULT;
std::string NCCL_CTRAN_KINETO_PROFILE_DIR;
std::string NCCL_CTRAN_KINETO_PROFILE_DIR_DEFAULT;
int NCCL_CTRAN_NUM_KERNEL_P2PELEMS;
int NCCL_CTRAN_NUM_KERNEL_P2PELEMS_DEFAULT;
enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING;
enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING_DEFAULT;
int NCCL_CTRAN_PROFILING_REPORT_COUNT;
int NCCL_CTRAN_PROFILING_REPORT_COUNT_DEFAULT;
enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER;
enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER_DEFAULT;
int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT;
int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT_DEFAULT;
int NCCL_CTRAN_RING_MAX_OUTSTANDING;
int NCCL_CTRAN_RING_MAX_OUTSTANDING_DEFAULT;
uint64_t NCCL_CTRAN_RING_STEP;
uint64_t NCCL_CTRAN_RING_STEP_DEFAULT;
uint64_t NCCL_CTRAN_SHARED_DEVBUF_SIZE;
uint64_t NCCL_CTRAN_SHARED_DEVBUF_SIZE_DEFAULT;
std::string NCCL_CTRAN_TOPO_FILE;
std::string NCCL_CTRAN_TOPO_FILE_DEFAULT;
std::vector<std::string> NCCL_CTRAN_TOPO_FILE_KEYS;
std::vector<std::string> NCCL_CTRAN_TOPO_FILE_KEYS_DEFAULT;
std::string NCCL_CUDA_PATH;
std::string NCCL_CUDA_PATH_DEFAULT;
int64_t NCCL_CUMEM_ENABLE;
int64_t NCCL_CUMEM_ENABLE_DEFAULT;
int NCCL_DDA_ALLREDUCE_MAX_BLOCKS;
int NCCL_DDA_ALLREDUCE_MAX_BLOCKS_DEFAULT;
uint64_t NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD;
uint64_t NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD_DEFAULT;
uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD;
uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_DEFAULT;
uint64_t NCCL_DDA_TMPBUFF_SIZE;
uint64_t NCCL_DDA_TMPBUFF_SIZE_DEFAULT;
std::string NCCL_DEBUG;
std::string NCCL_DEBUG_DEFAULT;
std::string NCCL_DEBUG_FILE;
std::string NCCL_DEBUG_FILE_DEFAULT;
std::string NCCL_DEBUG_SUBSYS;
std::string NCCL_DEBUG_SUBSYS_DEFAULT;
int64_t NCCL_DMABUF_ENABLE;
int64_t NCCL_DMABUF_ENABLE_DEFAULT;
int64_t NCCL_GDRCOPY_ENABLE;
int64_t NCCL_GDRCOPY_ENABLE_DEFAULT;
int64_t NCCL_GDRCOPY_FIFO_ENABLE;
int64_t NCCL_GDRCOPY_FIFO_ENABLE_DEFAULT;
bool NCCL_GDRCOPY_FLUSH_ENABLE;
bool NCCL_GDRCOPY_FLUSH_ENABLE_DEFAULT;
bool NCCL_GDRCOPY_SYNC_ENABLE;
bool NCCL_GDRCOPY_SYNC_ENABLE_DEFAULT;
int64_t NCCL_GDR_FLUSH_DISABLE;
int64_t NCCL_GDR_FLUSH_DISABLE_DEFAULT;
std::string NCCL_GRAPH_DUMP_FILE;
std::string NCCL_GRAPH_DUMP_FILE_DEFAULT;
int64_t NCCL_GRAPH_DUMP_FILE_RANK;
int64_t NCCL_GRAPH_DUMP_FILE_RANK_DEFAULT;
std::string NCCL_GRAPH_FILE;
std::string NCCL_GRAPH_FILE_DEFAULT;
bool NCCL_GRAPH_MIXING_SUPPORT;
bool NCCL_GRAPH_MIXING_SUPPORT_DEFAULT;
int64_t NCCL_GRAPH_REGISTER;
int64_t NCCL_GRAPH_REGISTER_DEFAULT;
std::string NCCL_HOSTID;
std::string NCCL_HOSTID_DEFAULT;
int64_t NCCL_IB_ADAPTIVE_ROUTING;
int64_t NCCL_IB_ADAPTIVE_ROUTING_DEFAULT;
int64_t NCCL_IB_AR_THRESHOLD;
int64_t NCCL_IB_AR_THRESHOLD_DEFAULT;
int64_t NCCL_IB_DISABLE;
int64_t NCCL_IB_DISABLE_DEFAULT;
int64_t NCCL_IB_GID_INDEX;
int64_t NCCL_IB_GID_INDEX_DEFAULT;
std::string NCCL_IB_HCA_PREFIX;
std::string NCCL_IB_HCA_PREFIX_DEFAULT;
std::vector<std::string> NCCL_IB_HCA;
std::vector<std::string> NCCL_IB_HCA_DEFAULT;
int64_t NCCL_IB_MERGE_VFS;
int64_t NCCL_IB_MERGE_VFS_DEFAULT;
int64_t NCCL_IB_PCI_RELAXED_ORDERING;
int64_t NCCL_IB_PCI_RELAXED_ORDERING_DEFAULT;
int64_t NCCL_IB_PKEY;
int64_t NCCL_IB_PKEY_DEFAULT;
int64_t NCCL_IB_QPS_PER_CONNECTION;
int64_t NCCL_IB_QPS_PER_CONNECTION_DEFAULT;
int64_t NCCL_IB_RETRY_CNT;
int64_t NCCL_IB_RETRY_CNT_DEFAULT;
int64_t NCCL_IB_SL;
int64_t NCCL_IB_SL_DEFAULT;
int64_t NCCL_IB_SPLIT_DATA_ON_QPS;
int64_t NCCL_IB_SPLIT_DATA_ON_QPS_DEFAULT;
int64_t NCCL_IB_TC;
int64_t NCCL_IB_TC_DEFAULT;
int64_t NCCL_IB_TIMEOUT;
int64_t NCCL_IB_TIMEOUT_DEFAULT;
int64_t NCCL_IB_USE_INLINE;
int64_t NCCL_IB_USE_INLINE_DEFAULT;
int64_t NCCL_IGNORE_CPU_AFFINITY;
int64_t NCCL_IGNORE_CPU_AFFINITY_DEFAULT;
int64_t NCCL_IGNORE_DISABLED_P2P;
int64_t NCCL_IGNORE_DISABLED_P2P_DEFAULT;
int NCCL_L1_SHARED_MEMORY_CARVEOUT;
int NCCL_L1_SHARED_MEMORY_CARVEOUT_DEFAULT;
std::string NCCL_LAUNCH_MODE;
std::string NCCL_LAUNCH_MODE_DEFAULT;
int64_t NCCL_LL128_BUFFSIZE;
int64_t NCCL_LL128_BUFFSIZE_DEFAULT;
int64_t NCCL_LL128_NTHREADS;
int64_t NCCL_LL128_NTHREADS_DEFAULT;
int64_t NCCL_LL_BUFFSIZE;
int64_t NCCL_LL_BUFFSIZE_DEFAULT;
int64_t NCCL_LOCAL_REGISTER;
int64_t NCCL_LOCAL_REGISTER_DEFAULT;
enum NCCL_LOGGER_MODE NCCL_LOGGER_MODE;
enum NCCL_LOGGER_MODE NCCL_LOGGER_MODE_DEFAULT;
int NCCL_MAX_CTAS;
int NCCL_MAX_CTAS_DEFAULT;
int NCCL_MAX_NCHANNELS;
int NCCL_MAX_NCHANNELS_DEFAULT;
int NCCL_MAX_NRINGS;
int NCCL_MAX_NRINGS_DEFAULT;
int NCCL_MAX_P2P_NCHANNELS;
int NCCL_MAX_P2P_NCHANNELS_DEFAULT;
enum NCCL_MEM_SYNC_DOMAIN NCCL_MEM_SYNC_DOMAIN;
enum NCCL_MEM_SYNC_DOMAIN NCCL_MEM_SYNC_DOMAIN_DEFAULT;
int NCCL_MIN_CTAS;
int NCCL_MIN_CTAS_DEFAULT;
int NCCL_MIN_NCHANNELS;
int NCCL_MIN_NCHANNELS_DEFAULT;
int NCCL_MIN_NRINGS;
int NCCL_MIN_NRINGS_DEFAULT;
int NCCL_MIN_P2P_NCHANNELS;
int NCCL_MIN_P2P_NCHANNELS_DEFAULT;
int64_t NCCL_NCHANNELS_PER_NET_PEER;
int64_t NCCL_NCHANNELS_PER_NET_PEER_DEFAULT;
std::string NCCL_NETWORK;
std::string NCCL_NETWORK_DEFAULT;
int64_t NCCL_NET_DISABLE_INTRA;
int64_t NCCL_NET_DISABLE_INTRA_DEFAULT;
int64_t NCCL_NET_FORCE_FLUSH;
int64_t NCCL_NET_FORCE_FLUSH_DEFAULT;
std::string NCCL_NET_GDR_LEVEL;
std::string NCCL_NET_GDR_LEVEL_DEFAULT;
int64_t NCCL_NET_GDR_READ;
int64_t NCCL_NET_GDR_READ_DEFAULT;
int64_t NCCL_NET_OVERHEAD;
int64_t NCCL_NET_OVERHEAD_DEFAULT;
std::string NCCL_NET_PLUGIN;
std::string NCCL_NET_PLUGIN_DEFAULT;
int64_t NCCL_NET_SHARED_BUFFERS;
int64_t NCCL_NET_SHARED_BUFFERS_DEFAULT;
bool NCCL_NET_SHARED_COMMS;
bool NCCL_NET_SHARED_COMMS_DEFAULT;
int NCCL_NSOCKS_PERTHREAD;
int NCCL_NSOCKS_PERTHREAD_DEFAULT;
int64_t NCCL_NTHREADS;
int64_t NCCL_NTHREADS_DEFAULT;
int64_t NCCL_NVB_DISABLE;
int64_t NCCL_NVB_DISABLE_DEFAULT;
int64_t NCCL_NVB_PRECONNECT;
int64_t NCCL_NVB_PRECONNECT_DEFAULT;
int64_t NCCL_NVLS_ENABLE;
int64_t NCCL_NVLS_ENABLE_DEFAULT;
int NCCL_NVLS_NCHANNELS;
int NCCL_NVLS_NCHANNELS_DEFAULT;
bool NCCL_P2P_DIRECT_DISABLE;
bool NCCL_P2P_DIRECT_DISABLE_DEFAULT;
bool NCCL_P2P_DISABLE;
bool NCCL_P2P_DISABLE_DEFAULT;
std::string NCCL_P2P_LEVEL;
std::string NCCL_P2P_LEVEL_DEFAULT;
int64_t NCCL_P2P_LL_THRESHOLD;
int64_t NCCL_P2P_LL_THRESHOLD_DEFAULT;
int64_t NCCL_P2P_NET_CHUNKSIZE;
int64_t NCCL_P2P_NET_CHUNKSIZE_DEFAULT;
int64_t NCCL_P2P_NVL_CHUNKSIZE;
int64_t NCCL_P2P_NVL_CHUNKSIZE_DEFAULT;
int64_t NCCL_P2P_PCI_CHUNKSIZE;
int64_t NCCL_P2P_PCI_CHUNKSIZE_DEFAULT;
int64_t NCCL_P2P_PXN_LEVEL;
int64_t NCCL_P2P_PXN_LEVEL_DEFAULT;
int64_t NCCL_P2P_READ_ENABLE;
int64_t NCCL_P2P_READ_ENABLE_DEFAULT;
int64_t NCCL_P2P_USE_CUDA_MEMCPY;
int64_t NCCL_P2P_USE_CUDA_MEMCPY_DEFAULT;
int64_t NCCL_PROGRESS_APPENDOP_FREQ;
int64_t NCCL_PROGRESS_APPENDOP_FREQ_DEFAULT;
std::string NCCL_PROTO;
std::string NCCL_PROTO_DEFAULT;
int64_t NCCL_PROXY_APPEND_BATCH_SIZE;
int64_t NCCL_PROXY_APPEND_BATCH_SIZE_DEFAULT;
int64_t NCCL_PROXY_DUMP_SIGNAL;
int64_t NCCL_PROXY_DUMP_SIGNAL_DEFAULT;
std::string NCCL_PROXY_PROFILE;
std::string NCCL_PROXY_PROFILE_DEFAULT;
std::string NCCL_PROXY_PROFILE_DIR;
std::string NCCL_PROXY_PROFILE_DIR_DEFAULT;
int64_t NCCL_PXN_DISABLE;
int64_t NCCL_PXN_DISABLE_DEFAULT;
int64_t NCCL_REPORT_CONNECT_PROGRESS;
int64_t NCCL_REPORT_CONNECT_PROGRESS_DEFAULT;
enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO;
enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO_DEFAULT;
int64_t NCCL_SET_STACK_SIZE;
int64_t NCCL_SET_STACK_SIZE_DEFAULT;
int64_t NCCL_SET_THREAD_NAME;
int64_t NCCL_SET_THREAD_NAME_DEFAULT;
bool NCCL_SHM_DISABLE;
bool NCCL_SHM_DISABLE_DEFAULT;
int64_t NCCL_SHM_LOCALITY;
int64_t NCCL_SHM_LOCALITY_DEFAULT;
int64_t NCCL_SHM_MEMCPY_MODE;
int64_t NCCL_SHM_MEMCPY_MODE_DEFAULT;
bool NCCL_SHM_USE_CUDA_MEMCPY;
bool NCCL_SHM_USE_CUDA_MEMCPY_DEFAULT;
std::string NCCL_SOCKET_FAMILY;
std::string NCCL_SOCKET_FAMILY_DEFAULT;
std::string NCCL_SOCKET_IFNAME;
std::string NCCL_SOCKET_IFNAME_DEFAULT;
int NCCL_SOCKET_NTHREADS;
int NCCL_SOCKET_NTHREADS_DEFAULT;
std::string NCCL_THREAD_THRESHOLDS;
std::string NCCL_THREAD_THRESHOLDS_DEFAULT;
std::string NCCL_TOPO_DUMP_FILE;
std::string NCCL_TOPO_DUMP_FILE_DEFAULT;
int64_t NCCL_TOPO_DUMP_FILE_RANK;
int64_t NCCL_TOPO_DUMP_FILE_RANK_DEFAULT;
std::string NCCL_TOPO_FILE;
std::string NCCL_TOPO_FILE_DEFAULT;
std::string NCCL_TUNER_PLUGIN;
std::string NCCL_TUNER_PLUGIN_DEFAULT;
int64_t NCCL_WARN_ENABLE_DEBUG_INFO;
int64_t NCCL_WARN_ENABLE_DEBUG_INFO_DEFAULT;
int64_t NCCL_WORK_FIFO_DEPTH;
int64_t NCCL_WORK_FIFO_DEPTH_DEFAULT;

void initEnvSet(std::unordered_set<std::string>& env) {
  env.insert("CUDA_LAUNCH_BLOCKING");
  env.insert("NCCL_AGG_CHANNEL_SIZE");
  env.insert("NCCL_ALGO");
  env.insert("NCCL_ALLGATHER_ALGO");
  env.insert("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
  env.insert("NCCL_ALLREDUCE_ALGO");
  env.insert("NCCL_ALLTOALLV_ALGO");
  env.insert("NCCL_ALLTOALL_ALGO");
  env.insert("NCCL_BUFFSIZE");
  env.insert("NCCL_CGA_CLUSTER_SIZE");
  env.insert("NCCL_CHECK_POINTERS");
  env.insert("NCCL_CHUNK_SIZE");
  env.insert("NCCL_COLLNET_ENABLE");
  env.insert("NCCL_COLLNET_NODE_THRESHOLD");
  env.insert("NCCL_COMM_BLOCKING");
  env.insert("NCCL_COMM_ID");
  env.insert("NCCL_COMM_SPLIT_SHARE_RESOURCES");
  env.insert("NCCL_CONNECT_ROUND_MAX_PEERS");
  env.insert("NCCL_CREATE_THREAD_CONTEXT");
  env.insert("NCCL_CROSS_NIC");
  env.insert("NCCL_CTRAN_AG_RD_RTR");
  env.insert("NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS");
  env.insert("NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE");
  env.insert("NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS");
  env.insert("NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE");
  env.insert("NCCL_CTRAN_ALLTOALL_THRESHOLD");
  env.insert("NCCL_CTRAN_BACKENDS");
  env.insert("NCCL_CTRAN_IB_CTRL_TC");
  env.insert("NCCL_CTRAN_IB_MAX_QPS");
  env.insert("NCCL_CTRAN_IB_QP_SCALING_THRESHOLD");
  env.insert("NCCL_CTRAN_IB_TRAFFIC_PROFILNG");
  env.insert("NCCL_CTRAN_KINETO_PROFILE_DIR");
  env.insert("NCCL_CTRAN_NUM_KERNEL_P2PELEMS");
  env.insert("NCCL_CTRAN_PROFILING");
  env.insert("NCCL_CTRAN_PROFILING_REPORT_COUNT");
  env.insert("NCCL_CTRAN_REGISTER");
  env.insert("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT");
  env.insert("NCCL_CTRAN_RING_MAX_OUTSTANDING");
  env.insert("NCCL_CTRAN_RING_STEP");
  env.insert("NCCL_CTRAN_SHARED_DEVBUF_SIZE");
  env.insert("NCCL_CTRAN_TOPO_FILE");
  env.insert("NCCL_CTRAN_TOPO_FILE_KEYS");
  env.insert("NCCL_CUDA_PATH");
  env.insert("NCCL_CUMEM_ENABLE");
  env.insert("NCCL_DDA_ALLREDUCE_MAX_BLOCKS");
  env.insert("NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD");
  env.insert("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD");
  env.insert("NCCL_DDA_TMPBUFF_SIZE");
  env.insert("NCCL_DEBUG");
  env.insert("NCCL_DEBUG_FILE");
  env.insert("NCCL_DEBUG_SUBSYS");
  env.insert("NCCL_DMABUF_ENABLE");
  env.insert("NCCL_GDRCOPY_ENABLE");
  env.insert("NCCL_GDRCOPY_FIFO_ENABLE");
  env.insert("NCCL_GDRCOPY_FLUSH_ENABLE");
  env.insert("NCCL_GDRCOPY_SYNC_ENABLE");
  env.insert("NCCL_GDR_FLUSH_DISABLE");
  env.insert("NCCL_GRAPH_DUMP_FILE");
  env.insert("NCCL_GRAPH_DUMP_FILE_RANK");
  env.insert("NCCL_GRAPH_FILE");
  env.insert("NCCL_GRAPH_MIXING_SUPPORT");
  env.insert("NCCL_GRAPH_REGISTER");
  env.insert("NCCL_HOSTID");
  env.insert("NCCL_IB_ADAPTIVE_ROUTING");
  env.insert("NCCL_IB_AR_THRESHOLD");
  env.insert("NCCL_IB_DISABLE");
  env.insert("NCCL_IB_GID_INDEX");
  env.insert("NCCL_IB_HCA");
  env.insert("NCCL_IB_MERGE_VFS");
  env.insert("NCCL_IB_PCI_RELAXED_ORDERING");
  env.insert("NCCL_IB_PKEY");
  env.insert("NCCL_IB_QPS_PER_CONNECTION");
  env.insert("NCCL_IB_RETRY_CNT");
  env.insert("NCCL_IB_SL");
  env.insert("NCCL_IB_SPLIT_DATA_ON_QPS");
  env.insert("NCCL_IB_TC");
  env.insert("NCCL_IB_TIMEOUT");
  env.insert("NCCL_IB_USE_INLINE");
  env.insert("NCCL_IGNORE_CPU_AFFINITY");
  env.insert("NCCL_IGNORE_DISABLED_P2P");
  env.insert("NCCL_L1_SHARED_MEMORY_CARVEOUT");
  env.insert("NCCL_LAUNCH_MODE");
  env.insert("NCCL_LL128_BUFFSIZE");
  env.insert("NCCL_LL128_NTHREADS");
  env.insert("NCCL_LL_BUFFSIZE");
  env.insert("NCCL_LOCAL_REGISTER");
  env.insert("NCCL_LOGGER_MODE");
  env.insert("NCCL_MAX_CTAS");
  env.insert("NCCL_MAX_NCHANNELS");
  env.insert("NCCL_MAX_NRINGS");
  env.insert("NCCL_MAX_P2P_NCHANNELS");
  env.insert("NCCL_MEM_SYNC_DOMAIN");
  env.insert("NCCL_MIN_CTAS");
  env.insert("NCCL_MIN_NCHANNELS");
  env.insert("NCCL_MIN_NRINGS");
  env.insert("NCCL_MIN_P2P_NCHANNELS");
  env.insert("NCCL_NCHANNELS_PER_NET_PEER");
  env.insert("NCCL_NET");
  env.insert("NCCL_NET_DISABLE_INTRA");
  env.insert("NCCL_NET_FORCE_FLUSH");
  env.insert("NCCL_NET_GDR_LEVEL");
  env.insert("NCCL_NET_GDR_READ");
  env.insert("NCCL_NET_OVERHEAD");
  env.insert("NCCL_NET_PLUGIN");
  env.insert("NCCL_NET_SHARED_BUFFERS");
  env.insert("NCCL_NET_SHARED_COMMS");
  env.insert("NCCL_NSOCKS_PERTHREAD");
  env.insert("NCCL_NTHREADS");
  env.insert("NCCL_NVB_DISABLE");
  env.insert("NCCL_NVB_PRECONNECT");
  env.insert("NCCL_NVLS_ENABLE");
  env.insert("NCCL_NVLS_NCHANNELS");
  env.insert("NCCL_P2P_DIRECT_DISABLE");
  env.insert("NCCL_P2P_DISABLE");
  env.insert("NCCL_P2P_LEVEL");
  env.insert("NCCL_P2P_LL_THRESHOLD");
  env.insert("NCCL_P2P_NET_CHUNKSIZE");
  env.insert("NCCL_P2P_NVL_CHUNKSIZE");
  env.insert("NCCL_P2P_PCI_CHUNKSIZE");
  env.insert("NCCL_P2P_PXN_LEVEL");
  env.insert("NCCL_P2P_READ_ENABLE");
  env.insert("NCCL_P2P_USE_CUDA_MEMCPY");
  env.insert("NCCL_PROGRESS_APPENDOP_FREQ");
  env.insert("NCCL_PROTO");
  env.insert("NCCL_PROXY_APPEND_BATCH_SIZE");
  env.insert("NCCL_PROXY_DUMP_SIGNAL");
  env.insert("NCCL_PROXY_PROFILE");
  env.insert("NCCL_PROXY_PROFILE_DIR");
  env.insert("NCCL_PXN_DISABLE");
  env.insert("NCCL_REPORT_CONNECT_PROGRESS");
  env.insert("NCCL_SENDRECV_ALGO");
  env.insert("NCCL_SET_STACK_SIZE");
  env.insert("NCCL_SET_THREAD_NAME");
  env.insert("NCCL_SHM_DISABLE");
  env.insert("NCCL_SHM_LOCALITY");
  env.insert("NCCL_SHM_MEMCPY_MODE");
  env.insert("NCCL_SHM_USE_CUDA_MEMCPY");
  env.insert("NCCL_SOCKET_FAMILY");
  env.insert("NCCL_SOCKET_IFNAME");
  env.insert("NCCL_SOCKET_NTHREADS");
  env.insert("NCCL_THREAD_THRESHOLDS");
  env.insert("NCCL_TOPO_DUMP_FILE");
  env.insert("NCCL_TOPO_DUMP_FILE_RANK");
  env.insert("NCCL_TOPO_FILE");
  env.insert("NCCL_TUNER_PLUGIN");
  env.insert("NCCL_WARN_ENABLE_DEBUG_INFO");
  env.insert("NCCL_WORK_FIFO_DEPTH");
}

void readCvarEnv() {
  CUDA_LAUNCH_BLOCKING = env2str("CUDA_LAUNCH_BLOCKING", "");
  CUDA_LAUNCH_BLOCKING_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_AGG_CHANNEL_SIZE = env2num<int64_t>("NCCL_AGG_CHANNEL_SIZE", "-2");
  NCCL_AGG_CHANNEL_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_ALGO = env2str("NCCL_ALGO", "");
  NCCL_ALGO_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  if (getenv("NCCL_ALLGATHER_ALGO") == nullptr) {
    NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLGATHER_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::orig;
    } else if (str == std::string("ctdirect")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctdirect;
    } else if (str == std::string("ctring")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctring;
    } else if (str == std::string("ctrd")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctrd;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_ALLGATHER_ALGO", str.c_str());
    }
  }
  NCCL_ALLGATHER_ALGO_DEFAULT = NCCL_ALLGATHER_ALGO::orig;

  NCCL_ALLOC_P2P_NET_LL_BUFFERS = env2num<int64_t>("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0");
  NCCL_ALLOC_P2P_NET_LL_BUFFERS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  if (getenv("NCCL_ALLREDUCE_ALGO") == nullptr) {
    NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLREDUCE_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::orig;
    } else if (str == std::string("dda")) {
      NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::dda;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_ALLREDUCE_ALGO", str.c_str());
    }
  }
  NCCL_ALLREDUCE_ALGO_DEFAULT = NCCL_ALLREDUCE_ALGO::orig;

  if (getenv("NCCL_ALLTOALLV_ALGO") == nullptr) {
    NCCL_ALLTOALLV_ALGO = NCCL_ALLTOALLV_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLTOALLV_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLTOALLV_ALGO = NCCL_ALLTOALLV_ALGO::orig;
    } else if (str == std::string("ctran")) {
      NCCL_ALLTOALLV_ALGO = NCCL_ALLTOALLV_ALGO::ctran;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_ALLTOALLV_ALGO", str.c_str());
    }
  }
  NCCL_ALLTOALLV_ALGO_DEFAULT = NCCL_ALLTOALLV_ALGO::orig;

  if (getenv("NCCL_ALLTOALL_ALGO") == nullptr) {
    NCCL_ALLTOALL_ALGO = NCCL_ALLTOALL_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLTOALL_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLTOALL_ALGO = NCCL_ALLTOALL_ALGO::orig;
    } else if (str == std::string("ctran")) {
      NCCL_ALLTOALL_ALGO = NCCL_ALLTOALL_ALGO::ctran;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_ALLTOALL_ALGO", str.c_str());
    }
  }
  NCCL_ALLTOALL_ALGO_DEFAULT = NCCL_ALLTOALL_ALGO::orig;

  NCCL_BUFFSIZE = env2num<int64_t>("NCCL_BUFFSIZE", "-2");
  NCCL_BUFFSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_CGA_CLUSTER_SIZE = env2num<int64_t>("NCCL_CGA_CLUSTER_SIZE", "-1");
  NCCL_CGA_CLUSTER_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_CHECK_POINTERS = env2num<int64_t>("NCCL_CHECK_POINTERS", "0");
  NCCL_CHECK_POINTERS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_CHUNK_SIZE = env2num<int64_t>("NCCL_CHUNK_SIZE", "0");
  NCCL_CHUNK_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_COLLNET_ENABLE = env2str("NCCL_COLLNET_ENABLE", "");
  NCCL_COLLNET_ENABLE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_COLLNET_NODE_THRESHOLD = env2num<int64_t>("NCCL_COLLNET_NODE_THRESHOLD", "2");
  NCCL_COLLNET_NODE_THRESHOLD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_COMM_BLOCKING = env2num<int64_t>("NCCL_COMM_BLOCKING", "-1");
  NCCL_COMM_BLOCKING_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_COMM_ID = env2str("NCCL_COMM_ID", "");
  NCCL_COMM_ID_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_COMM_SPLIT_SHARE_RESOURCES = env2num<int>("NCCL_COMM_SPLIT_SHARE_RESOURCES", "MIN");
  NCCL_COMM_SPLIT_SHARE_RESOURCES_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "MIN");

  NCCL_CONNECT_ROUND_MAX_PEERS = env2num<int64_t>("NCCL_CONNECT_ROUND_MAX_PEERS", "128");
  NCCL_CONNECT_ROUND_MAX_PEERS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "128");

  NCCL_CREATE_THREAD_CONTEXT = env2num<int64_t>("NCCL_CREATE_THREAD_CONTEXT", "0");
  NCCL_CREATE_THREAD_CONTEXT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_CROSS_NIC = env2num<int64_t>("NCCL_CROSS_NIC", "2");
  NCCL_CROSS_NIC_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_CTRAN_AG_RD_RTR = env2bool("NCCL_CTRAN_AG_RD_RTR", "True");
  NCCL_CTRAN_AG_RD_RTR_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "True");

  NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS = env2num<int>("NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS", "64");
  NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "64");

  NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE = env2num<int>("NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE", "640");
  NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "640");

  NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS = env2num<int>("NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS", "-1");
  NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE = env2num<int>("NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE", "-1");
  NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_CTRAN_ALLTOALL_THRESHOLD = env2num<uint64_t>("NCCL_CTRAN_ALLTOALL_THRESHOLD", "32768");
  NCCL_CTRAN_ALLTOALL_THRESHOLD_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "32768");

  {
    NCCL_CTRAN_BACKENDS.clear();
    auto tokens = env2strlist("NCCL_CTRAN_BACKENDS", "ib");
    for (auto token : tokens) {
      if (token == std::string("ib")) {
        NCCL_CTRAN_BACKENDS.emplace_back(NCCL_CTRAN_BACKENDS::ib);
      } else {
        CVAR_WARN_UNKNOWN_VALUE("NCCL_CTRAN_BACKENDS", token.c_str());
      }
    }
  }
  NCCL_CTRAN_BACKENDS_DEFAULT.clear();
  NCCL_CTRAN_BACKENDS_DEFAULT.emplace_back(NCCL_CTRAN_BACKENDS::ib);

  NCCL_CTRAN_IB_CTRL_TC = env2num<uint64_t>("NCCL_CTRAN_IB_CTRL_TC", "192");
  NCCL_CTRAN_IB_CTRL_TC_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "192");

  NCCL_CTRAN_IB_MAX_QPS = env2num<int>("NCCL_CTRAN_IB_MAX_QPS", "1");
  NCCL_CTRAN_IB_MAX_QPS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_CTRAN_IB_QP_SCALING_THRESHOLD = env2num<uint64_t>("NCCL_CTRAN_IB_QP_SCALING_THRESHOLD", "1048576");
  NCCL_CTRAN_IB_QP_SCALING_THRESHOLD_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "1048576");

  NCCL_CTRAN_IB_TRAFFIC_PROFILNG = env2bool("NCCL_CTRAN_IB_TRAFFIC_PROFILNG", "False");
  NCCL_CTRAN_IB_TRAFFIC_PROFILNG_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_CTRAN_KINETO_PROFILE_DIR = env2str("NCCL_CTRAN_KINETO_PROFILE_DIR", "/tmp");
  NCCL_CTRAN_KINETO_PROFILE_DIR_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "/tmp");

  NCCL_CTRAN_NUM_KERNEL_P2PELEMS = env2num<int>("NCCL_CTRAN_NUM_KERNEL_P2PELEMS", "65536");
  NCCL_CTRAN_NUM_KERNEL_P2PELEMS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "65536");

  if (getenv("NCCL_CTRAN_PROFILING") == nullptr) {
    NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::none;
  } else {
    std::string str(getenv("NCCL_CTRAN_PROFILING"));
    if (str == std::string("none")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::none;
    } else if (str == std::string("stdout")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::stdout;
    } else if (str == std::string("info")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::info;
    } else if (str == std::string("kineto")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::kineto;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_CTRAN_PROFILING", str.c_str());
    }
  }
  NCCL_CTRAN_PROFILING_DEFAULT = NCCL_CTRAN_PROFILING::none;

  NCCL_CTRAN_PROFILING_REPORT_COUNT = env2num<int>("NCCL_CTRAN_PROFILING_REPORT_COUNT", "100");
  NCCL_CTRAN_PROFILING_REPORT_COUNT_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "100");

  if (getenv("NCCL_CTRAN_REGISTER") == nullptr) {
    NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::lazy;
  } else {
    std::string str(getenv("NCCL_CTRAN_REGISTER"));
    if (str == std::string("none")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::none;
    } else if (str == std::string("lazy")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::lazy;
    } else if (str == std::string("eager")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::eager;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_CTRAN_REGISTER", str.c_str());
    }
  }
  NCCL_CTRAN_REGISTER_DEFAULT = NCCL_CTRAN_REGISTER::lazy;

  NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = env2num<int>("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT", "-1");
  NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_CTRAN_RING_MAX_OUTSTANDING = env2num<int>("NCCL_CTRAN_RING_MAX_OUTSTANDING", "8");
  NCCL_CTRAN_RING_MAX_OUTSTANDING_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "8");

  NCCL_CTRAN_RING_STEP = env2num<uint64_t>("NCCL_CTRAN_RING_STEP", "4194304");
  NCCL_CTRAN_RING_STEP_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "4194304");

  NCCL_CTRAN_SHARED_DEVBUF_SIZE = env2num<uint64_t>("NCCL_CTRAN_SHARED_DEVBUF_SIZE", "8388608");
  NCCL_CTRAN_SHARED_DEVBUF_SIZE_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "8388608");

  NCCL_CTRAN_TOPO_FILE = env2str("NCCL_CTRAN_TOPO_FILE", "");
  NCCL_CTRAN_TOPO_FILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_CTRAN_TOPO_FILE_KEYS.clear();
  NCCL_CTRAN_TOPO_FILE_KEYS = env2strlist("NCCL_CTRAN_TOPO_FILE_KEYS", "");
  NCCL_CTRAN_TOPO_FILE_KEYS_DEFAULT.clear();
  NCCL_CTRAN_TOPO_FILE_KEYS_DEFAULT = env2strlist("NCCL_ENV_DO_NOT_SET", "");

  NCCL_CUDA_PATH = env2str("NCCL_CUDA_PATH", "");
  NCCL_CUDA_PATH_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_CUMEM_ENABLE = env2num<int64_t>("NCCL_CUMEM_ENABLE", "-2");
  NCCL_CUMEM_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_DDA_ALLREDUCE_MAX_BLOCKS = env2num<int>("NCCL_DDA_ALLREDUCE_MAX_BLOCKS", "24");
  NCCL_DDA_ALLREDUCE_MAX_BLOCKS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "24");

  NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD = env2num<uint64_t>("NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD", "1048576");
  NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "1048576");

  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD = env2num<uint64_t>("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD", "262144");
  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "262144");

  NCCL_DDA_TMPBUFF_SIZE = env2num<uint64_t>("NCCL_DDA_TMPBUFF_SIZE", "33554432");
  NCCL_DDA_TMPBUFF_SIZE_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "33554432");

  NCCL_DEBUG = env2str("NCCL_DEBUG", "");
  NCCL_DEBUG_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_DEBUG_FILE = env2str("NCCL_DEBUG_FILE", "");
  NCCL_DEBUG_FILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_DEBUG_SUBSYS = env2str("NCCL_DEBUG_SUBSYS", "");
  NCCL_DEBUG_SUBSYS_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_DMABUF_ENABLE = env2num<int64_t>("NCCL_DMABUF_ENABLE", "1");
  NCCL_DMABUF_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_GDRCOPY_ENABLE = env2num<int64_t>("NCCL_GDRCOPY_ENABLE", "0");
  NCCL_GDRCOPY_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_GDRCOPY_FIFO_ENABLE = env2num<int64_t>("NCCL_GDRCOPY_FIFO_ENABLE", "1");
  NCCL_GDRCOPY_FIFO_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_GDRCOPY_FLUSH_ENABLE = env2bool("NCCL_GDRCOPY_FLUSH_ENABLE", "False");
  NCCL_GDRCOPY_FLUSH_ENABLE_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_GDRCOPY_SYNC_ENABLE = env2bool("NCCL_GDRCOPY_SYNC_ENABLE", "True");
  NCCL_GDRCOPY_SYNC_ENABLE_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "True");

  NCCL_GDR_FLUSH_DISABLE = env2num<int64_t>("NCCL_GDR_FLUSH_DISABLE", "0");
  NCCL_GDR_FLUSH_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_GRAPH_DUMP_FILE = env2str("NCCL_GRAPH_DUMP_FILE", "");
  NCCL_GRAPH_DUMP_FILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_GRAPH_DUMP_FILE_RANK = env2num<int64_t>("NCCL_GRAPH_DUMP_FILE_RANK", "0");
  NCCL_GRAPH_DUMP_FILE_RANK_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_GRAPH_FILE = env2str("NCCL_GRAPH_FILE", "");
  NCCL_GRAPH_FILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_GRAPH_MIXING_SUPPORT = env2bool("NCCL_GRAPH_MIXING_SUPPORT", "True");
  NCCL_GRAPH_MIXING_SUPPORT_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "True");

  NCCL_GRAPH_REGISTER = env2num<int64_t>("NCCL_GRAPH_REGISTER", "1");
  NCCL_GRAPH_REGISTER_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_HOSTID = env2str("NCCL_HOSTID", "");
  NCCL_HOSTID_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_IB_ADAPTIVE_ROUTING = env2num<int64_t>("NCCL_IB_ADAPTIVE_ROUTING", "-2");
  NCCL_IB_ADAPTIVE_ROUTING_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_IB_AR_THRESHOLD = env2num<int64_t>("NCCL_IB_AR_THRESHOLD", "8192");
  NCCL_IB_AR_THRESHOLD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "8192");

  NCCL_IB_DISABLE = env2num<int64_t>("NCCL_IB_DISABLE", "0");
  NCCL_IB_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_GID_INDEX = env2num<int64_t>("NCCL_IB_GID_INDEX", "0");
  NCCL_IB_GID_INDEX_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  std::vector<std::string> NCCL_IB_HCA_allPrefixes{"^", "="};
  NCCL_IB_HCA.clear();
  std::tie(NCCL_IB_HCA_PREFIX, NCCL_IB_HCA) = env2prefixedStrlist("NCCL_IB_HCA", "", NCCL_IB_HCA_allPrefixes);
  NCCL_IB_HCA_DEFAULT.clear();
  std::tie(NCCL_IB_HCA_PREFIX_DEFAULT, NCCL_IB_HCA_DEFAULT) = env2prefixedStrlist("NCCL_ENV_DO_NOT_SET", "", NCCL_IB_HCA_allPrefixes);

  NCCL_IB_MERGE_VFS = env2num<int64_t>("NCCL_IB_MERGE_VFS", "1");
  NCCL_IB_MERGE_VFS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_IB_PCI_RELAXED_ORDERING = env2num<int64_t>("NCCL_IB_PCI_RELAXED_ORDERING", "2");
  NCCL_IB_PCI_RELAXED_ORDERING_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_IB_PKEY = env2num<int64_t>("NCCL_IB_PKEY", "0");
  NCCL_IB_PKEY_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_QPS_PER_CONNECTION = env2num<int64_t>("NCCL_IB_QPS_PER_CONNECTION", "1");
  NCCL_IB_QPS_PER_CONNECTION_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_IB_RETRY_CNT = env2num<int64_t>("NCCL_IB_RETRY_CNT", "7");
  NCCL_IB_RETRY_CNT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "7");

  NCCL_IB_SL = env2num<int64_t>("NCCL_IB_SL", "0");
  NCCL_IB_SL_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_SPLIT_DATA_ON_QPS = env2num<int64_t>("NCCL_IB_SPLIT_DATA_ON_QPS", "1");
  NCCL_IB_SPLIT_DATA_ON_QPS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_IB_TC = env2num<int64_t>("NCCL_IB_TC", "0");
  NCCL_IB_TC_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_TIMEOUT = env2num<int64_t>("NCCL_IB_TIMEOUT", "18");
  NCCL_IB_TIMEOUT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "18");

  NCCL_IB_USE_INLINE = env2num<int64_t>("NCCL_IB_USE_INLINE", "0");
  NCCL_IB_USE_INLINE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IGNORE_CPU_AFFINITY = env2num<int64_t>("NCCL_IGNORE_CPU_AFFINITY", "0");
  NCCL_IGNORE_CPU_AFFINITY_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IGNORE_DISABLED_P2P = env2num<int64_t>("NCCL_IGNORE_DISABLED_P2P", "0");
  NCCL_IGNORE_DISABLED_P2P_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_L1_SHARED_MEMORY_CARVEOUT = env2num<int>("NCCL_L1_SHARED_MEMORY_CARVEOUT", "0");
  NCCL_L1_SHARED_MEMORY_CARVEOUT_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_LAUNCH_MODE = env2str("NCCL_LAUNCH_MODE", "");
  NCCL_LAUNCH_MODE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_LL128_BUFFSIZE = env2num<int64_t>("NCCL_LL128_BUFFSIZE", "-2");
  NCCL_LL128_BUFFSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_LL128_NTHREADS = env2num<int64_t>("NCCL_LL128_NTHREADS", "-2");
  NCCL_LL128_NTHREADS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_LL_BUFFSIZE = env2num<int64_t>("NCCL_LL_BUFFSIZE", "-2");
  NCCL_LL_BUFFSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_LOCAL_REGISTER = env2num<int64_t>("NCCL_LOCAL_REGISTER", "0");
  NCCL_LOCAL_REGISTER_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  if (getenv("NCCL_LOGGER_MODE") == nullptr) {
    NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::sync;
  } else {
    std::string str(getenv("NCCL_LOGGER_MODE"));
    if (str == std::string("sync")) {
      NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::sync;
    } else if (str == std::string("async")) {
      NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::async;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_LOGGER_MODE", str.c_str());
    }
  }
  NCCL_LOGGER_MODE_DEFAULT = NCCL_LOGGER_MODE::sync;

  NCCL_MAX_CTAS = env2num<int>("NCCL_MAX_CTAS", "MIN");
  NCCL_MAX_CTAS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "MIN");

  NCCL_MAX_NCHANNELS = env2num<int>("NCCL_MAX_NCHANNELS", "-2");
  NCCL_MAX_NCHANNELS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MAX_NRINGS = env2num<int>("NCCL_MAX_NRINGS", "-2");
  NCCL_MAX_NRINGS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MAX_P2P_NCHANNELS = env2num<int>("NCCL_MAX_P2P_NCHANNELS", "MAX");
  NCCL_MAX_P2P_NCHANNELS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "MAX");

  if (getenv("NCCL_MEM_SYNC_DOMAIN") == nullptr) {
    NCCL_MEM_SYNC_DOMAIN = NCCL_MEM_SYNC_DOMAIN::remote;
  } else {
    std::string str(getenv("NCCL_MEM_SYNC_DOMAIN"));
    if (str == std::string("local")) {
      NCCL_MEM_SYNC_DOMAIN = NCCL_MEM_SYNC_DOMAIN::local;
    } else if (str == std::string("remote")) {
      NCCL_MEM_SYNC_DOMAIN = NCCL_MEM_SYNC_DOMAIN::remote;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_MEM_SYNC_DOMAIN", str.c_str());
    }
  }
  NCCL_MEM_SYNC_DOMAIN_DEFAULT = NCCL_MEM_SYNC_DOMAIN::remote;

  NCCL_MIN_CTAS = env2num<int>("NCCL_MIN_CTAS", "MIN");
  NCCL_MIN_CTAS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "MIN");

  NCCL_MIN_NCHANNELS = env2num<int>("NCCL_MIN_NCHANNELS", "-2");
  NCCL_MIN_NCHANNELS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MIN_NRINGS = env2num<int>("NCCL_MIN_NRINGS", "-2");
  NCCL_MIN_NRINGS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MIN_P2P_NCHANNELS = env2num<int>("NCCL_MIN_P2P_NCHANNELS", "1");
  NCCL_MIN_P2P_NCHANNELS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_NCHANNELS_PER_NET_PEER = env2num<int64_t>("NCCL_NCHANNELS_PER_NET_PEER", "2");
  NCCL_NCHANNELS_PER_NET_PEER_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_NETWORK = env2str("NCCL_NET", "");
  NCCL_NETWORK_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_NET_DISABLE_INTRA = env2num<int64_t>("NCCL_NET_DISABLE_INTRA", "0");
  NCCL_NET_DISABLE_INTRA_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_NET_FORCE_FLUSH = env2num<int64_t>("NCCL_NET_FORCE_FLUSH", "0");
  NCCL_NET_FORCE_FLUSH_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_NET_GDR_LEVEL = env2str("NCCL_NET_GDR_LEVEL", "");
  NCCL_NET_GDR_LEVEL_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_NET_GDR_READ = env2num<int64_t>("NCCL_NET_GDR_READ", "-2");
  NCCL_NET_GDR_READ_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NET_OVERHEAD = env2num<int64_t>("NCCL_NET_OVERHEAD", "-2");
  NCCL_NET_OVERHEAD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NET_PLUGIN = env2str("NCCL_NET_PLUGIN", "libnccl-net.so");
  NCCL_NET_PLUGIN_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "libnccl-net.so");

  NCCL_NET_SHARED_BUFFERS = env2num<int64_t>("NCCL_NET_SHARED_BUFFERS", "-2");
  NCCL_NET_SHARED_BUFFERS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NET_SHARED_COMMS = env2bool("NCCL_NET_SHARED_COMMS", "True");
  NCCL_NET_SHARED_COMMS_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "True");

  NCCL_NSOCKS_PERTHREAD = env2num<int>("NCCL_NSOCKS_PERTHREAD", "-2");
  NCCL_NSOCKS_PERTHREAD_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NTHREADS = env2num<int64_t>("NCCL_NTHREADS", "-2");
  NCCL_NTHREADS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NVB_DISABLE = env2num<int64_t>("NCCL_NVB_DISABLE", "0");
  NCCL_NVB_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_NVB_PRECONNECT = env2num<int64_t>("NCCL_NVB_PRECONNECT", "1");
  NCCL_NVB_PRECONNECT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_NVLS_ENABLE = env2num<int64_t>("NCCL_NVLS_ENABLE", "2");
  NCCL_NVLS_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_NVLS_NCHANNELS = env2num<int>("NCCL_NVLS_NCHANNELS", "16");
  NCCL_NVLS_NCHANNELS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "16");

  NCCL_P2P_DIRECT_DISABLE = env2bool("NCCL_P2P_DIRECT_DISABLE", "False");
  NCCL_P2P_DIRECT_DISABLE_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_P2P_DISABLE = env2bool("NCCL_P2P_DISABLE", "False");
  NCCL_P2P_DISABLE_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_P2P_LEVEL = env2str("NCCL_P2P_LEVEL", "");
  NCCL_P2P_LEVEL_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_P2P_LL_THRESHOLD = env2num<int64_t>("NCCL_P2P_LL_THRESHOLD", "16384");
  NCCL_P2P_LL_THRESHOLD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "16384");

  NCCL_P2P_NET_CHUNKSIZE = env2num<int64_t>("NCCL_P2P_NET_CHUNKSIZE", "131072");
  NCCL_P2P_NET_CHUNKSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "131072");

  NCCL_P2P_NVL_CHUNKSIZE = env2num<int64_t>("NCCL_P2P_NVL_CHUNKSIZE", "524288");
  NCCL_P2P_NVL_CHUNKSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "524288");

  NCCL_P2P_PCI_CHUNKSIZE = env2num<int64_t>("NCCL_P2P_PCI_CHUNKSIZE", "131072");
  NCCL_P2P_PCI_CHUNKSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "131072");

  NCCL_P2P_PXN_LEVEL = env2num<int64_t>("NCCL_P2P_PXN_LEVEL", "2");
  NCCL_P2P_PXN_LEVEL_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_P2P_READ_ENABLE = env2num<int64_t>("NCCL_P2P_READ_ENABLE", "-2");
  NCCL_P2P_READ_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_P2P_USE_CUDA_MEMCPY = env2num<int64_t>("NCCL_P2P_USE_CUDA_MEMCPY", "0");
  NCCL_P2P_USE_CUDA_MEMCPY_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_PROGRESS_APPENDOP_FREQ = env2num<int64_t>("NCCL_PROGRESS_APPENDOP_FREQ", "8");
  NCCL_PROGRESS_APPENDOP_FREQ_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "8");

  NCCL_PROTO = env2str("NCCL_PROTO", "");
  NCCL_PROTO_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_PROXY_APPEND_BATCH_SIZE = env2num<int64_t>("NCCL_PROXY_APPEND_BATCH_SIZE", "16");
  NCCL_PROXY_APPEND_BATCH_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "16");

  NCCL_PROXY_DUMP_SIGNAL = env2num<int64_t>("NCCL_PROXY_DUMP_SIGNAL", "-1");
  NCCL_PROXY_DUMP_SIGNAL_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_PROXY_PROFILE = env2str("NCCL_PROXY_PROFILE", "");
  NCCL_PROXY_PROFILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_PROXY_PROFILE_DIR = env2str("NCCL_PROXY_PROFILE_DIR", "/tmp");
  NCCL_PROXY_PROFILE_DIR_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "/tmp");

  NCCL_PXN_DISABLE = env2num<int64_t>("NCCL_PXN_DISABLE", "0");
  NCCL_PXN_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_REPORT_CONNECT_PROGRESS = env2num<int64_t>("NCCL_REPORT_CONNECT_PROGRESS", "0");
  NCCL_REPORT_CONNECT_PROGRESS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  if (getenv("NCCL_SENDRECV_ALGO") == nullptr) {
    NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_SENDRECV_ALGO"));
    if (str == std::string("orig")) {
      NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;
    } else if (str == std::string("ctran")) {
      NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::ctran;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_SENDRECV_ALGO", str.c_str());
    }
  }
  NCCL_SENDRECV_ALGO_DEFAULT = NCCL_SENDRECV_ALGO::orig;

  NCCL_SET_STACK_SIZE = env2num<int64_t>("NCCL_SET_STACK_SIZE", "0");
  NCCL_SET_STACK_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_SET_THREAD_NAME = env2num<int64_t>("NCCL_SET_THREAD_NAME", "0");
  NCCL_SET_THREAD_NAME_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_SHM_DISABLE = env2bool("NCCL_SHM_DISABLE", "False");
  NCCL_SHM_DISABLE_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_SHM_LOCALITY = env2num<int64_t>("NCCL_SHM_LOCALITY", "2");
  NCCL_SHM_LOCALITY_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_SHM_MEMCPY_MODE = env2num<int64_t>("NCCL_SHM_MEMCPY_MODE", "1");
  NCCL_SHM_MEMCPY_MODE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_SHM_USE_CUDA_MEMCPY = env2bool("NCCL_SHM_USE_CUDA_MEMCPY", "False");
  NCCL_SHM_USE_CUDA_MEMCPY_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_SOCKET_FAMILY = env2str("NCCL_SOCKET_FAMILY", "");
  NCCL_SOCKET_FAMILY_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_SOCKET_IFNAME = env2str("NCCL_SOCKET_IFNAME", "");
  NCCL_SOCKET_IFNAME_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_SOCKET_NTHREADS = env2num<int>("NCCL_SOCKET_NTHREADS", "-2");
  NCCL_SOCKET_NTHREADS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_THREAD_THRESHOLDS = env2str("NCCL_THREAD_THRESHOLDS", "");
  NCCL_THREAD_THRESHOLDS_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_TOPO_DUMP_FILE = env2str("NCCL_TOPO_DUMP_FILE", "");
  NCCL_TOPO_DUMP_FILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_TOPO_DUMP_FILE_RANK = env2num<int64_t>("NCCL_TOPO_DUMP_FILE_RANK", "0");
  NCCL_TOPO_DUMP_FILE_RANK_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_TOPO_FILE = env2str("NCCL_TOPO_FILE", "/var/run/nvidia-topologyd/virtualTopology.xml");
  NCCL_TOPO_FILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "/var/run/nvidia-topologyd/virtualTopology.xml");

  NCCL_TUNER_PLUGIN = env2str("NCCL_TUNER_PLUGIN", "");
  NCCL_TUNER_PLUGIN_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_WARN_ENABLE_DEBUG_INFO = env2num<int64_t>("NCCL_WARN_ENABLE_DEBUG_INFO", "0");
  NCCL_WARN_ENABLE_DEBUG_INFO_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_WORK_FIFO_DEPTH = env2num<int64_t>("NCCL_WORK_FIFO_DEPTH", "65536");
  NCCL_WORK_FIFO_DEPTH_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "65536");

}

// Automatically generated by ./maint/extractcvars.py --- END

