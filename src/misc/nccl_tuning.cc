#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>

#include "debug.h"
#include "nccl_tuning.h"

void* tunerPluginLib = nullptr;

ncclResult_t ncclLoadPerformanceTuner(ncclPerformanceTuner_t** tuner) {
  // Initialize to nullptr by default if plugin tuner cannot be loaded.
  *tuner = nullptr;

  char ncclTunerPluginName[128];
  const char* name = getenv("NCCL_TUNER_PLUGIN");
  if (name && strlen(name)) {
    snprintf(ncclTunerPluginName, 128, "libnccl-tuner-%s.so", name);
    INFO(NCCL_INIT, "Plugin name set by env to %s", ncclTunerPluginName);
  } else {
    sprintf(ncclTunerPluginName, "libnccl-tuner.so");
  }

  void* tunerPluginLib = dlopen(ncclTunerPluginName, RTLD_LAZY | RTLD_LOCAL);
  if (tunerPluginLib == nullptr) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      INFO(NCCL_TUNING, "Performance tuner: no plugin found '%s', using default tuner instead.", name);
    } else {
      INFO(NCCL_TUNING, "Performance tuner: plugin load '%s' returned error (%d : %s), using default tuner instead.", name, errno, dlerror());
    }
    return ncclSystemError;
  }

 *tuner = (ncclPerformanceTuner_t*)dlsym(tunerPluginLib, NCCL_PERFORMANCE_TUNER_SYMBOL);
  if (*tuner == nullptr) {
    INFO(NCCL_TUNING, "Performance tuner: failed to find ncclPerformanceTunerSymbol in plugin (%s), using default tuner instead.", name);
    dlclose(tunerPluginLib);
    tunerPluginLib = nullptr;
    return ncclSystemError;
  }

  INFO(NCCL_TUNING, "Using performance tuner: '%s'", (*tuner)->name);
  return ncclSuccess;
}

ncclResult_t ncclClosePerformanceTuner(ncclPerformanceTuner_t** tuner) {
  if (tunerPluginLib != nullptr) {
    INFO(NCCL_TUNING, "Closing performance tuner: '%s'", (*tuner)->name);
    dlclose(tunerPluginLib);
  }
  tunerPluginLib = nullptr;
  *tuner = nullptr;
  return ncclSuccess;
}
