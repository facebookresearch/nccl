/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"
#include "nccl_cvars.h"
#include "Logger.h"
#include "tuner.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>

static const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

static void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[s] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    setenv(envVar, envValue, 0);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

static pthread_mutex_t initEnvLock = PTHREAD_MUTEX_INITIALIZER;
static bool isInitialized = false;

void initEnv() {
  /* make sure initEnv is called only once.  Use a lock in case
   * multiple threads initialize it simultaneously. */
  if (__atomic_load_n(&isInitialized, __ATOMIC_ACQUIRE)) {
    return;
  }
  pthread_mutex_lock(&initEnvLock);
  isInitialized = true;

  char confFilePath[1024];
  const char * userDir = userHomeDir();
  if (userDir) {
    sprintf(confFilePath, "%s/.nccl.conf", userDir);
    setEnvFile(confFilePath);
  }
  sprintf(confFilePath, "/etc/nccl.conf");
  setEnvFile(confFilePath);

  // Load tuner plugin after set default config file and before ncclCvarInit
  // to allow tuner plugin overwrite any environment variables if specified
  // FIXME: the INFO log in ncclLoadTunerPlugin won't be printed out since CVARs are not initialized yet
  ncclTuner_t *tuner;
  ncclLoadTunerPlugin(&tuner);
  if (tuner) {
    // 0 indicates that the tuner is one-off (e.g., set global variables if a tuning file specified)
    // and will not be associated with any communicator
    // it should be destroyed immediately
    tuner->init(0, 0, nullptr);
    tuner->destroy();
  }
  // reset debug level to let NCCL initialize it properly later after ncclCvarInit
  // FIXME: this is a temporary workaorund because ncclLoadTunerPlugin calls ncclDebugInit before ncclCvarInit below
  ncclDebugLevel = -1;

  ncclCvarInit();

  // Load Logger thread
  NcclLogger::init();

  __atomic_store_n(&isInitialized, true, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&initEnvLock);
}
