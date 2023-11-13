/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_ERR_H_
#define NCCL_ERR_H_

/* *INDENT-OFF* */
/* Error type for plugins */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclRemoteError             =  6 } ncclResult_t;
/* *INDENT-ON* */

#endif
