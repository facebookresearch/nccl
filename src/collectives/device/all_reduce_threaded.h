// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if !defined (ALL_REDUCE_THREADED_H_INCLUDED)
#define ALL_REDUCE_THREADED_H_INCLUDED

#define DECL_THREADED_FUNC_NRANKS(T,NRANKS)                             \
    template                                                            \
    __global__ void ncclKernel_AllReduce_Threaded_Flat<T,NRANKS>(uintptr_t *barrierMbox, \
                                                                 uintptr_t barrierFlag, int rank, \
                                                                 const T *sendbuff, T *recvbuff, size_t count); \
    template                                                            \
    __global__ void ncclKernel_AllReduce_Threaded_Tree<T,NRANKS>(uintptr_t *barrierMbox, uintptr_t barrierFlag, \
                                                                 int rank, const T *sendbuff, T *tmpbuff, T *recvbuff, \
                                                                 size_t count); \
    template                                                            \
    __global__ void ncclKernel_AllReduce_Threaded_HCM_Flat<T,NRANKS>(uintptr_t *cliqueBarrierMbox, uintptr_t *localMbox, \
                                                                     uintptr_t *peerMbox, uintptr_t barrierFlag, int cliqueRank, \
                                                                     const T *sendbuff, T *tmpbuff, T *recvbuff, size_t count); \
    template                                                            \
    __global__ void ncclKernel_AllReduce_Threaded_HCM_Tree<T,NRANKS>(uintptr_t *cliqueBarrierMbox, uintptr_t *localMbox, \
                                                                     uintptr_t *peerMbox, uintptr_t barrierFlag, int cliqueRank, \
                                                                     const T *sendbuff, T *tmpbuff, T *recvbuff, size_t count)

#define DECL_THREADED_FUNC(T)                                           \
    DECL_THREADED_FUNC_NRANKS(T,2);                                     \
    DECL_THREADED_FUNC_NRANKS(T,4);                                     \
    DECL_THREADED_FUNC_NRANKS(T,8);                                     \
    DECL_THREADED_FUNC_NRANKS(T,16)

#endif /* ALL_REDUCE_THREADED_H_INCLUDED */
