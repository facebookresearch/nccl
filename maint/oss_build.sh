#!/bin/bash

hash=""
# guess default DEV_SIGNATURE
is_git=$(git rev-parse --is-inside-work-tree)
is_hg=$(hg id)
if [ $is_git ]; then
    hash="git-"$(git rev-parse --short HEAD)
elif [ $is_hg ]; then
    hash="hg-"$(hg id)
else
    echo "Cannot detect source repository hash. Skip"
fi

set -e

BUILDDIR=${BUILDDIR:=/tmp/${USER}/nccl-exp/2.20.3/build}
NVCC_ARCH=${NVCC_ARCH:="a100,h100"}
CUDA_HOME=${CUDA_HOME:="`realpath ../../tp2/cuda/12.2.2/x86_64`"}
DEV_SIGNATURE=${DEV_SIGNATURE:="$hash"}
NCCL_FP8=${NCCL_FP8:=0}

echo "BUILDDIR=${BUILDDIR}"
echo "NVCC_ARCH=${NVCC_ARCH}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "DEV_SIGNATURE=${DEV_SIGNATURE}"
echo "NCCL_FP8=${NCCL_FP8}"

IFS=',' read -ra arch_array <<< "$NVCC_ARCH"
arch_gencode=""
for arch in "${arch_array[@]}"
do
    case "$arch" in
       "p100")
       arch_gencode="$arch_gencode -gencode=arch=compute_60,code=sm_60"
        ;;
       "v100")
       arch_gencode="$arch_gencode -gencode=arch=compute_70,code=sm_70"
        ;;
       "a100")
       arch_gencode="$arch_gencode -gencode=arch=compute_80,code=sm_80"
       ;;
       "h100")
        arch_gencode="$arch_gencode -gencode=arch=compute_90,code=sm_90"
       ;;
    esac
done

echo "NVCC_GENCODE=$arch_gencode"

# build libnccl
export BUILDDIR=$BUILDDIR
make -j src.build NVCC_GENCODE="$arch_gencode" CUDA_HOME="$CUDA_HOME" DEV_SIGNATURE="$DEV_SIGNATURE" NCCL_FP8="$NCCL_FP8"

# sanity check
pushd examples
export NCCL_HOME=$BUILDDIR
export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH=$BUILDDIR/lib
make all NVCC_GENCODE="$arch_gencode" CUDA_HOME="$CUDA_HOME" DEV_SIGNATURE="$DEV_SIGNATURE"

set +e

TIMEOUT=10s
timeout $TIMEOUT $BUILDDIR/examples/HelloWorld
if [ "$?" == "124" ]; then
    echo "Program TIMEOUT in ${TIMEOUT}. Terminate."
fi
popd
