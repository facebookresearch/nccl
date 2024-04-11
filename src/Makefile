#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../makefiles/common.mk
include ../makefiles/version.mk

##### src files
INCEXPORTS  := nccl.h nccl_net.h
LIBSRCFILES := init.cc init_nvtx.cc channel.cc bootstrap.cc transport.cc enqueue.cc group.cc debug.cc proxy.cc net.cc \
		misc/cudawrap.cc misc/nvmlwrap.cc misc/ibvsymbols.cc misc/ibvwrap.cc misc/gdrwrap.cc \
		misc/utils.cc misc/argcheck.cc misc/socket.cc misc/shmutils.cc misc/profiler.cc misc/param.cc misc/strongstream.cc \
		misc/ipcsocket.cc \
		transport/p2p.cc transport/shm.cc transport/net.cc transport/net_socket.cc transport/net_ib.cc transport/coll_net.cc transport/nvls.cc \
                collectives/sendrecv.cc collectives/all_reduce.cc collectives/all_gather.cc collectives/broadcast.cc collectives/reduce.cc collectives/reduce_scatter.cc \
                graph/topo.cc graph/paths.cc graph/search.cc graph/connect.cc graph/rings.cc graph/trees.cc graph/tuning.cc graph/xml.cc
LIBSRCFILES += collectives/all_reduce_sparse_block.cc setinfo.cc commSplitType.cc
LIBSRCFILES += misc/tuner.cc
LIBSRCFILES += misc/nccl_cvars.cc
LIBSRCFILES += ctran/backends/ib/CtranIb.cc ctran/backends/ib/CtranIbImpl.cc \
			   ctran/backends/ib/CtranIbRequest.cc ctran/backends/ib/CtranIbVc.cc
LIBSRCFILES += ctran/gpe/CtranGpe.cc ctran/gpe/CtranGpeImpl.cc
LIBSRCFILES += ctran/mapper/CtranMapper.cc ctran/mapper/CtranMapperRequest.cc
LIBSRCFILES += ctran/Ctran.cc
LIBSRCFILES += ctran/utils/CtranAvlTree.cc ctran/utils/CtranAvlTreeElem.cc ctran/utils/CtranTopoFile.cc
LIBSRCFILES += ctran/algos/CtranAlgo.cc
LIBSRCFILES += ctran/algos/sendrecv.cc ctran/algos/AllGather/AllGatherDirect.cc ctran/algos/AllGather/AllGatherRing.cc ctran/algos/AllGather/AllGatherRecDbl.cc
LIBSRCFILES += ctran/algos/AllToAll/AllToAllvImpl.cc ctran/algos/AllToAll/AllToAll.cc ctran/algos/AllToAll/AllToAllv.cc
LIBSRCFILES += algorithms/AlgoDirector.cc \
               algorithms/AlgoInit.cc \
               algorithms/AlgoManagerBase.cc \
               algorithms/AlgoUtils.cc \
               algorithms/DdaMemHandler.cc \
               algorithms/DdaThreadedData.cc \
               algorithms/allreduce/AlgoAllReduceDdaNvsFlatIpc.cc \
               algorithms/allreduce/AlgoAllReduceDdaNvsFlatThreaded.cc \
               algorithms/allreduce/AlgoAllReduceDdaNvsTreeIpc.cc \
               algorithms/allreduce/AlgoAllReduceDdaNvsTreeThreaded.cc \
               algorithms/allreduce/AlgoAllReduceDdaNvsScatGatIpc.cc \
               algorithms/allreduce/AlgoManagerAllReduce.cc
LIBSRCFILES += collectives/all_to_allv.cc collectives/all_to_all.cc
LIBSRCFILES += colltrace/CollTrace.cc
LIBSRCFILES += colltrace/ProxyTrace.cc colltrace/ProxyMock.cc
LIBSRCFILES += logger/Logger.cc logger/EventMgr.cc
LIBSRCFILES += window.cc
LIBSRCFILES += commHash.cc
LIBSRCFILES += commDump.cc

INCLUDES := -Iinclude
INCLUDES += -Ialgorithms -Ialgorithms/allreduce
INCLUDES += -Ictran -Ictran/utils -Ictran/backends/ib -Ictran/mapper -Ictran/gpe -Ictran/algos -Icolltrace/ -Ilogger/
INCLUDES += -Ictran/algos/AllToAll

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### pkgconfig files
PKGCONFIGFILE := nccl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
##### target files
CUDARTLIB  ?= cudart_static

ifeq ($(CUDARTLIB), cudart_static)
	# Use compatibility shim only with static cudart; see https://github.com/NVIDIA/nccl/issues/658
	LIBSRCFILES += enhcompat.cc
endif

INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d)
LDFLAGS    += -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl

DEVICELIB  := $(BUILDDIR)/obj/collectives/device/colldevice.a

##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

$(DEVICELIB): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C collectives/device

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in ../makefiles/version.mk
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    -e "s/\$${nccl:DevSignature}/$(DEV_SIGNATURE)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVICELIB)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(DEVICELIB) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

null :=
space := $(null) #
comma := ,

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVICELIB)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	printf "create $@\naddlib $(DEVICELIB)\naddmod $(subst $(space),$(comma),$(strip $(LIBOBJ)))\nsave\nend" | ar -M

$(PKGDIR)/nccl.pc : nccl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${nccl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	install -m 644 $< $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	$(MAKE) -C collectives/device clean
	rm -rf ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
