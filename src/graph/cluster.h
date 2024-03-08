// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_CLUSTER_H_
#define NCCL_CLUSTER_H_

#include "core.h"
#include "graph.h"

#define NCCL_TOPO_CLUSTER_NODE_TYPES 7
#define CLUSTER_SUPER_SPINE 0
#define CLUSTER_SPINE 1
#define CLUSTER_LEAF 2
#define CLUSTER_HOST 3
#define CLUSTER_NIC 4
extern const char* topoClusterNodeTypeStr[];

// FQDN: Fully Qualified Domain Name
#define MAX_FQDN_SIZE 255
#define NCCL_CLUSTER_TOPO_UNDEF (-1)

struct ncclTopoClusterNode {
  int type; // SWITCH types, HOST, NIC
  // Type specific data
  union {
    // E.g., super_spine id="c082" speed="28571" latency="23"
    struct {
      char name[MAX_FQDN_SIZE];
      float bw;
      float latency;
    }superSpine;
    // E.g., spine id="ctsw001.c082.f00.ftw6.tfbnw.net" speed="200000" latency="15"
    struct {
      char name[MAX_FQDN_SIZE];
      float bw;
      float latency;
    }spine;
    // E.g., leaf id="rtsw040.c082.f00.ftw6.tfbnw.net" speed="200000" latency="7"
    struct {
      char name[MAX_FQDN_SIZE];
      float bw;
      float latency;
    }leaf;
    // E.g., host id="rtptest1685.ftw6.facebook.com"
    struct {
      char name[MAX_FQDN_SIZE];
    }host;
    // E.g., nic id="mlx5_0" dev="0" speed="200000"
    struct {
      char name[MAX_FQDN_SIZE];
      int dev;
      float bw;
    }nic;
  };
  struct ncclTopoClusterNode *parent;
  int children;
};

struct ncclTopoClusterSet {
  int count;
  struct ncclTopoClusterNode nodes[NCCL_TOPO_MAX_CLUSTER_NODES];
};

struct ncclTopoCluster {
  struct ncclTopoClusterSet nodes[NCCL_TOPO_CLUSTER_NODE_TYPES];
  int topologyType; // FAT tree, dragonfly etc.
  int hosts;
  int nics;
};

ncclResult_t ncclTopoGetClusterFromXml(struct ncclXml* xml, struct ncclTopoCluster** topoCluster);

#endif
