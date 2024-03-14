// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "cluster.h"
#include "cluster_xml.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_TOPO_CLUSTER_FILE
   type        : string
   default     : "/var/run/nvidia-topologyd/cluster.xml"
   description : |-
     Path to an cluster XML file to load.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

const char* topoClusterNodeTypeStr[] = { "CLUSTER_SUPER_SPINE", "CLUSTER_SPINE", "CLUSTER_LEAF", "CLUSTER_HOST", "CLUSTER_NIC" };

/******************************************************************/
/********* Network Cluster Topology Creation Functions ************/
/******************************************************************/

ncclResult_t ncclTopoCreateClusterNode(struct ncclTopoCluster* cluster, struct ncclTopoClusterNode** node, int type) {
  if (cluster->nodes[type].count == NCCL_TOPO_MAX_CLUSTER_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoClusterNode* n = cluster->nodes[type].nodes+cluster->nodes[type].count;
  cluster->nodes[type].count++;
  n->type = type;
  switch(type) {
    case CLUSTER_SUPER_SPINE:
      strcpy(n->superSpine.name, "");
      n->superSpine.bw = 0.0;
      n->superSpine.latency = 0.0;
      break;
    case CLUSTER_SPINE:
      strcpy(n->spine.name, "");
      n->spine.bw = 0.0;
      n->spine.latency = 0.0;
      break;
    case CLUSTER_LEAF:
      strcpy(n->leaf.name, "");
      n->leaf.bw = 0.0;
      n->leaf.latency = 0.0;
      break;
    case CLUSTER_HOST:
      strcpy(n->host.name, "");
      break;
    case CLUSTER_NIC:
      strcpy(n->nic.name, "");
      n->nic.dev= NCCL_CLUSTER_TOPO_UNDEF;
      n->nic.bw = 0.0;
      break;
    default:
      WARN("Error : unknown cluster node type %d", type);
      break;
  };
  n->parent = NULL;
  n->children = 0;
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoClusterPrint(struct ncclTopoCluster* s) {
  INFO(NCCL_GRAPH, "=== Cluster : hosts %d nics %d  ===", s->hosts, s->nics);
  int spines_traversed = 0;
  int leafs_traversed = 0;
  int hosts_traversed = 0;
  int nics_traversed = 0;
  for (int i=0; i<s->nodes[CLUSTER_SUPER_SPINE].count; i++) {
    INFO(NCCL_GRAPH, "  === super_spine %s speed %.2f latency %.2f ===",
        s->nodes[CLUSTER_SUPER_SPINE].nodes[i].superSpine.name,
        s->nodes[CLUSTER_SUPER_SPINE].nodes[i].superSpine.bw,
        s->nodes[CLUSTER_SUPER_SPINE].nodes[i].superSpine.latency);
    for(int j = 0; j<s->nodes[CLUSTER_SUPER_SPINE].nodes[i].children; j++) {
      INFO(NCCL_GRAPH, "    === spine %s speed %.2f latency %.2f ===",
          s->nodes[CLUSTER_SPINE].nodes[j+spines_traversed].spine.name,
          s->nodes[CLUSTER_SPINE].nodes[j+spines_traversed].spine.bw,
          s->nodes[CLUSTER_SPINE].nodes[j+spines_traversed].spine.latency);

      // Example on how to get to parent
      ncclTopoClusterNode *parentNode = s->nodes[CLUSTER_SPINE].nodes[j+spines_traversed].parent;
      INFO(NCCL_GRAPH, "    === spine_parent %s  ===", parentNode->superSpine.name);

      for(int k = 0; k<s->nodes[CLUSTER_SPINE].nodes[j+spines_traversed].children; k++) {
        INFO(NCCL_GRAPH, "      === leaf %s speed %.2f latency %.2f ===",
            s->nodes[CLUSTER_LEAF].nodes[k+leafs_traversed].leaf.name,
            s->nodes[CLUSTER_LEAF].nodes[k+leafs_traversed].leaf.bw,
            s->nodes[CLUSTER_LEAF].nodes[k+leafs_traversed].leaf.latency);
        for(int l = 0; l<s->nodes[CLUSTER_LEAF].nodes[k+leafs_traversed].children; l++) {
          INFO(NCCL_GRAPH, "        === host %s  ===",
              s->nodes[CLUSTER_HOST].nodes[l+hosts_traversed].host.name);
          for(int m = 0; m<s->nodes[CLUSTER_HOST].nodes[l+hosts_traversed].children; m++) {
            INFO(NCCL_GRAPH, "          === nic %s speed %.2f ===",
                s->nodes[CLUSTER_NIC].nodes[m+nics_traversed].nic.name,
                s->nodes[CLUSTER_NIC].nodes[m+nics_traversed].nic.bw);
          }
          nics_traversed += s->nodes[CLUSTER_HOST].nodes[l+hosts_traversed].children;
        }
        hosts_traversed += s->nodes[CLUSTER_LEAF].nodes[k+leafs_traversed].children;
      }
      leafs_traversed += s->nodes[CLUSTER_SPINE].nodes[j+spines_traversed].children;
    }
    spines_traversed += s->nodes[CLUSTER_SUPER_SPINE].nodes[i].children;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddClusterNic(struct ncclClusterXmlNode* xmlNic, struct ncclTopoCluster* cluster, struct ncclTopoClusterNode *host) {
  struct ncclTopoClusterNode* nic;
  NCCLCHECK(ncclTopoCreateClusterNode(cluster, &nic, CLUSTER_NIC));
  const char* str;
  NCCLCHECK(clusterXmlGetAttr(xmlNic, "id", &str));
  strcpy(nic->nic.name, str);
  clusterXmlGetAttrFloat(xmlNic, "speed", &nic->nic.bw);
  clusterXmlGetAttrInt(xmlNic, "dev", &nic->nic.dev);
  nic->parent = host;
  cluster->nics += 1;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddHost(struct ncclClusterXmlNode* xmlHost, struct ncclTopoCluster* cluster, struct ncclTopoClusterNode *leaf) {
  struct ncclTopoClusterNode* host;
  NCCLCHECK(ncclTopoCreateClusterNode(cluster, &host, CLUSTER_HOST));
  const char* str;
  NCCLCHECK(clusterXmlGetAttr(xmlHost, "id", &str));
  strcpy(host->host.name, str);
  host->parent = leaf;
  host->children = xmlHost->nSubs;
  for (int s=0; s<xmlHost->nSubs; s++) {
    struct ncclClusterXmlNode* node = xmlHost->subs[s];
    if (strcmp(node->name, "nic") == 0) NCCLCHECK(ncclTopoAddClusterNic(node, cluster, host));
  }
  cluster->hosts += 1;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddLeaf(struct ncclClusterXmlNode* xmlLeaf, struct ncclTopoCluster* cluster, struct ncclTopoClusterNode *spine) {
  struct ncclTopoClusterNode* leaf;
  NCCLCHECK(ncclTopoCreateClusterNode(cluster, &leaf, CLUSTER_LEAF));
  const char* str;
  NCCLCHECK(clusterXmlGetAttr(xmlLeaf, "id", &str));
  strcpy(leaf->leaf.name, str);
  clusterXmlGetAttrFloat(xmlLeaf, "speed", &leaf->leaf.bw);
  clusterXmlGetAttrFloat(xmlLeaf, "latency", &leaf->leaf.latency);
  leaf->parent = spine;
  leaf->children = xmlLeaf->nSubs;
  for (int s=0; s<xmlLeaf->nSubs; s++) {
    struct ncclClusterXmlNode* node = xmlLeaf->subs[s];
    if (strcmp(node->name, "host") == 0) NCCLCHECK(ncclTopoAddHost(node, cluster, leaf));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddSpine(struct ncclClusterXmlNode* xmlSpine, struct ncclTopoCluster* cluster, struct ncclTopoClusterNode *superSpine) {
  struct ncclTopoClusterNode* spine;
  NCCLCHECK(ncclTopoCreateClusterNode(cluster, &spine, CLUSTER_SPINE));
  const char* str;
  NCCLCHECK(clusterXmlGetAttr(xmlSpine, "id", &str));
  strcpy(spine->spine.name, str);
  clusterXmlGetAttrFloat(xmlSpine, "speed", &spine->spine.bw);
  clusterXmlGetAttrFloat(xmlSpine, "latency", &spine->spine.latency);
  spine->parent = superSpine;
  spine->children = xmlSpine->nSubs;
  for (int s=0; s<xmlSpine->nSubs; s++) {
    struct ncclClusterXmlNode* node = xmlSpine->subs[s];
    if (strcmp(node->name, "leaf") == 0) NCCLCHECK(ncclTopoAddLeaf(node, cluster, spine));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddSuperSpine(struct ncclClusterXmlNode* xmlSuperSpine, struct ncclTopoCluster* cluster) {
  struct ncclTopoClusterNode* superSpine;
  NCCLCHECK(ncclTopoCreateClusterNode(cluster, &superSpine, CLUSTER_SUPER_SPINE));
  const char* str;
  NCCLCHECK(clusterXmlGetAttr(xmlSuperSpine, "id", &str));
  strcpy(superSpine->superSpine.name, str);
  clusterXmlGetAttrFloat(xmlSuperSpine, "speed", &superSpine->superSpine.bw);
  clusterXmlGetAttrFloat(xmlSuperSpine, "latency", &superSpine->superSpine.latency);
  superSpine->parent = NULL;
  superSpine->children = xmlSuperSpine->nSubs;
  for (int s=0; s<xmlSuperSpine->nSubs; s++) {
    struct ncclClusterXmlNode* node = xmlSuperSpine->subs[s];
    if (strcmp(node->name, "spine") == 0) NCCLCHECK(ncclTopoAddSpine(node, cluster, superSpine));
  }
  // Note: superSpines may not be neccessarily switches. i.e. topology with spine switch mash
  return ncclSuccess;
}

ncclResult_t ncclTopoGetClusterFromXml(struct ncclClusterXml* xml, struct ncclTopoCluster** topoCluster) {
  NCCLCHECK(ncclCalloc(topoCluster, 1));
  struct ncclClusterXmlNode* topNode;
  NCCLCHECK(clusterXmlFindTag(xml, "cluster", &topNode));
  for (int s=0; s<topNode->nSubs; s++) {
    struct ncclClusterXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "super_spine") == 0) NCCLCHECK(ncclTopoAddSuperSpine(node, *topoCluster));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetCluster(struct ncclComm* comm, struct ncclTopoCluster** cluster) {
  ncclResult_t res = ncclSuccess;
  struct ncclClusterXml* xml = NULL;
  if (!NCCL_TOPO_CLUSTER_FILE.empty()) {
    bool found = false;
    NCCLCHECK(ncclCalloc(&xml, 1));
    INFO(NCCL_ENV, "NCCL_TOPO_CLUSTER_FILE set by environment to %s", NCCL_TOPO_CLUSTER_FILE.c_str());
    NCCLCHECKGOTO(
        ncclTopoClusterGetXmlFromFile(
            NCCL_TOPO_CLUSTER_FILE.data(),
            xml,
            (NCCL_TOPO_CLUSTER_FILE != NCCL_TOPO_CLUSTER_FILE_DEFAULT),
            &found),
        res,
        cleanup);
    if (found) {
      NCCLCHECKGOTO(ncclTopoGetClusterFromXml(xml, cluster), res, cleanup);
    }
  }
  // Add default code as else once it's mainstream.

 cleanup:
  free(xml);
  return res;
}
