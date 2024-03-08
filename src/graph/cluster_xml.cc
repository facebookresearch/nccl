// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include "core.h"
#include "nvmlwrap.h"
#include "cluster_xml.h"

/*******************/
/* XML File Parser */
/*******************/

extern ncclResult_t xmlGetChar(FILE* file, char* c);
extern ncclResult_t xmlGetValue(FILE* file, char* value, char* last);
extern ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last);
extern ncclResult_t xmlSkipComment(FILE* file, char* start, char next);

ncclResult_t clusterXmlGetNode(FILE* file, struct ncclClusterXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(xmlSkipComment(file, node->name+3, c));
    return clusterXmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclClusterXml*, struct ncclClusterXmlNode*);

struct xmlHandler {
  const char * name;
  xmlHandlerFunc_t func;
};

ncclResult_t clusterXmlLoadSub(FILE* file, struct ncclClusterXml* xml, struct ncclClusterXmlNode* head, struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == MAX_CLUSTER_NODES) {
      WARN("Error : XML parser is limited to 1024 nodes");
      return ncclInternalError;
    }
    struct ncclClusterXmlNode* node = xml->nodes+xml->maxIndex;
    memset(node, 0, sizeof(struct ncclClusterXmlNode));
    NCCLCHECK(clusterXmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) head->subs[head->nSubs++] = node;
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(clusterXmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNetworkNic(FILE* file, struct ncclClusterXml* xml, struct ncclClusterXmlNode* head) {
  NCCLCHECK(clusterXmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadHost(FILE* file, struct ncclClusterXml* xml, struct ncclClusterXmlNode* head) {
  struct xmlHandler handlers[] = { { "nic", ncclTopoXmlLoadNetworkNic } };
  NCCLCHECK(clusterXmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadLeaf(FILE* file, struct ncclClusterXml* xml, struct ncclClusterXmlNode* head) {
  struct xmlHandler handlers[] = { { "host", ncclTopoXmlLoadHost } };
  NCCLCHECK(clusterXmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSpine(FILE* file, struct ncclClusterXml* xml, struct ncclClusterXmlNode* head) {
  struct xmlHandler handlers[] = { { "leaf", ncclTopoXmlLoadLeaf } };
  NCCLCHECK(clusterXmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSuperSpine(FILE* file, struct ncclClusterXml* xml, struct ncclClusterXmlNode* head) {
  struct xmlHandler handlers[] = { { "spine", ncclTopoXmlLoadSpine } };
  NCCLCHECK(clusterXmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCluster(FILE* file, struct ncclClusterXml* xml, struct ncclClusterXmlNode* head) {
  int version;
  NCCLCHECK(clusterXmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_CLUSTER_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version, NCCL_TOPO_CLUSTER_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* topologyType;
  NCCLCHECK(clusterXmlGetAttr(head, "type", &topologyType));
  if (topologyType != NULL) INFO(NCCL_GRAPH, "Loading topology %s", topologyType);
  else INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = { { "super_spine", ncclTopoXmlLoadSuperSpine } };
  NCCLCHECK(clusterXmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoClusterGetXmlFromFile(const char* xmlTopoFile, struct ncclClusterXml* xml, int warn, bool *found) {
  FILE* file = fopen(xmlTopoFile, "r");
  *found = false;
  if (file == NULL) {
    if (warn) {
      WARN("Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    }
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = { { "cluster", ncclTopoXmlLoadCluster } };
  xml->maxIndex = 0;
  NCCLCHECK(clusterXmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  *found = true;
  return ncclSuccess;
}
