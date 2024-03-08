// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CLUSTER_XML_H_
#define CLUSTER_XML_H_

#include "nccl.h"
#include "xml.h"
#include "debug.h"
#include "checks.h"
#include <stdlib.h>

// A few constraints to make the implementation easy
#define MAX_CLUSTER_SUBS 1024
#define MAX_CLUSTER_NODES 65536

struct ncclClusterXmlNode {
  char name[MAX_STR_LEN+1];
  struct {
    char key[MAX_STR_LEN+1];
    char value[MAX_STR_LEN+1];
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
  int nAttrs;
  int type;
  struct ncclClusterXmlNode* parent;
  struct ncclClusterXmlNode* subs[MAX_CLUSTER_SUBS];
  int nSubs;
};

struct ncclClusterXml {
  struct ncclClusterXmlNode nodes[MAX_CLUSTER_NODES];
  int maxIndex;
};

/* File functions */
#define NCCL_TOPO_CLUSTER_XML_VERSION 1
ncclResult_t ncclTopoClusterGetXmlFromFile(const char* xmlTopoFile, struct ncclClusterXml* xml, int warn, bool *found);
ncclResult_t ncclTopoDumpClusterXmlToFile(const char* xmlTopoFile, struct ncclClusterXml* xml);

/**************/
/* XML Struct */
/* Functions  */
/**************/

static ncclResult_t clusterXmlGetAttrIndex(struct ncclClusterXmlNode* node, const char* attrName, int* index) {
  *index = -1;
  const int nAttrs = node->nAttrs;
  for (int a=0; a<nAttrs; a++) {
    if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN) == 0) {
      *index = a;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t clusterXmlGetAttr(struct ncclClusterXmlNode* node, const char* attrName, const char** value) {
  int index;
  NCCLCHECK(clusterXmlGetAttrIndex(node, attrName, &index));
  *value = index == -1 ? NULL : node->attrs[index].value;
  return ncclSuccess;
}

static ncclResult_t clusterXmlGetAttrStr(struct ncclClusterXmlNode* node, const char* attrName, const char** value) {
  NCCLCHECK(clusterXmlGetAttr(node, attrName, value));
  if (*value == NULL) {
    WARN("Attribute %s of node %s not found", attrName, node->name);
    return ncclInternalError;
  }
  return ncclSuccess;
}
static ncclResult_t clusterXmlGetAttrInt(struct ncclClusterXmlNode* node, const char* attrName, int* value) {
  const char* str;
  NCCLCHECK(clusterXmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t clusterXmlGetAttrIntDefault(struct ncclClusterXmlNode* node, const char* attrName, int* value, int defaultValue) {
  const char* str;
  NCCLCHECK(clusterXmlGetAttr(node, attrName, &str));
  *value = str ? strtol(str, NULL, 0) : defaultValue;
  return ncclSuccess;
}


static ncclResult_t clusterXmlGetAttrFloat(struct ncclClusterXmlNode* node, const char* attrName, float* value) {
  const char* str;
  NCCLCHECK(clusterXmlGetAttrStr(node, attrName, &str));
  *value = strtof(str, NULL);
  return ncclSuccess;
}

static ncclResult_t clusterXmlFindTag(struct ncclClusterXml* xml, const char* tagName, struct ncclClusterXmlNode** node) {
  *node = NULL;
  for (int i=0; i<xml->maxIndex; i++) {
    struct ncclClusterXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t clusterXmlFindTagKv(struct ncclClusterXml* xml, const char* tagName, struct ncclClusterXmlNode** node, const char* attrName, const char* attrValue) {
  *node = NULL;
  for (int i=0; i<xml->maxIndex; i++) {
    struct ncclClusterXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      const char* value;
      NCCLCHECK(clusterXmlGetAttr(n, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *node = n;
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t clusterXmlSetAttr(struct ncclClusterXmlNode* node, const char* attrName, const char* value) {
  int index;
  NCCLCHECK(clusterXmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t clusterXmlSetAttrIfUnset(struct ncclClusterXmlNode* node, const char* attrName, const char* value) {
  int index;
  NCCLCHECK(clusterXmlGetAttrIndex(node, attrName, &index));
  if (index != -1) return ncclSuccess;
  index = node->nAttrs++;
  strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  node->attrs[index].key[MAX_STR_LEN] = '\0';
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t clusterXmlSetAttrInt(struct ncclClusterXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(clusterXmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t clusterXmlSetAttrFloat(struct ncclClusterXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(clusterXmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%g", value);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t clusterXmlUnsetAttr(struct ncclClusterXmlNode* node, const char* attrName) {
  int index;
  NCCLCHECK(clusterXmlGetAttrIndex(node, attrName, &index));
  if (index == -1) return ncclSuccess;
  for (int i=index+1; i<node->nAttrs; i++) {
    strcpy(node->attrs[i-1].key, node->attrs[i].key);
    strcpy(node->attrs[i-1].value, node->attrs[i].value);
  }
  node->nAttrs--;
  return ncclSuccess;
}

static ncclResult_t clusterXmlGetSub(struct ncclClusterXmlNode* node, const char* subName, struct ncclClusterXmlNode** sub) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    if (strcmp(node->subs[s]->name, subName) == 0) {
      *sub = node->subs[s];
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t clusterXmlGetSubKv(struct ncclClusterXmlNode* node, const char* subName, struct ncclClusterXmlNode** sub, const char* attrName, const char* attrValue) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    struct ncclClusterXmlNode* subNode = node->subs[s];
    if (strcmp(subNode->name, subName) == 0) {
      const char* value;
      NCCLCHECK(clusterXmlGetAttr(subNode, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *sub = node->subs[s];
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}
static ncclResult_t clusterXmlGetSubKvInt(struct ncclClusterXmlNode* node, const char* subName, struct ncclClusterXmlNode** sub, const char* attrName, const int attrValue) {
  char strValue[10];
  snprintf(strValue, 10, "%d", attrValue);
  NCCLCHECK(clusterXmlGetSubKv(node, subName, sub, attrName, strValue));
  return ncclSuccess;
}

static ncclResult_t clusterXmlAddNode(struct ncclClusterXml* xml, struct ncclClusterXmlNode* parent, const char* subName, struct ncclClusterXmlNode** sub) {
  if (xml->maxIndex == MAX_CLUSTER_NODES) {
    WARN("Error : too many XML nodes (max %d)", MAX_CLUSTER_NODES);
    return ncclInternalError;
  }
  struct ncclClusterXmlNode* s = xml->nodes+xml->maxIndex++;
  s->nSubs = 0;
  s->nAttrs = 0;
  *sub = s;
  s->parent = parent;
  if (parent) parent->subs[parent->nSubs++] = s;
  strncpy(s->name, subName, MAX_STR_LEN);
  s->name[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t clusterXmlRemoveNode(struct ncclClusterXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  struct ncclClusterXmlNode* parent = node->parent;
  if (parent == NULL) return ncclSuccess;
  int shift = 0;
  for (int s=0; s<parent->nSubs; s++) {
    if (parent->subs[s] == node) shift = 1;
    else if (shift) parent->subs[s-1] = parent->subs[s];
  }
  parent->nSubs--;
  return ncclSuccess;
}

#endif
