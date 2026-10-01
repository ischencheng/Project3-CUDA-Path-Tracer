#pragma once

#include "sceneStructs.h"

#include <vector>

// Traversal keeps an explicit stack, so the tree depth must stay below it.
#define BVH_STACK_SIZE 64

struct BVHBuildSettings
{
    int maxLeafSize = 4;    // nodes with at most this many triangles become leaves
    int maxDepth = 48;      // hard limit, must stay below BVH_STACK_SIZE
    int numBins = 16;       // SAH bins per axis
};

struct BVHBuildStats
{
    int nodeCount = 0;
    int leafCount = 0;
    int depth = 0;
    int maxLeafTriangles = 0;
    double buildMs = 0.0;
};

// Builds a binned-SAH BVH over primitives given by their bounds and centroids.
// `order` receives the primitive permutation: leaf ranges index into it, so
// the caller must reorder its primitive arrays accordingly. Node indices in
// `nodes` start at 0 with the root.
BVHBuildStats buildBVH(const std::vector<AABB>& primBounds,
    const std::vector<glm::vec3>& centroids,
    const BVHBuildSettings& settings,
    std::vector<BVHNode>& nodes,
    std::vector<int>& order);
