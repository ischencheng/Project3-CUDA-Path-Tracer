#include "bvh.h"

#include <algorithm>
#include <chrono>

namespace
{

struct Bin
{
    AABB bounds = AABB::empty();
    int count = 0;
};

class Builder
{
public:
    Builder(const std::vector<AABB>& primBounds, const std::vector<glm::vec3>& centroids,
        const BVHBuildSettings& settings, std::vector<BVHNode>& nodes, std::vector<int>& order)
        : primBounds(primBounds), centroids(centroids), settings(settings), nodes(nodes), order(order)
    {
        bins.resize(settings.numBins);
        leftArea.resize(settings.numBins);
        leftCount.resize(settings.numBins);
    }

    void build(BVHBuildStats& stats)
    {
        const int n = (int)primBounds.size();
        order.resize(n);
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
        }
        nodes.clear();
        nodes.reserve(std::max(1, 2 * n / std::max(1, settings.maxLeafSize)));
        BVHNode root;
        root.leftFirst = 0;
        root.triCount = n;
        nodes.push_back(root);
        updateBounds(0);
        subdivide(0, 0, stats);
        stats.nodeCount = (int)nodes.size();
    }

private:
    const std::vector<AABB>& primBounds;
    const std::vector<glm::vec3>& centroids;
    const BVHBuildSettings& settings;
    std::vector<BVHNode>& nodes;
    std::vector<int>& order;
    std::vector<Bin> bins;
    std::vector<float> leftArea;
    std::vector<int> leftCount;

    void updateBounds(int nodeIdx)
    {
        BVHNode& node = nodes[nodeIdx];
        AABB b = AABB::empty();
        for (int i = 0; i < node.triCount; i++)
        {
            b.grow(primBounds[order[node.leftFirst + i]]);
        }
        node.aabbMin = b.min;
        node.aabbMax = b.max;
    }

    int binIndex(float c, float minC, float scale) const
    {
        return std::min(settings.numBins - 1, std::max(0, (int)((c - minC) * scale)));
    }

    // Evaluates the SAH cost of every bin boundary along every axis.
    // Returns the best cost (FLT_MAX if the centroids cannot be separated).
    float findBestSplit(int first, int count, int& bestAxis, int& bestBin, AABB& centroidBounds)
    {
        centroidBounds = AABB::empty();
        for (int i = 0; i < count; i++)
        {
            centroidBounds.grow(centroids[order[first + i]]);
        }

        float bestCost = FLT_MAX;
        const int numBins = settings.numBins;
        for (int axis = 0; axis < 3; axis++)
        {
            float minC = centroidBounds.min[axis];
            float extent = centroidBounds.max[axis] - minC;
            if (extent <= 0.0f)
            {
                continue;
            }
            for (Bin& b : bins)
            {
                b = Bin();
            }
            float scale = numBins / extent;
            for (int i = 0; i < count; i++)
            {
                int prim = order[first + i];
                Bin& b = bins[binIndex(centroids[prim][axis], minC, scale)];
                b.count++;
                b.bounds.grow(primBounds[prim]);
            }

            // Sweep from the left, then from the right, evaluating
            // cost = N_left * A_left + N_right * A_right at each plane.
            AABB acc = AABB::empty();
            int sum = 0;
            for (int i = 0; i < numBins - 1; i++)
            {
                sum += bins[i].count;
                acc.grow(bins[i].bounds);
                leftCount[i] = sum;
                leftArea[i] = acc.surfaceArea();
            }
            acc = AABB::empty();
            sum = 0;
            for (int i = numBins - 1; i > 0; i--)
            {
                sum += bins[i].count;
                acc.grow(bins[i].bounds);
                float cost = leftCount[i - 1] * leftArea[i - 1] + sum * acc.surfaceArea();
                if (leftCount[i - 1] > 0 && sum > 0 && cost < bestCost)
                {
                    bestCost = cost;
                    bestAxis = axis;
                    bestBin = i;    // primitives in bins [0, i) go left
                }
            }
        }
        return bestCost;
    }

    void subdivide(int nodeIdx, int depth, BVHBuildStats& stats)
    {
        stats.depth = std::max(stats.depth, depth);
        const int first = nodes[nodeIdx].leftFirst;
        const int count = nodes[nodeIdx].triCount;

        auto makeLeaf = [&]() {
            stats.leafCount++;
            stats.maxLeafTriangles = std::max(stats.maxLeafTriangles, count);
        };

        if (count <= settings.maxLeafSize || depth >= settings.maxDepth)
        {
            makeLeaf();
            return;
        }

        int axis = 0;
        int splitBin = 0;
        AABB centroidBounds;
        float splitCost = findBestSplit(first, count, axis, splitBin, centroidBounds);
        if (splitCost == FLT_MAX)
        {
            makeLeaf();     // all centroids coincide
            return;
        }
        // SAH termination: splitting a small node is not worth an extra
        // traversal step if it does not reduce the expected intersection cost.
        AABB nodeBounds;
        nodeBounds.min = nodes[nodeIdx].aabbMin;
        nodeBounds.max = nodes[nodeIdx].aabbMax;
        float leafCost = count * nodeBounds.surfaceArea();
        if (splitCost >= leafCost && count <= 4 * settings.maxLeafSize)
        {
            makeLeaf();
            return;
        }

        // Partition the primitive range in place by bin.
        float minC = centroidBounds.min[axis];
        float scale = settings.numBins / (centroidBounds.max[axis] - minC);
        int i = first;
        int j = first + count - 1;
        while (i <= j)
        {
            if (binIndex(centroids[order[i]][axis], minC, scale) < splitBin)
            {
                i++;
            }
            else
            {
                std::swap(order[i], order[j--]);
            }
        }
        int leftCount = i - first;
        if (leftCount == 0 || leftCount == count)
        {
            makeLeaf();
            return;
        }

        int leftChild = (int)nodes.size();
        nodes.emplace_back();
        nodes.emplace_back();
        nodes[leftChild].leftFirst = first;
        nodes[leftChild].triCount = leftCount;
        nodes[leftChild + 1].leftFirst = i;
        nodes[leftChild + 1].triCount = count - leftCount;
        nodes[nodeIdx].leftFirst = leftChild;
        nodes[nodeIdx].triCount = 0;
        updateBounds(leftChild);
        updateBounds(leftChild + 1);
        subdivide(leftChild, depth + 1, stats);
        subdivide(leftChild + 1, depth + 1, stats);
    }
};

} // namespace

BVHBuildStats buildBVH(const std::vector<AABB>& primBounds,
    const std::vector<glm::vec3>& centroids,
    const BVHBuildSettings& settings,
    std::vector<BVHNode>& nodes,
    std::vector<int>& order)
{
    auto start = std::chrono::high_resolution_clock::now();
    BVHBuildSettings s = settings;
    s.maxDepth = std::min(s.maxDepth, BVH_STACK_SIZE - 2);
    s.maxLeafSize = std::max(1, s.maxLeafSize);
    s.numBins = std::max(2, s.numBins);

    BVHBuildStats stats;
    Builder builder(primBounds, centroids, s, nodes, order);
    builder.build(stats);
    stats.buildMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
    return stats;
}
