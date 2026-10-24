#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& file, const glm::mat4& transform, int materialOverride);
    void buildNode(int node, int start, int count, int maxLeafSize, int depth);
public:
    Scene(std::string filename);
    void buildBVH(int maxLeafSize);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvh;
    std::vector<Texture> textures;
    std::vector<uchar4> texels;
    std::vector<Material> materials;
    RenderState state;
};
