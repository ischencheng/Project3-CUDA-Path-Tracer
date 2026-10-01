#pragma once

#include "bvh.h"
#include "sceneStructs.h"

#include <string>
#include <unordered_map>
#include <vector>

struct MeshData;

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName, const BVHBuildSettings* bvhOverride);
    int loadMesh(const std::string& path, int defaultMaterial);
    void computeWorldBounds(Geom& geom) const;
public:
    Scene(std::string filename, const BVHBuildSettings* bvhOverride = nullptr);

    // Changes the output resolution, keeping the vertical field of view.
    void setResolution(int width, int height);

    int addMaterial(const Material& m);
    int addTexture(TextureData&& texture);
    // Loads an image file (relative to the scene file) once; -1 on failure.
    int loadTextureFile(const std::string& file);
    // Builds the BVH of `mesh`, appends it to the global geometry arrays and
    // returns its mesh id.
    int addMesh(const MeshData& mesh);

    std::string sceneFile;
    std::string sceneDir;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // Triangle meshes: shared global arrays, see MeshInfo.
    BVHBuildSettings bvhSettings;
    std::vector<MeshInfo> meshes;
    std::vector<BVHNode> bvhNodes;
    std::vector<TriangleGeom> triGeoms;     // BVH order
    std::vector<Triangle> triangles;        // BVH order
    std::vector<glm::vec3> vertexNormals;
    std::vector<glm::vec2> vertexUVs;
    std::vector<glm::vec4> vertexTangents;
    std::unordered_map<std::string, int> meshCache;    // file path -> mesh id
    std::vector<TextureData> textures;
    std::unordered_map<std::string, int> textureCache; // file path -> texture id
    double totalBvhBuildMs = 0.0;
};
