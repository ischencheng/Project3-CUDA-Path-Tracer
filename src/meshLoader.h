#pragma once

#include "sceneStructs.h"

#include <string>
#include <vector>

class Scene;

// Triangle mesh in the object space of the scene object that loads it. All
// node transforms of the source file are already applied.
struct MeshData
{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;     // empty, or one per position
    std::vector<glm::vec2> uvs;         // empty, or one per position
    std::vector<glm::vec4> tangents;    // empty, or one per position (w = handedness)
    std::vector<glm::ivec3> indices;
    std::vector<int> materials;         // global material id per triangle
};

// Loaders append the materials (and textures) they need to the scene and
// return false on failure. `defaultMaterial` is used for primitives without
// a material.
bool loadGLTF(const std::string& path, Scene& scene, int defaultMaterial, MeshData& mesh);
bool loadOBJ(const std::string& path, Scene& scene, int defaultMaterial, MeshData& mesh);

// Computes per-vertex tangents from positions, normals and UVs (Lengyel).
void computeTangents(MeshData& mesh);
