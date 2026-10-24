#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    // meshes only: triangle range and world space bounding box
    int triStart;
    int triCount;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int bvhRoot;
};

// the two children of an interior node sit next to each other
struct BVHNode
{
    glm::vec3 bboxMin;
    // leaf: first triangle, interior: left child
    int start;
    glm::vec3 bboxMax;
    // 0 for interior nodes
    int count;
};

// already in world space
struct Triangle
{
    glm::vec3 v[3];
    glm::vec3 n[3];
    glm::vec2 uv[3];
    // w is the handedness of the bitangent
    glm::vec4 tangent;
    int materialid;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    // -1 means no texture
    int baseColorTexture = -1;
    int normalTexture = -1;
    // procedural checker in world space, 0 turns it off
    float checkerScale = 0.0f;
    glm::vec3 checkerColor;
};

// rgba8 texels live in one big buffer
struct Texture
{
    int width;
    int height;
    int offset;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool outside;
  // meshes only, the shader looks up uv and tangent when it needs them
  int triangle;
  glm::vec2 bary;
};
