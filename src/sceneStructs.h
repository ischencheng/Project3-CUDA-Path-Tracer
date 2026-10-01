#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
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
};

enum MaterialType
{
    MATERIAL_DIFFUSE = 0,
    MATERIAL_SPECULAR,
    MATERIAL_EMITTING,
    MATERIAL_TYPE_COUNT
};

struct Material
{
    int type;
    glm::vec3 color;
    float emittance;
    float roughness;
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
    glm::vec3 throughput;   // product of BSDF * cos / pdf along the path so far
    int pixelIndex;
    int remainingBounces;   // <= 0 means the path is terminated
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
// Only the data needed to identify the hit is stored here; the shading kernel
// reconstructs the surface attributes, which keeps this struct cheap to sort.
struct ShadeableIntersection
{
    float t;            // world-space distance along the (unit) ray, < 0 for a miss
    int materialId;
    int geomId;
};

// Surface attributes reconstructed for shading.
struct SurfaceHit
{
    glm::vec3 position;
    glm::vec3 normal;       // geometric normal, facing the incoming ray
    bool frontFace;         // true if the ray hit the outside of the surface
};
