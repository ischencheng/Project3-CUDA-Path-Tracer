#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <cfloat>

#include <string>
#include <vector>

enum GeomType
{
    SPHERE,
    CUBE,
    MESH        // instance of a triangle mesh with its own BVH
};

struct AABB
{
    glm::vec3 min;
    glm::vec3 max;

    __host__ __device__ static AABB empty()
    {
        AABB b;
        b.min = glm::vec3(FLT_MAX);
        b.max = glm::vec3(-FLT_MAX);
        return b;
    }

    __host__ __device__ void grow(glm::vec3 p)
    {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    __host__ __device__ void grow(const AABB& b)
    {
        min = glm::min(min, b.min);
        max = glm::max(max, b.max);
    }

    __host__ __device__ float surfaceArea() const
    {
        glm::vec3 e = max - min;
        return e.x < 0.0f ? 0.0f : 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;         // for meshes: overrides the per-triangle materials if >= 0
    int meshId;             // MESH only
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    AABB worldBounds;
};

// A triangle mesh asset. Triangles and BVH nodes of all meshes live in shared
// global arrays; the indices below are absolute.
struct MeshInfo
{
    int triOffset;
    int triCount;
    int rootNode;
    int nodeCount;
    AABB bounds;            // object space
};

// 32-byte BVH node. Leaves (triCount > 0) reference the triangle range
// [leftFirst, leftFirst + triCount); inner nodes store their two children at
// leftFirst and leftFirst + 1.
struct alignas(16) BVHNode
{
    glm::vec3 aabbMin;
    int leftFirst;
    glm::vec3 aabbMax;
    int triCount;
};

// Triangle positions in the layout used for intersection: a vertex and two
// edges as float4 so each is a single 128-bit load.
struct TriangleGeom
{
    float4 v0;
    float4 e1;
    float4 e2;
};

// Vertex indices into the global attribute arrays.
struct Triangle
{
    int v[3];
    int materialId;
};

enum MaterialType
{
    MATERIAL_DIFFUSE = 0,   // Lambertian
    MATERIAL_SPECULAR,      // conductor: perfect mirror, or GGX when rough
    MATERIAL_DIELECTRIC,    // glass/water: Fresnel reflection + refraction, smooth or rough
    MATERIAL_PBR,           // metallic-roughness (glTF): diffuse + GGX specular
    MATERIAL_EMITTING,      // pure emitter
    MATERIAL_TYPE_COUNT
};

struct Material
{
    int type;
    glm::vec3 color;        // albedo / specular F0 / transmission tint / base color
    glm::vec3 emission;     // emitted radiance (color * emittance)
    float roughness;        // perceptual roughness, GGX alpha = roughness^2
    float metallic;
    float ior;
    glm::vec3 absorption;   // Beer-Lambert absorption coefficient inside dielectrics
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
    float lensRadius;       // thin lens aperture radius, 0 for a pinhole camera
    float focalDistance;    // distance to the plane in focus along the view direction
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
    glm::vec3 backgroundColor;
};

enum PathFlags
{
    PATH_FLAG_DELTA_BOUNCE = 1,     // the last scattering event was a delta lobe
};

struct PathSegment
{
    Ray ray;
    glm::vec3 throughput;   // product of BSDF * cos / pdf along the path so far
    int pixelIndex;
    int remainingBounces;   // <= 0 means the path is terminated
    int mediumMaterial;     // material whose interior the ray travels through, -1 outside
    float lastPdf;          // solid angle pdf of the BSDF sample that spawned the ray
    int flags;              // PathFlags
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
    int primId;         // triangle index for meshes, -1 for analytic shapes
    glm::vec2 bary;     // barycentrics (b1, b2) of the triangle hit
};

// Surface attributes reconstructed for shading.
struct SurfaceHit
{
    glm::vec3 position;
    glm::vec3 normal;           // geometric normal, facing the incoming ray
    glm::vec3 shadingNormal;    // interpolated normal, same side as `normal`
    glm::vec3 tangent;          // world-space tangent (for normal mapping), may be zero
    float tangentSign;          // bitangent handedness
    glm::vec2 uv;
    bool frontFace;             // true if the ray hit the outside of the surface
};

// Device-side view of the scene, passed to kernels by value.
struct SceneView
{
    const Geom* geoms;
    int geomCount;
    const Material* materials;
    const MeshInfo* meshes;
    const BVHNode* bvhNodes;
    const TriangleGeom* triGeoms;
    const Triangle* triangles;
    const glm::vec3* normals;
    const glm::vec2* uvs;
    const glm::vec4* tangents;
    int useBVH;         // 0: test every triangle of a mesh
    int cullBounds;     // test each object's world AABB before its geometry
};
