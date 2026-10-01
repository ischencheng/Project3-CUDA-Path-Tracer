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
    float time;             // shutter time in [0, 1) for motion blur
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
    AABB worldBounds;       // covers the whole motion
    int lightIndex;         // index into the light list for emissive spheres/cubes, else -1
    glm::vec3 motion;       // world space displacement over the shutter interval
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

enum ProceduralType
{
    PROCEDURAL_NONE = 0,
    PROCEDURAL_CHECKER,     // UV checkerboard
    PROCEDURAL_MARBLE,      // turbulence-distorted veins (object space)
    PROCEDURAL_WOOD,        // noisy rings around the object's y axis
    PROCEDURAL_NOISE,       // fractal gradient noise
    PROCEDURAL_TYPE_COUNT
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

    // Textures (indices into the scene texture list, -1 if unused). Base
    // color and emissive textures are sRGB encoded; the others are linear.
    int baseColorTex;
    int metallicRoughnessTex;   // glTF: roughness in G, metalness in B
    int normalTex;              // tangent space normal map
    int emissiveTex;
    float normalScale;
    glm::vec2 uvScale;          // tiling of the image textures

    // Procedural texture blending `color` towards procColor2.
    int procedural;             // ProceduralType
    glm::vec3 procColor2;
    float procScale;
    // Procedural bump mapping (0 strength disables it).
    float bumpStrength;
    float bumpScale;
};

// Material with every optional feature disabled.
inline Material makeDefaultMaterial()
{
    Material m{};
    m.type = MATERIAL_DIFFUSE;
    m.color = glm::vec3(0.8f);
    m.ior = 1.5f;
    m.baseColorTex = -1;
    m.metallicRoughnessTex = -1;
    m.normalTex = -1;
    m.emissiveTex = -1;
    m.normalScale = 1.0f;
    m.uvScale = glm::vec2(1.0f);
    m.procedural = PROCEDURAL_NONE;
    m.procScale = 1.0f;
    m.bumpScale = 1.0f;
    return m;
}

// CPU copy of an 8-bit RGBA texture.
struct TextureData
{
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;
    std::string name;
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

// CPU side environment map.
struct EnvironmentMap
{
    int width = 0;
    int height = 0;
    std::vector<float> rgba;        // linear radiance, 4 floats per texel
    std::vector<float> func;
    std::vector<float> marginalCdf;
    std::vector<float> conditionalCdf;
    float integral = 0.0f;
    float intensity = 1.0f;
    float rotation = 0.0f;          // radians
};

enum PathFlags
{
    PATH_FLAG_DELTA_BOUNCE = 1,     // the last scattering event was a delta lobe
    PATH_FLAG_AOV_DONE = 2,         // denoiser features were recorded for this path
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
    glm::vec3 objectPosition;   // hit point in object space (solid textures)
    glm::vec3 normal;           // geometric normal, facing the incoming ray
    glm::vec3 shadingNormal;    // interpolated normal, same side as `normal`
    glm::vec3 tangent;          // world-space tangent (for normal mapping), may be zero
    float tangentSign;          // bitangent handedness
    glm::vec2 uv;
    bool frontFace;             // true if the ray hit the outside of the surface
};

enum LightType
{
    LIGHT_TRIANGLE = 0,
    LIGHT_SPHERE,
    LIGHT_CUBE
};

// An emitting surface that can be sampled for next event estimation.
struct Light
{
    int type;
    int geomId;
    int primId;             // triangle index for LIGHT_TRIANGLE
    float area;             // world space surface area
    float selectPdf;        // probability of picking this light (env choice included)
    glm::vec3 v0;           // world space triangle for LIGHT_TRIANGLE
    glm::vec3 e1;
    glm::vec3 e2;
};

struct LightsView
{
    const Light* lights;
    const float* cdf;       // count + 1 entries, power-proportional
    int count;
    float envSelectProb;    // probability of sampling the environment instead
};

// Environment: either a constant color or an equirectangular HDR map with a
// piecewise-constant importance sampling distribution.
struct EnvironmentView
{
    int hasMap;
    int enabled;            // the environment emits anything at all
    glm::vec3 color;        // constant radiance when there is no map
    float intensity;
    float rotation;         // radians around +y
    cudaTextureObject_t texture;
    int width;
    int height;
    const float* func;          // width * height sampling weights (luminance * sin theta)
    const float* marginalCdf;   // height + 1
    const float* conditionalCdf;// height * (width + 1)
    float integral;             // mean of func over the unit square
};

// Shadow ray queued by the shading kernel for next event estimation.
struct ShadowRay
{
    glm::vec3 origin;
    float maxT;
    glm::vec3 direction;
    int pixelIndex;         // -1: no shadow ray for this path
    glm::vec3 contribution; // added to the pixel if the ray is unoccluded
    float time;
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
    const cudaTextureObject_t* textures;
    const int* triLightIndex;   // light index per triangle (or NULL)
    const Light* lights;
    int useBVH;         // 0: test every triangle of a mesh
    int cullBounds;     // test each object's world AABB before its geometry
};
