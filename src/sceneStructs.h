#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

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
};

// Surface attributes reconstructed for shading.
struct SurfaceHit
{
    glm::vec3 position;
    glm::vec3 normal;       // geometric normal, facing the incoming ray
    bool frontFace;         // true if the ray hit the outside of the surface
};
