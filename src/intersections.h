#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

// All intersection routines are defined inline in this header so that nvcc can
// inline them into the kernels. With separable compilation, calls into another
// translation unit cannot be inlined and pass large structs (Geom is ~230
// bytes) through local memory.

// Smallest ray parameter accepted as a hit. Spawned rays are already offset
// from their surface, so this only guards against numerical self-hits.
#define T_MIN 1e-5f

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// Transforms a world-space ray into the object space of `geom`. The direction
// is deliberately NOT normalized: the ray parameter t is then identical in
// world and object space, so hits from different objects compare directly.
__host__ __device__ inline Ray toObjectSpace(const Geom& geom, const Ray& r)
{
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f));
    return q;
}

// CHECKITOUT
/**
 * Test intersection between an object-space ray and the unit cube, which
 * ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ inline float boxIntersectionTest(const Ray& q)
{
    glm::vec3 invDir = 1.0f / q.direction;
    glm::vec3 t0 = (glm::vec3(-0.5f) - q.origin) * invDir;
    glm::vec3 t1 = (glm::vec3(0.5f) - q.origin) * invDir;
    glm::vec3 tNear = glm::min(t0, t1);
    glm::vec3 tFar = glm::max(t0, t1);
    float tmin = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
    float tmax = fminf(fminf(tFar.x, tFar.y), tFar.z);
    if (tmax < tmin || tmax <= T_MIN)
    {
        return -1.0f;
    }
    // Starting inside the box: the exit point is the hit.
    return tmin > T_MIN ? tmin : tmax;
}

// CHECKITOUT
/**
 * Test intersection between an object-space ray and the sphere of radius 0.5
 * centered at the origin. The ray direction does not need to be normalized.
 *
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ inline float sphereIntersectionTest(const Ray& q)
{
    const float radius = 0.5f;
    float a = glm::dot(q.direction, q.direction);
    float b = glm::dot(q.origin, q.direction);
    float c = glm::dot(q.origin, q.origin) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant < 0.0f)
    {
        return -1.0f;
    }
    float root = sqrtf(discriminant);
    float t = (-b - root) / a;
    if (t <= T_MIN)
    {
        t = (-b + root) / a;     // the ray starts inside the sphere
    }
    return t > T_MIN ? t : -1.0f;
}

__host__ __device__ inline float geomIntersectionTest(const Geom& geom, const Ray& r)
{
    Ray q = toObjectSpace(geom, r);
    if (geom.type == CUBE)
    {
        return boxIntersectionTest(q);
    }
    return sphereIntersectionTest(q);
}

// Closest hit over all scene geometry.
__host__ __device__ inline void intersectScene(
    const Geom* geoms, int geomCount, const Ray& ray, ShadeableIntersection& isect)
{
    float tMin = FLT_MAX;
    int hitGeom = -1;
    for (int i = 0; i < geomCount; i++)
    {
        // TODO: add more intersection tests here... triangle? metaball? CSG?
        float t = geomIntersectionTest(geoms[i], ray);
        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t < tMin)
        {
            tMin = t;
            hitGeom = i;
        }
    }

    if (hitGeom < 0)
    {
        isect.t = -1.0f;
        isect.materialId = -1;
        isect.geomId = -1;
        return;
    }
    isect.t = tMin;
    isect.materialId = geoms[hitGeom].materialid;
    isect.geomId = hitGeom;
}

// Rebuilds the full surface description of a hit for shading.
__host__ __device__ inline SurfaceHit computeSurfaceHit(
    const Geom* geoms, const Ray& ray, const ShadeableIntersection& isect)
{
    SurfaceHit hit;
    const Geom& geom = geoms[isect.geomId];
    hit.position = ray.origin + isect.t * ray.direction;

    Ray q = toObjectSpace(geom, ray);
    glm::vec3 p = q.origin + isect.t * q.direction;
    glm::vec3 objectNormal;
    if (geom.type == CUBE)
    {
        glm::vec3 a = glm::abs(p);
        if (a.x >= a.y && a.x >= a.z) objectNormal = glm::vec3(p.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
        else if (a.y >= a.z)          objectNormal = glm::vec3(0.0f, p.y > 0.0f ? 1.0f : -1.0f, 0.0f);
        else                          objectNormal = glm::vec3(0.0f, 0.0f, p.z > 0.0f ? 1.0f : -1.0f);
    }
    else
    {
        objectNormal = p;
    }

    glm::vec3 n = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(objectNormal, 0.0f)));
    hit.frontFace = glm::dot(n, ray.direction) < 0.0f;
    hit.normal = hit.frontFace ? n : -n;
    return hit;
}
