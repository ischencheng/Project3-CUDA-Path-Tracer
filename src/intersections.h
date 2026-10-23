#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


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

// moller-trumbore, returns the distance and the barycentric (u, v)
__host__ __device__ inline float triangleIntersectionTest(const Triangle& tri, const Ray& r, glm::vec2& bary)
{
    glm::vec3 e1 = tri.v[1] - tri.v[0];
    glm::vec3 e2 = tri.v[2] - tri.v[0];
    glm::vec3 p = glm::cross(r.direction, e2);
    float det = glm::dot(e1, p);
    if (fabsf(det) < 1e-10f)
    {
        return -1.0f;
    }
    float invDet = 1.0f / det;
    glm::vec3 s = r.origin - tri.v[0];
    float u = glm::dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }
    glm::vec3 q = glm::cross(s, e1);
    float v = glm::dot(r.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return -1.0f;
    }
    bary = glm::vec2(u, v);
    return glm::dot(e2, q) * invDet;
}

// slab test, returns where the ray enters the box (0 if it starts inside)
__host__ __device__ inline float boundingBoxIntersectionTest(const glm::vec3& bmin, const glm::vec3& bmax, const Ray& r, const glm::vec3& invDir)
{
    glm::vec3 t0 = (bmin - r.origin) * invDir;
    glm::vec3 t1 = (bmax - r.origin) * invDir;
    glm::vec3 tNear = glm::min(t0, t1);
    glm::vec3 tFar = glm::max(t0, t1);
    float tEnter = fmaxf(fmaxf(tNear.x, tNear.y), fmaxf(tNear.z, 0.0f));
    float tExit = fminf(fminf(tFar.x, tFar.y), tFar.z);
    return tEnter <= tExit ? tEnter : -1.0f;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    const Geom& box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    const Geom& sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);
