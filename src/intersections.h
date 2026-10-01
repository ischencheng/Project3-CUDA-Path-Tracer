#pragma once

#include "sceneStructs.h"
#include "bvh.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

// All intersection routines are defined inline in this header so that nvcc can
// inline them into the kernels. With separable compilation, calls into another
// translation unit cannot be inlined and pass large structs (Geom is ~250
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

__host__ __device__ inline glm::vec3 toVec3(float4 v)
{
    return glm::vec3(v.x, v.y, v.z);
}

// Transforms a world-space ray into the object space of `geom`. The direction
// is deliberately NOT normalized: the ray parameter t is then identical in
// world and object space, so hits from different objects compare directly.
// Moving objects are translated by motion * time, which is the same as
// moving the ray origin the opposite way.
__host__ __device__ inline Ray toObjectSpace(const Geom& geom, const Ray& r)
{
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin - geom.motion * r.time, 1.0f));
    q.direction = multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f));
    q.time = r.time;
    return q;
}

__host__ __device__ inline glm::vec3 safeInverse(glm::vec3 d)
{
    // Avoid 0 * inf = NaN in the slab test for axis-parallel rays.
    const float big = 1e30f;
    return glm::vec3(
        fabsf(d.x) > 1e-30f ? 1.0f / d.x : copysignf(big, d.x),
        fabsf(d.y) > 1e-30f ? 1.0f / d.y : copysignf(big, d.y),
        fabsf(d.z) > 1e-30f ? 1.0f / d.z : copysignf(big, d.z));
}

// Slab test. Returns the entry distance, or FLT_MAX if the box is missed or
// lies entirely beyond tMax.
__host__ __device__ inline float rayAABB(glm::vec3 origin, glm::vec3 invDir,
    glm::vec3 bmin, glm::vec3 bmax, float tMax)
{
    glm::vec3 t0 = (bmin - origin) * invDir;
    glm::vec3 t1 = (bmax - origin) * invDir;
    glm::vec3 tn = glm::min(t0, t1);
    glm::vec3 tf = glm::max(t0, t1);
    float tEnter = fmaxf(fmaxf(fmaxf(tn.x, tn.y), tn.z), 0.0f);
    float tExit = fminf(fminf(fminf(tf.x, tf.y), tf.z), tMax);
    return tEnter <= tExit ? tEnter : FLT_MAX;
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
    glm::vec3 invDir = safeInverse(q.direction);
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

// Moller-Trumbore. Works with unnormalized directions; t is in ray units.
__host__ __device__ inline bool triangleIntersectionTest(const Ray& q,
    glm::vec3 v0, glm::vec3 e1, glm::vec3 e2, float tMax, float& t, float& b1, float& b2)
{
    glm::vec3 pvec = glm::cross(q.direction, e2);
    float det = glm::dot(e1, pvec);
    if (det == 0.0f)
    {
        return false;
    }
    float invDet = 1.0f / det;
    glm::vec3 tvec = q.origin - v0;
    b1 = glm::dot(tvec, pvec) * invDet;
    if (b1 < 0.0f || b1 > 1.0f)
    {
        return false;
    }
    glm::vec3 qvec = glm::cross(tvec, e1);
    b2 = glm::dot(q.direction, qvec) * invDet;
    if (b2 < 0.0f || b1 + b2 > 1.0f)
    {
        return false;
    }
    t = glm::dot(e2, qvec) * invDet;
    return t > T_MIN && t < tMax;
}

__device__ inline void loadTriangle(const TriangleGeom* tris, int i, glm::vec3& v0, glm::vec3& e1, glm::vec3& e2)
{
    const float4* p = reinterpret_cast<const float4*>(tris + i);
    v0 = toVec3(__ldg(p));
    e1 = toVec3(__ldg(p + 1));
    e2 = toVec3(__ldg(p + 2));
}

__device__ inline void loadNode(const BVHNode* nodes, int i, glm::vec3& bmin, glm::vec3& bmax,
    int& leftFirst, int& triCount)
{
    const float4* p = reinterpret_cast<const float4*>(nodes + i);
    float4 a = __ldg(p);
    float4 b = __ldg(p + 1);
    bmin = glm::vec3(a.x, a.y, a.z);
    leftFirst = __float_as_int(a.w);
    bmax = glm::vec3(b.x, b.y, b.z);
    triCount = __float_as_int(b.w);
}

// Tests the triangles [first, first + count). Updates tBest/prim/bary on a
// closer hit. With anyHit, returns on the first hit found.
__device__ inline bool intersectTriangleRange(const TriangleGeom* tris, int first, int count,
    const Ray& q, float& tBest, int& prim, glm::vec2& bary, bool anyHit)
{
    bool hit = false;
    for (int i = first; i < first + count; i++)
    {
        glm::vec3 v0, e1, e2;
        loadTriangle(tris, i, v0, e1, e2);
        float t, b1, b2;
        if (triangleIntersectionTest(q, v0, e1, e2, tBest, t, b1, b2))
        {
            tBest = t;
            prim = i;
            bary = glm::vec2(b1, b2);
            hit = true;
            if (anyHit)
            {
                return true;
            }
        }
    }
    return hit;
}

// Iterative BVH traversal (object space). Children are visited near to far
// and far children are pushed with their entry distance so that they can be
// skipped once a closer hit is known.
__device__ inline bool intersectBVH(const SceneView& scene, const MeshInfo& mesh, const Ray& q,
    float& tBest, int& prim, glm::vec2& bary, bool anyHit)
{
    glm::vec3 invDir = safeInverse(q.direction);
    glm::vec3 bmin, bmax;
    int leftFirst, triCount;
    int node = mesh.rootNode;
    loadNode(scene.bvhNodes, node, bmin, bmax, leftFirst, triCount);
    if (rayAABB(q.origin, invDir, bmin, bmax, tBest) == FLT_MAX)
    {
        return false;
    }

    int stackNode[BVH_STACK_SIZE];
    float stackDist[BVH_STACK_SIZE];
    int sp = 0;
    bool hit = false;
    while (true)
    {
        if (triCount > 0)
        {
            if (intersectTriangleRange(scene.triGeoms, leftFirst, triCount, q, tBest, prim, bary, anyHit))
            {
                hit = true;
                if (anyHit)
                {
                    return true;
                }
            }
            // pop the next node that can still contain a closer hit
            node = -1;
            while (sp > 0)
            {
                --sp;
                if (stackDist[sp] < tBest)
                {
                    node = stackNode[sp];
                    break;
                }
            }
            if (node < 0)
            {
                break;
            }
            loadNode(scene.bvhNodes, node, bmin, bmax, leftFirst, triCount);
            continue;
        }

        int c1 = leftFirst;
        int c2 = leftFirst + 1;
        glm::vec3 min1, max1, min2, max2;
        int lf1, tc1, lf2, tc2;
        loadNode(scene.bvhNodes, c1, min1, max1, lf1, tc1);
        loadNode(scene.bvhNodes, c2, min2, max2, lf2, tc2);
        float d1 = rayAABB(q.origin, invDir, min1, max1, tBest);
        float d2 = rayAABB(q.origin, invDir, min2, max2, tBest);
        if (d2 < d1)
        {
            float td = d1; d1 = d2; d2 = td;
            int tn = c1; c1 = c2; c2 = tn;
            glm::vec3 tmin = min1; min1 = min2; min2 = tmin;
            glm::vec3 tmax = max1; max1 = max2; max2 = tmax;
            int t = lf1; lf1 = lf2; lf2 = t;
            t = tc1; tc1 = tc2; tc2 = t;
        }
        if (d1 == FLT_MAX)
        {
            node = -1;
            while (sp > 0)
            {
                --sp;
                if (stackDist[sp] < tBest)
                {
                    node = stackNode[sp];
                    break;
                }
            }
            if (node < 0)
            {
                break;
            }
            loadNode(scene.bvhNodes, node, bmin, bmax, leftFirst, triCount);
            continue;
        }
        if (d2 != FLT_MAX)
        {
            stackNode[sp] = c2;
            stackDist[sp] = d2;
            sp++;
        }
        node = c1;
        leftFirst = lf1;
        triCount = tc1;
    }
    return hit;
}

__device__ inline bool intersectMesh(const SceneView& scene, const MeshInfo& mesh, const Ray& q,
    float& tBest, int& prim, glm::vec2& bary, bool anyHit)
{
    if (scene.useBVH)
    {
        return intersectBVH(scene, mesh, q, tBest, prim, bary, anyHit);
    }
    // brute force over every triangle of the mesh
    return intersectTriangleRange(scene.triGeoms, mesh.triOffset, mesh.triCount, q, tBest, prim, bary, anyHit);
}

// Closest hit over all scene geometry. Objects are tested in a flat loop (the
// top level); triangle meshes descend into their own BVH (the bottom level).
__device__ inline void intersectScene(const SceneView& scene, const Ray& ray, ShadeableIntersection& isect)
{
    float tBest = FLT_MAX;
    int hitGeom = -1;
    int hitPrim = -1;
    glm::vec2 hitBary(0.0f);
    glm::vec3 invDir = safeInverse(ray.direction);

    for (int i = 0; i < scene.geomCount; i++)
    {
        const Geom& geom = scene.geoms[i];
        if (scene.cullBounds
            && rayAABB(ray.origin, invDir, geom.worldBounds.min, geom.worldBounds.max, tBest) == FLT_MAX)
        {
            continue;
        }
        Ray q = toObjectSpace(geom, ray);
        // TODO: add more intersection tests here... triangle? metaball? CSG?
        if (geom.type == MESH)
        {
            int prim;
            glm::vec2 bary;
            if (intersectMesh(scene, scene.meshes[geom.meshId], q, tBest, prim, bary, false))
            {
                hitGeom = i;
                hitPrim = prim;
                hitBary = bary;
            }
        }
        else
        {
            float t = geom.type == CUBE ? boxIntersectionTest(q) : sphereIntersectionTest(q);
            // Compute the minimum t from the intersection tests to determine
            // what scene geometry object was hit first.
            if (t > 0.0f && t < tBest)
            {
                tBest = t;
                hitGeom = i;
                hitPrim = -1;
            }
        }
    }

    if (hitGeom < 0)
    {
        isect.t = -1.0f;
        isect.materialId = -1;
        isect.geomId = -1;
        isect.primId = -1;
        return;
    }
    const Geom& geom = scene.geoms[hitGeom];
    isect.t = tBest;
    isect.geomId = hitGeom;
    isect.primId = hitPrim;
    isect.bary = hitBary;
    isect.materialId = (hitPrim >= 0 && geom.materialid < 0)
        ? scene.triangles[hitPrim].materialId : geom.materialid;
}

// Returns true if anything blocks the segment origin + t * direction for
// t in (0, maxT). Used for shadow rays.
__device__ inline bool occluded(const SceneView& scene, const Ray& ray, float maxT)
{
    glm::vec3 invDir = safeInverse(ray.direction);
    for (int i = 0; i < scene.geomCount; i++)
    {
        const Geom& geom = scene.geoms[i];
        if (scene.cullBounds
            && rayAABB(ray.origin, invDir, geom.worldBounds.min, geom.worldBounds.max, maxT) == FLT_MAX)
        {
            continue;
        }
        Ray q = toObjectSpace(geom, ray);
        if (geom.type == MESH)
        {
            float tBest = maxT;
            int prim;
            glm::vec2 bary;
            if (intersectMesh(scene, scene.meshes[geom.meshId], q, tBest, prim, bary, true))
            {
                return true;
            }
        }
        else
        {
            float t = geom.type == CUBE ? boxIntersectionTest(q) : sphereIntersectionTest(q);
            if (t > 0.0f && t < maxT)
            {
                return true;
            }
        }
    }
    return false;
}

// Rebuilds the full surface description of a hit for shading.
__device__ inline SurfaceHit computeSurfaceHit(const SceneView& scene, const Ray& ray,
    const ShadeableIntersection& isect)
{
    SurfaceHit hit;
    const Geom& geom = scene.geoms[isect.geomId];
    hit.position = ray.origin + isect.t * ray.direction;
    hit.tangent = glm::vec3(0.0f);
    hit.tangentSign = 1.0f;

    glm::vec3 objectNormal;
    glm::vec3 objectShadingNormal;
    glm::vec3 objectTangent(0.0f);
    if (isect.primId >= 0)
    {
        const Triangle tri = scene.triangles[isect.primId];
        glm::vec3 v0, e1, e2;
        loadTriangle(scene.triGeoms, isect.primId, v0, e1, e2);
        objectNormal = glm::cross(e1, e2);
        float b1 = isect.bary.x, b2 = isect.bary.y, b0 = 1.0f - b1 - b2;
        glm::vec3 n = b0 * scene.normals[tri.v[0]] + b1 * scene.normals[tri.v[1]] + b2 * scene.normals[tri.v[2]];
        objectShadingNormal = glm::dot(n, n) > 1e-12f ? n : objectNormal;
        hit.uv = b0 * scene.uvs[tri.v[0]] + b1 * scene.uvs[tri.v[1]] + b2 * scene.uvs[tri.v[2]];
        hit.objectPosition = v0 + b1 * e1 + b2 * e2;
        glm::vec4 t0 = scene.tangents[tri.v[0]];
        glm::vec4 t = b0 * t0 + b1 * scene.tangents[tri.v[1]] + b2 * scene.tangents[tri.v[2]];
        objectTangent = glm::vec3(t);
        hit.tangentSign = t0.w < 0.0f ? -1.0f : 1.0f;
    }
    else
    {
        Ray q = toObjectSpace(geom, ray);
        glm::vec3 p = q.origin + isect.t * q.direction;
        hit.objectPosition = p;
        if (geom.type == CUBE)
        {
            // Each face gets the [0,1]^2 square spanned by the other two axes.
            glm::vec3 a = glm::abs(p);
            if (a.x >= a.y && a.x >= a.z)
            {
                objectNormal = glm::vec3(p.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
                hit.uv = glm::vec2(p.z, p.y) + 0.5f;
                objectTangent = glm::vec3(0.0f, 0.0f, 1.0f);
            }
            else if (a.y >= a.z)
            {
                objectNormal = glm::vec3(0.0f, p.y > 0.0f ? 1.0f : -1.0f, 0.0f);
                hit.uv = glm::vec2(p.x, p.z) + 0.5f;
                objectTangent = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            else
            {
                objectNormal = glm::vec3(0.0f, 0.0f, p.z > 0.0f ? 1.0f : -1.0f);
                hit.uv = glm::vec2(p.x, p.y) + 0.5f;
                objectTangent = glm::vec3(1.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            objectNormal = p;
            float phi = atan2f(p.z, p.x);
            float theta = acosf(glm::clamp(2.0f * p.y, -1.0f, 1.0f));
            hit.uv = glm::vec2(phi * (0.5f * INV_PI) + 0.5f, 1.0f - theta * INV_PI);
            objectTangent = glm::vec3(-p.z, 0.0f, p.x);
        }
        objectShadingNormal = objectNormal;
    }

    glm::mat3 normalXform(geom.invTranspose);
    glm::vec3 n = glm::normalize(normalXform * objectNormal);
    // (translation-only motion leaves normals and tangents unchanged)
    glm::vec3 ns = glm::normalize(normalXform * objectShadingNormal);
    hit.frontFace = glm::dot(n, ray.direction) < 0.0f;
    hit.normal = hit.frontFace ? n : -n;
    hit.shadingNormal = hit.frontFace ? ns : -ns;
    if (glm::dot(hit.shadingNormal, hit.normal) <= 0.0f)
    {
        hit.shadingNormal = hit.normal;
    }
    if (glm::dot(objectTangent, objectTangent) > 0.0f)
    {
        hit.tangent = glm::normalize(glm::mat3(geom.transform) * objectTangent);
    }
    return hit;
}
