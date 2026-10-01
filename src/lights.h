#pragma once

#include "sceneStructs.h"
#include "mathUtils.h"
#include "intersections.h"
#include "textures.h"

#include <glm/glm.hpp>

// Light sampling for next event estimation and the matching pdfs for
// multiple importance sampling. All pdfs are with respect to solid angle at
// the shading point and include the probability of selecting the light.

// ---------------------------------------------------------------------------
// Environment

__host__ __device__ inline glm::vec2 directionToEquirect(glm::vec3 d, float rotation)
{
    float phi = atan2f(d.z, d.x) + rotation;
    float theta = acosf(glm::clamp(d.y, -1.0f, 1.0f));
    float u = phi * (0.5f * INV_PI) + 0.5f;
    u -= floorf(u);
    return glm::vec2(u, theta * INV_PI);
}

__host__ __device__ inline glm::vec3 equirectToDirection(glm::vec2 uv, float rotation)
{
    float phi = (uv.x - 0.5f) * TWO_PI - rotation;
    float theta = uv.y * PI;
    float sinTheta = sinf(theta);
    return glm::vec3(sinTheta * cosf(phi), cosf(theta), sinTheta * sinf(phi));
}

__device__ inline glm::vec3 environmentRadiance(const EnvironmentView& env, glm::vec3 dir)
{
    if (!env.hasMap)
    {
        return env.color;
    }
    glm::vec2 uv = directionToEquirect(dir, env.rotation);
    float4 c = tex2D<float4>(env.texture, uv.x, uv.y);
    return glm::vec3(c.x, c.y, c.z) * env.intensity;
}

// Largest i with cdf[i] <= u, for a cdf of n + 1 entries starting at 0.
__device__ inline int findInterval(const float* cdf, int n, float u)
{
    int lo = 0;
    int hi = n;
    while (lo + 1 < hi)
    {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] <= u) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Solid angle pdf of sampling `dir` from the environment (not including the
// environment selection probability).
__device__ inline float environmentPdf(const EnvironmentView& env, glm::vec3 dir)
{
    if (!env.hasMap)
    {
        return 0.25f * INV_PI;
    }
    glm::vec2 uv = directionToEquirect(dir, env.rotation);
    float sinTheta = sinf(uv.y * PI);
    if (sinTheta <= 0.0f || env.integral <= 0.0f)
    {
        return 0.0f;
    }
    int x = glm::min((int)(uv.x * env.width), env.width - 1);
    int y = glm::min((int)(uv.y * env.height), env.height - 1);
    float pdfUV = env.func[y * env.width + x] / env.integral;
    return pdfUV / (2.0f * PI * PI * sinTheta);
}

__device__ inline glm::vec3 sampleEnvironment(const EnvironmentView& env, glm::vec2 u, float& pdf)
{
    if (!env.hasMap)
    {
        pdf = 0.25f * INV_PI;
        return uniformSampleSphere(u);
    }
    // piecewise constant 2D distribution: marginal over rows, then the row
    int y = findInterval(env.marginalCdf, env.height, u.x);
    float c0 = env.marginalCdf[y];
    float c1 = env.marginalCdf[y + 1];
    float dv = c1 > c0 ? (u.x - c0) / (c1 - c0) : 0.5f;
    const float* row = env.conditionalCdf + y * (env.width + 1);
    int x = findInterval(row, env.width, u.y);
    float r0 = row[x];
    float r1 = row[x + 1];
    float du = r1 > r0 ? (u.y - r0) / (r1 - r0) : 0.5f;
    glm::vec2 uv((x + du) / env.width, (y + dv) / env.height);
    float sinTheta = sinf(uv.y * PI);
    if (sinTheta <= 0.0f || env.integral <= 0.0f)
    {
        pdf = 0.0f;
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }
    pdf = env.func[y * env.width + x] / env.integral / (2.0f * PI * PI * sinTheta);
    return equirectToDirection(uv, env.rotation);
}

// ---------------------------------------------------------------------------
// Area lights

struct LightSample
{
    glm::vec3 wi;           // direction from the shading point to the light
    float distance;
    glm::vec3 radiance;
    float pdf;              // solid angle, including selection probability
};

// Emission of a light-emitting triangle at barycentrics (b1, b2).
__device__ inline glm::vec3 triangleEmission(const SceneView& scene, int prim, glm::vec2 bary)
{
    const Triangle tri = scene.triangles[prim];
    const Material& m = scene.materials[tri.materialId];
    glm::vec3 e = m.emission;
    if (m.emissiveTex >= 0)
    {
        float b0 = 1.0f - bary.x - bary.y;
        glm::vec2 uv = b0 * scene.uvs[tri.v[0]] + bary.x * scene.uvs[tri.v[1]] + bary.y * scene.uvs[tri.v[2]];
        e *= srgbToLinear(glm::vec3(sampleTexture(scene.textures[m.emissiveTex], uv * m.uvScale)));
    }
    return e;
}

// pdf (solid angle) of sampling the point on `light` seen from `ref` along
// `dir` at distance `dist`, where the light surface normal there is `n`.
__device__ inline float areaLightPdf(const SceneView& scene, const Light& light, glm::vec3 ref,
    glm::vec3 dir, float dist, glm::vec3 n)
{
    if (light.type == LIGHT_SPHERE)
    {
        const Geom& g = scene.geoms[light.geomId];
        float r = 0.5f * g.scale.x;
        float d2 = glm::dot(g.translation - ref, g.translation - ref);
        if (d2 <= r * r)
        {
            // inside the sphere: uniform area sampling
            float cosLight = fabsf(glm::dot(n, dir));
            return cosLight > 0.0f ? light.selectPdf * dist * dist / (cosLight * light.area) : 0.0f;
        }
        float cosMax = sqrtf(fmaxf(0.0f, 1.0f - r * r / d2));
        return light.selectPdf / (TWO_PI * (1.0f - cosMax));
    }
    float cosLight = fabsf(glm::dot(n, dir));
    return cosLight > 0.0f ? light.selectPdf * dist * dist / (cosLight * light.area) : 0.0f;
}

__device__ inline bool sampleAreaLight(const SceneView& scene, const Light& light, glm::vec3 ref,
    glm::vec2 u, LightSample& ls)
{
    glm::vec3 p;
    glm::vec3 n;
    glm::vec3 emission;
    if (light.type == LIGHT_TRIANGLE)
    {
        glm::vec2 b = uniformSampleTriangle(u);
        p = light.v0 + b.x * light.e1 + b.y * light.e2;
        n = glm::normalize(glm::cross(light.e1, light.e2));
        emission = triangleEmission(scene, light.primId, b);
    }
    else if (light.type == LIGHT_SPHERE)
    {
        const Geom& g = scene.geoms[light.geomId];
        glm::vec3 c = g.translation;
        float r = 0.5f * g.scale.x;
        glm::vec3 toCenter = c - ref;
        float d2 = glm::dot(toCenter, toCenter);
        emission = scene.materials[g.materialid].emission;
        if (d2 <= r * r)
        {
            p = c + r * uniformSampleSphere(u);
            n = glm::normalize(p - c);
        }
        else
        {
            // Sample the cone of directions subtended by the sphere.
            float d = sqrtf(d2);
            float sin2Max = r * r / d2;
            float cosMax = sqrtf(fmaxf(0.0f, 1.0f - sin2Max));
            float cosTheta = 1.0f - u.x * (1.0f - cosMax);
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float phi = TWO_PI * u.y;
            Frame f = makeFrame(toCenter / d);
            glm::vec3 wi = f.toWorld(glm::vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta));
            // distance to the near intersection with the sphere
            float b = d * cosTheta;
            float disc = fmaxf(0.0f, r * r - d2 * sinTheta * sinTheta);
            float t = b - sqrtf(disc);
            ls.wi = wi;
            ls.distance = t;
            ls.radiance = emission;
            ls.pdf = light.selectPdf / (TWO_PI * (1.0f - cosMax));
            return ls.pdf > 0.0f && t > 0.0f;
        }
    }
    else
    {
        // Cube: pick a face proportionally to its area, then a point on it.
        const Geom& g = scene.geoms[light.geomId];
        emission = scene.materials[g.materialid].emission;
        glm::vec3 s = g.scale;
        float ax = s.y * s.z, ay = s.x * s.z, az = s.x * s.y;
        float total = ax + ay + az;
        float pick = u.x * total;
        int axis;
        float sign = 1.0f;
        if (pick < ax) { axis = 0; u.x = pick / ax; }
        else if (pick < ax + ay) { axis = 1; u.x = (pick - ax) / ay; }
        else { axis = 2; u.x = (pick - ax - ay) / az; }
        if (u.x < 0.5f) { sign = -1.0f; u.x *= 2.0f; }
        else { u.x = (u.x - 0.5f) * 2.0f; }
        glm::vec3 local(0.0f);
        glm::vec3 nl(0.0f);
        local[axis] = 0.5f * sign;
        nl[axis] = sign;
        local[(axis + 1) % 3] = u.x - 0.5f;
        local[(axis + 2) % 3] = u.y - 0.5f;
        p = glm::vec3(g.transform * glm::vec4(local, 1.0f));
        n = glm::normalize(glm::vec3(g.invTranspose * glm::vec4(nl, 0.0f)));
    }

    glm::vec3 d = p - ref;
    float dist2 = glm::dot(d, d);
    if (dist2 <= 0.0f)
    {
        return false;
    }
    ls.distance = sqrtf(dist2);
    ls.wi = d / ls.distance;
    ls.radiance = emission;
    float cosLight = fabsf(glm::dot(n, ls.wi));
    if (cosLight <= 1e-6f)
    {
        return false;
    }
    ls.pdf = light.selectPdf * dist2 / (cosLight * light.area);
    return true;
}

// Picks an area light proportionally to its power.
__device__ inline int selectLight(const LightsView& lights, float u)
{
    return glm::min(findInterval(lights.cdf, lights.count, u), lights.count - 1);
}
