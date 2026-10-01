#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "utilities.h"

// Small math helpers shared by the device code.

__host__ __device__ inline float luminance(glm::vec3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__host__ __device__ inline float maxComponent(glm::vec3 c)
{
    return fmaxf(c.x, fmaxf(c.y, c.z));
}

__host__ __device__ inline float sqr(float x)
{
    return x * x;
}

__host__ __device__ inline bool isFiniteVec(glm::vec3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

// Orthonormal frame around a unit normal (Duff et al. 2017, branchless).
struct Frame
{
    glm::vec3 t;
    glm::vec3 b;
    glm::vec3 n;

    __host__ __device__ glm::vec3 toLocal(glm::vec3 v) const
    {
        return glm::vec3(glm::dot(v, t), glm::dot(v, b), glm::dot(v, n));
    }

    __host__ __device__ glm::vec3 toWorld(glm::vec3 v) const
    {
        return v.x * t + v.y * b + v.z * n;
    }
};

__host__ __device__ inline Frame makeFrame(glm::vec3 n)
{
    Frame f;
    float sign = copysignf(1.0f, n.z);
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    f.t = glm::vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    f.b = glm::vec3(b, sign + n.y * n.y * a, -n.y);
    f.n = n;
    return f;
}

// Frame from a normal and an approximate tangent (Gram-Schmidt).
__host__ __device__ inline Frame makeFrame(glm::vec3 n, glm::vec3 tangent, float handedness)
{
    glm::vec3 t = tangent - n * glm::dot(n, tangent);
    float len2 = glm::dot(t, t);
    if (!(len2 > 1e-12f))
    {
        return makeFrame(n);
    }
    Frame f;
    f.n = n;
    f.t = t * (1.0f / sqrtf(len2));
    f.b = glm::cross(n, f.t) * handedness;
    return f;
}

// Shirley-Chiu concentric mapping of [0,1)^2 onto the unit disk.
__host__ __device__ inline glm::vec2 concentricSampleDisk(glm::vec2 u)
{
    glm::vec2 o = 2.0f * u - glm::vec2(1.0f);
    if (o.x == 0.0f && o.y == 0.0f)
    {
        return glm::vec2(0.0f);
    }
    float r, theta;
    if (fabsf(o.x) > fabsf(o.y))
    {
        r = o.x;
        theta = (PI / 4.0f) * (o.y / o.x);
    }
    else
    {
        r = o.y;
        theta = (PI / 2.0f) - (PI / 4.0f) * (o.x / o.y);
    }
    return r * glm::vec2(cosf(theta), sinf(theta));
}

__host__ __device__ inline glm::vec3 cosineSampleHemisphere(glm::vec2 u)
{
    glm::vec2 d = concentricSampleDisk(u);
    float z = sqrtf(fmaxf(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return glm::vec3(d.x, d.y, z);
}

__host__ __device__ inline glm::vec3 uniformSampleSphere(glm::vec2 u)
{
    float z = 1.0f - 2.0f * u.x;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * u.y;
    return glm::vec3(r * cosf(phi), r * sinf(phi), z);
}

// Uniform barycentrics on a triangle (Heitz 2019 low-distortion map would also
// work; the square-root parameterization is fine here).
__host__ __device__ inline glm::vec2 uniformSampleTriangle(glm::vec2 u)
{
    float su = sqrtf(u.x);
    return glm::vec2(1.0f - su, u.y * su);
}

// Power heuristic with beta = 2 for multiple importance sampling.
__host__ __device__ inline float powerHeuristic(float pdfA, float pdfB)
{
    float a = pdfA * pdfA;
    float b = pdfB * pdfB;
    return a + b > 0.0f ? a / (a + b) : 0.0f;
}
