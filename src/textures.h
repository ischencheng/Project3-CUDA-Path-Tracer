#pragma once

#include "sceneStructs.h"
#include "mathUtils.h"
#include "interactions.h"

#include <glm/glm.hpp>

// Texture lookups, procedural textures and the evaluation of a Material at a
// surface point into BSDF parameters.

__device__ inline glm::vec4 sampleTexture(cudaTextureObject_t tex, glm::vec2 uv)
{
    float4 c = tex2D<float4>(tex, uv.x, uv.y);
    return glm::vec4(c.x, c.y, c.z, c.w);
}

__host__ __device__ inline float srgbToLinear(float c)
{
    return c <= 0.04045f ? c * (1.0f / 12.92f) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

__host__ __device__ inline glm::vec3 srgbToLinear(glm::vec3 c)
{
    return glm::vec3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z));
}

// ---------------------------------------------------------------------------
// Procedural noise: 3D gradient (Perlin-style) noise with hashed gradients.

__host__ __device__ inline uint32_t noiseHash(int x, int y, int z)
{
    uint32_t h = (uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ (uint32_t)z * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

__host__ __device__ inline float gradientDot(uint32_t h, glm::vec3 d)
{
    // 12 edge directions of a cube, as in improved Perlin noise.
    switch (h % 12u)
    {
    case 0: return d.x + d.y;
    case 1: return -d.x + d.y;
    case 2: return d.x - d.y;
    case 3: return -d.x - d.y;
    case 4: return d.x + d.z;
    case 5: return -d.x + d.z;
    case 6: return d.x - d.z;
    case 7: return -d.x - d.z;
    case 8: return d.y + d.z;
    case 9: return -d.y + d.z;
    case 10: return d.y - d.z;
    default: return -d.y - d.z;
    }
}

__host__ __device__ inline float fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Returns noise in roughly [-1, 1].
__host__ __device__ inline float gradientNoise(glm::vec3 p)
{
    glm::vec3 pf = glm::floor(p);
    glm::ivec3 i(pf);
    glm::vec3 f = p - pf;
    glm::vec3 u(fade(f.x), fade(f.y), fade(f.z));
    float n000 = gradientDot(noiseHash(i.x, i.y, i.z), f);
    float n100 = gradientDot(noiseHash(i.x + 1, i.y, i.z), f - glm::vec3(1, 0, 0));
    float n010 = gradientDot(noiseHash(i.x, i.y + 1, i.z), f - glm::vec3(0, 1, 0));
    float n110 = gradientDot(noiseHash(i.x + 1, i.y + 1, i.z), f - glm::vec3(1, 1, 0));
    float n001 = gradientDot(noiseHash(i.x, i.y, i.z + 1), f - glm::vec3(0, 0, 1));
    float n101 = gradientDot(noiseHash(i.x + 1, i.y, i.z + 1), f - glm::vec3(1, 0, 1));
    float n011 = gradientDot(noiseHash(i.x, i.y + 1, i.z + 1), f - glm::vec3(0, 1, 1));
    float n111 = gradientDot(noiseHash(i.x + 1, i.y + 1, i.z + 1), f - glm::vec3(1, 1, 1));
    float nx00 = glm::mix(n000, n100, u.x);
    float nx10 = glm::mix(n010, n110, u.x);
    float nx01 = glm::mix(n001, n101, u.x);
    float nx11 = glm::mix(n011, n111, u.x);
    return glm::mix(glm::mix(nx00, nx10, u.y), glm::mix(nx01, nx11, u.y), u.z);
}

// Fractal Brownian motion: octaves of noise with halving amplitude.
__host__ __device__ inline float fbm(glm::vec3 p, int octaves)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    for (int o = 0; o < octaves; o++)
    {
        sum += amplitude * gradientNoise(p);
        p *= 2.03f;
        amplitude *= 0.5f;
    }
    return sum;
}

__host__ __device__ inline float turbulence(glm::vec3 p, int octaves)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    for (int o = 0; o < octaves; o++)
    {
        sum += amplitude * fabsf(gradientNoise(p));
        p *= 2.03f;
        amplitude *= 0.5f;
    }
    return sum;
}

// Blend factor in [0, 1] between Material::color and procColor2.
__host__ __device__ inline float proceduralPattern(int type, glm::vec3 p, glm::vec2 uv, float scale)
{
    switch (type)
    {
    case PROCEDURAL_CHECKER:
    {
        glm::vec2 c = glm::floor(uv * scale);
        return fmodf(fabsf(c.x + c.y), 2.0f);
    }
    case PROCEDURAL_MARBLE:
    {
        // thin veins where a turbulence-distorted sine wave crosses zero
        glm::vec3 q = p * scale;
        float s = sinf(q.x * 2.0f + 6.0f * turbulence(q, 5));
        return powf(1.0f - fabsf(s), 6.0f);
    }
    case PROCEDURAL_WOOD:
    {
        // concentric rings around the y axis, wobbled by low frequency noise
        glm::vec3 q = p * scale;
        float r = sqrtf(q.x * q.x + q.z * q.z) + 0.6f * fbm(q * glm::vec3(0.5f, 0.1f, 0.5f), 3);
        float ring = r * 4.0f - floorf(r * 4.0f);
        return glm::smoothstep(0.0f, 0.15f, ring) * (1.0f - glm::smoothstep(0.55f, 1.0f, ring));
    }
    case PROCEDURAL_NOISE:
        return glm::clamp(0.5f + fbm(p * scale, 6), 0.0f, 1.0f);
    default:
        return 0.0f;
    }
}

// Height field used for procedural bump mapping.
__host__ __device__ inline float bumpHeight(glm::vec3 p, float scale)
{
    return fbm(p * scale, 4);
}

// Everything the shading kernel needs to know about the surface material at
// a hit point, after textures have been applied.
struct MaterialEval
{
    BSDFParams bsdf;
    glm::vec3 emission;
    glm::vec3 shadingNormal;    // possibly perturbed by normal/bump mapping
    glm::vec3 albedo;           // first-hit albedo (denoiser feature)
};

__device__ inline MaterialEval evaluateMaterial(const Material& m, const SurfaceHit& hit,
    const cudaTextureObject_t* textures, glm::vec3 wo)
{
    MaterialEval e;
    glm::vec2 uv = hit.uv * m.uvScale;

    glm::vec3 color = m.color;
    if (m.baseColorTex >= 0)
    {
        color *= srgbToLinear(glm::vec3(sampleTexture(textures[m.baseColorTex], uv)));
    }
    if (m.procedural != PROCEDURAL_NONE)
    {
        float t = proceduralPattern(m.procedural, hit.objectPosition, hit.uv, m.procScale);
        color = glm::mix(color, m.procColor2, t);
    }

    float roughness = m.roughness;
    float metallic = m.metallic;
    if (m.metallicRoughnessTex >= 0)
    {
        // glTF packs roughness in G and metalness in B (linear)
        glm::vec4 mr = sampleTexture(textures[m.metallicRoughnessTex], uv);
        roughness *= mr.y;
        metallic *= mr.z;
    }

    e.emission = m.emission;
    if (m.emissiveTex >= 0)
    {
        e.emission *= srgbToLinear(glm::vec3(sampleTexture(textures[m.emissiveTex], uv)));
    }

    // Shading frame perturbation.
    glm::vec3 n = hit.shadingNormal;
    if (m.normalTex >= 0 && glm::dot(hit.tangent, hit.tangent) > 0.0f)
    {
        glm::vec3 t = glm::normalize(hit.tangent - n * glm::dot(n, hit.tangent));
        glm::vec3 b = glm::cross(n, t) * hit.tangentSign;
        glm::vec3 c = glm::vec3(sampleTexture(textures[m.normalTex], uv)) * 2.0f - glm::vec3(1.0f);
        c.x *= m.normalScale;
        c.y *= m.normalScale;
        glm::vec3 mapped = c.x * t + c.y * b + c.z * n;
        if (glm::dot(mapped, mapped) > 1e-12f)
        {
            n = glm::normalize(mapped);
        }
    }
    if (m.bumpStrength > 0.0f)
    {
        // Bump mapping: tilt the normal against the gradient of a procedural
        // height field, estimated with forward differences along the
        // tangent plane (PBRT v3 9.3 with the displacement along n).
        // The height field is defined in world space so that the finite
        // differences step along the same axes as the world-space frame.
        Frame f = makeFrame(n);
        const float eps = 1e-3f;
        glm::vec3 p = hit.position;
        float h = bumpHeight(p, m.bumpScale);
        float dhdt = (bumpHeight(p + eps * f.t, m.bumpScale) - h) / eps;
        float dhdb = (bumpHeight(p + eps * f.b, m.bumpScale) - h) / eps;
        n = glm::normalize(n - m.bumpStrength * (dhdt * f.t + dhdb * f.b));
    }
    // keep the shading normal on the visible side
    if (glm::dot(n, wo) <= 0.0f)
    {
        n = hit.shadingNormal;
    }
    e.shadingNormal = n;

    e.bsdf.type = m.type;
    e.bsdf.color = color;
    e.bsdf.alpha = roughnessToAlpha(glm::clamp(roughness, 0.0f, 1.0f));
    e.bsdf.metallic = glm::clamp(metallic, 0.0f, 1.0f);
    e.bsdf.etap = hit.frontFace ? m.ior : 1.0f / m.ior;
    if (m.type == MATERIAL_DIELECTRIC && m.ior == 1.0f)
    {
        // An interface without an IOR change cannot bend rays, so it is
        // smooth whatever its roughness (as in PBRT); the rough code path
        // would build a zero-length half vector from wi = -wo.
        e.bsdf.alpha = 0.0f;
    }
    e.albedo = color;
    return e;
}
