#pragma once

#include "sceneStructs.h"
#include "mathUtils.h"
#include "utilities.h"

#include <glm/glm.hpp>

// BSDF models. Directions handed to these functions follow the PBRT
// convention: both wo and wi point away from the surface, and all local-space
// math happens in a frame whose +z axis is the shading normal on the side of
// wo (so wo.z > 0 for the incoming path).

// GGX alpha below which a lobe is treated as perfectly smooth (delta).
#define DELTA_ALPHA 1e-3f

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting. `u` is a uniform 2D sample in [0,1)^2.
 */
__host__ __device__ inline glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    glm::vec2 u)
{
    float up = sqrtf(u.x); // cos(theta)
    float over = sqrtf(1.0f - up * up); // sin(theta)
    float around = u.y * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (fabsf(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (fabsf(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cosf(around) * over * perpendicularDirection1
        + sinf(around) * over * perpendicularDirection2;
}

// ---------------------------------------------------------------------------
// Fresnel

__host__ __device__ inline glm::vec3 fresnelSchlick(glm::vec3 f0, float cosTheta)
{
    float m = 1.0f - glm::clamp(cosTheta, 0.0f, 1.0f);
    float m2 = m * m;
    return f0 + (glm::vec3(1.0f) - f0) * (m2 * m2 * m);
}

// Exact unpolarized Fresnel reflectance of a dielectric interface.
// `eta` is the relative IOR (transmitted side over incident side) for a ray
// arriving from the side the normal points to (cosThetaI > 0).
__host__ __device__ inline float fresnelDielectric(float cosThetaI, float eta)
{
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);
    if (cosThetaI < 0.0f)
    {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
    }
    float sin2ThetaI = 1.0f - cosThetaI * cosThetaI;
    float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0f)
    {
        return 1.0f;    // total internal reflection
    }
    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sin2ThetaT));
    float rParallel = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerpendicular = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

// Refracts `wi` (pointing away from the surface) through the interface with
// normal `n`. Returns false on total internal reflection. `etap` receives the
// relative IOR actually used (flipped if wi is on the back side of n).
__host__ __device__ inline bool refractDirection(glm::vec3 wi, glm::vec3 n, float eta, float& etap, glm::vec3& wt)
{
    float cosThetaI = glm::dot(n, wi);
    if (cosThetaI < 0.0f)
    {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
        n = -n;
    }
    float sin2ThetaI = fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0f)
    {
        return false;
    }
    float cosThetaT = sqrtf(1.0f - sin2ThetaT);
    wt = -wi / eta + (cosThetaI / eta - cosThetaT) * n;
    etap = eta;
    return true;
}

// ---------------------------------------------------------------------------
// Trowbridge-Reitz (GGX) microfacet distribution, isotropic.

__host__ __device__ inline float ggxD(glm::vec3 wm, float alpha)
{
    float a2 = alpha * alpha;
    float c2 = wm.z * wm.z;
    float d = c2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * d * d);
}

__host__ __device__ inline float ggxLambda(glm::vec3 w, float alpha)
{
    float c2 = w.z * w.z;
    if (c2 <= 0.0f)
    {
        return 0.0f;
    }
    float tan2 = fmaxf(0.0f, 1.0f - c2) / c2;
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tan2) - 1.0f);
}

__host__ __device__ inline float ggxG1(glm::vec3 w, float alpha)
{
    return 1.0f / (1.0f + ggxLambda(w, alpha));
}

// Height-correlated Smith masking-shadowing.
__host__ __device__ inline float ggxG(glm::vec3 wo, glm::vec3 wi, float alpha)
{
    return 1.0f / (1.0f + ggxLambda(wo, alpha) + ggxLambda(wi, alpha));
}

// Distribution of normals visible from w.
__host__ __device__ inline float ggxDVisible(glm::vec3 w, glm::vec3 wm, float alpha)
{
    return ggxG1(w, alpha) / fabsf(w.z) * ggxD(wm, alpha) * fabsf(glm::dot(w, wm));
}

// Samples a visible microfacet normal (Heitz 2018). Requires w.z > 0.
__host__ __device__ inline glm::vec3 ggxSampleVisibleNormal(glm::vec3 w, float alpha, glm::vec2 u)
{
    glm::vec3 vh = glm::normalize(glm::vec3(alpha * w.x, alpha * w.y, w.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    glm::vec3 t1 = lensq > 0.0f ? glm::vec3(-vh.y, vh.x, 0.0f) / sqrtf(lensq) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 t2 = glm::cross(vh, t1);
    float r = sqrtf(u.x);
    float phi = TWO_PI * u.y;
    float p1 = r * cosf(phi);
    float p2 = r * sinf(phi);
    float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1)) + s * p2;
    glm::vec3 nh = p1 * t1 + p2 * t2 + sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2)) * vh;
    return glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, fmaxf(1e-6f, nh.z)));
}

// ---------------------------------------------------------------------------
// BSDF interface

// Material parameters after texture lookups, ready for BSDF evaluation.
struct BSDFParams
{
    int type;
    glm::vec3 color;    // albedo / specular F0 / transmission tint / base color
    float alpha;        // GGX roughness (perceptual roughness squared)
    float metallic;
    float etap;         // relative IOR across the surface for the incoming side
};

struct BSDFSample
{
    glm::vec3 wi;       // world space
    glm::vec3 weight;   // f * |cos| / pdf
    float pdf;          // solid angle pdf (lobe probability for delta lobes)
    bool isDelta;
    bool isTransmission;
};

__host__ __device__ inline float roughnessToAlpha(float roughness)
{
    return roughness * roughness;
}

// Probability of picking the specular lobe of the metallic-roughness model.
__host__ __device__ inline float pbrSpecularProbability(const BSDFParams& p, glm::vec3 f0, float cosThetaO)
{
    float spec = luminance(fresnelSchlick(f0, cosThetaO));
    float diff = luminance(p.color) * (1.0f - p.metallic);
    float sum = spec + diff;
    return sum > 0.0f ? glm::clamp(spec / sum, 0.1f, 1.0f) : 1.0f;
}

__host__ __device__ inline glm::vec3 pbrF0(const BSDFParams& p)
{
    return glm::mix(glm::vec3(0.04f), p.color, p.metallic);
}

// Evaluates f * |cos(theta_i)| and the solid-angle pdf of sampling wi with
// sampleBSDF. Delta lobes evaluate to zero.
__host__ __device__ inline glm::vec3 evalBSDF(const BSDFParams& p, const Frame& frame,
    glm::vec3 woWorld, glm::vec3 wiWorld, float& pdf)
{
    pdf = 0.0f;
    glm::vec3 wo = frame.toLocal(woWorld);
    glm::vec3 wi = frame.toLocal(wiWorld);
    if (wo.z <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    switch (p.type)
    {
    case MATERIAL_DIFFUSE:
    {
        if (wi.z <= 0.0f) return glm::vec3(0.0f);
        pdf = wi.z * INV_PI;
        return p.color * INV_PI * wi.z;
    }
    case MATERIAL_SPECULAR:
    {
        if (p.alpha < DELTA_ALPHA || wi.z <= 0.0f) return glm::vec3(0.0f);
        glm::vec3 wm = glm::normalize(wo + wi);
        float d = ggxD(wm, p.alpha);
        glm::vec3 f = d * ggxG(wo, wi, p.alpha) * fresnelSchlick(p.color, glm::dot(wo, wm)) / (4.0f * wo.z * wi.z);
        pdf = ggxDVisible(wo, wm, p.alpha) / (4.0f * glm::dot(wo, wm));
        return f * wi.z;
    }
    case MATERIAL_PBR:
    {
        if (wi.z <= 0.0f) return glm::vec3(0.0f);
        glm::vec3 f0 = pbrF0(p);
        float pSpec = pbrSpecularProbability(p, f0, wo.z);
        glm::vec3 wm = glm::normalize(wo + wi);
        glm::vec3 fresnel = fresnelSchlick(f0, glm::dot(wo, wm));
        glm::vec3 f = (glm::vec3(1.0f) - fresnel) * p.color * (1.0f - p.metallic) * INV_PI;
        pdf = (1.0f - pSpec) * wi.z * INV_PI;
        if (p.alpha >= DELTA_ALPHA)
        {
            f += ggxD(wm, p.alpha) * ggxG(wo, wi, p.alpha) * fresnel / (4.0f * wo.z * wi.z);
            pdf += pSpec * ggxDVisible(wo, wm, p.alpha) / (4.0f * glm::dot(wo, wm));
        }
        return f * wi.z;
    }
    case MATERIAL_DIELECTRIC:
    {
        if (p.alpha < DELTA_ALPHA || wi.z == 0.0f) return glm::vec3(0.0f);
        bool reflect = wi.z > 0.0f;
        float etap = reflect ? 1.0f : p.etap;
        glm::vec3 wm = wi * etap + wo;
        if (glm::dot(wm, wm) == 0.0f) return glm::vec3(0.0f);
        wm = glm::normalize(wm);
        if (wm.z < 0.0f) wm = -wm;
        // Discard back-facing microfacets.
        if (glm::dot(wm, wi) * wi.z < 0.0f || glm::dot(wm, wo) * wo.z < 0.0f) return glm::vec3(0.0f);
        float r = fresnelDielectric(glm::dot(wo, wm), p.etap);
        float t = 1.0f - r;
        float d = ggxD(wm, p.alpha);
        float g = ggxG(wo, wi, p.alpha);
        if (reflect)
        {
            pdf = ggxDVisible(wo, wm, p.alpha) / (4.0f * fabsf(glm::dot(wo, wm))) * r;
            return glm::vec3(d * g * r / (4.0f * wi.z * wo.z)) * fabsf(wi.z);
        }
        float denom = sqr(glm::dot(wi, wm) + glm::dot(wo, wm) / etap);
        pdf = ggxDVisible(wo, wm, p.alpha) * fabsf(glm::dot(wi, wm)) / denom * t;
        float ft = d * t * g * fabsf(glm::dot(wi, wm) * glm::dot(wo, wm) / (wi.z * wo.z * denom)) / (etap * etap);
        return p.color * ft * fabsf(wi.z);
    }
    default:
        return glm::vec3(0.0f);
    }
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them and divide by the probability of the chosen branch.
 *
 * Returns false if the path is absorbed.
 */
__host__ __device__ inline bool sampleBSDF(const BSDFParams& p, const Frame& frame,
    glm::vec3 woWorld, float uLobe, glm::vec2 u, BSDFSample& s)
{
    glm::vec3 wo = frame.toLocal(woWorld);
    if (wo.z <= 0.0f)
    {
        return false;
    }
    glm::vec3 wi;
    s.isDelta = false;
    s.isTransmission = false;

    switch (p.type)
    {
    case MATERIAL_DIFFUSE:
    {
        // Ideal diffuse: f = albedo / pi and the cosine-weighted pdf is
        // cos / pi, so f * cos / pdf reduces to the albedo.
        wi = cosineSampleHemisphere(u);
        s.pdf = wi.z * INV_PI;
        s.weight = p.color;
        break;
    }
    case MATERIAL_SPECULAR:
    {
        if (p.alpha < DELTA_ALPHA)
        {
            // Perfect mirror: the BSDF is a delta distribution, so f * cos /
            // pdf reduces to the Fresnel reflectance (the specular tint at
            // normal incidence).
            wi = glm::vec3(-wo.x, -wo.y, wo.z);
            s.pdf = 1.0f;
            s.weight = fresnelSchlick(p.color, wo.z);
            s.isDelta = true;
            break;
        }
        glm::vec3 wm = ggxSampleVisibleNormal(wo, p.alpha, u);
        wi = glm::reflect(-wo, wm);
        if (wi.z <= 0.0f)
        {
            return false;
        }
        s.pdf = ggxDVisible(wo, wm, p.alpha) / (4.0f * glm::dot(wo, wm));
        s.weight = fresnelSchlick(p.color, glm::dot(wo, wm)) * (ggxG(wo, wi, p.alpha) / ggxG1(wo, p.alpha));
        break;
    }
    case MATERIAL_PBR:
    {
        glm::vec3 f0 = pbrF0(p);
        float pSpec = pbrSpecularProbability(p, f0, wo.z);
        bool smooth = p.alpha < DELTA_ALPHA;
        if (uLobe < pSpec)
        {
            if (smooth)
            {
                wi = glm::vec3(-wo.x, -wo.y, wo.z);
                s.pdf = pSpec;
                s.weight = fresnelSchlick(f0, wo.z) / pSpec;
                s.isDelta = true;
                break;
            }
            glm::vec3 wm = ggxSampleVisibleNormal(wo, p.alpha, u);
            wi = glm::reflect(-wo, wm);
        }
        else
        {
            wi = cosineSampleHemisphere(u);
        }
        if (wi.z <= 0.0f)
        {
            return false;
        }
        // One-sample MIS over the lobes: weight by the combined pdf.
        glm::vec3 fcos = evalBSDF(p, frame, woWorld, frame.toWorld(wi), s.pdf);
        if (s.pdf <= 0.0f)
        {
            return false;
        }
        s.weight = fcos / s.pdf;
        break;
    }
    case MATERIAL_DIELECTRIC:
    {
        if (p.alpha < DELTA_ALPHA)
        {
            float r = fresnelDielectric(wo.z, p.etap);
            s.isDelta = true;
            if (uLobe < r)
            {
                wi = glm::vec3(-wo.x, -wo.y, wo.z);
                s.pdf = r;
                s.weight = glm::vec3(1.0f);
            }
            else
            {
                float etap;
                if (!refractDirection(wo, glm::vec3(0.0f, 0.0f, 1.0f), p.etap, etap, wi))
                {
                    return false;
                }
                s.pdf = 1.0f - r;
                // Radiance is compressed by eta^2 when entering a denser medium.
                s.weight = p.color / (etap * etap);
                s.isTransmission = true;
            }
            break;
        }

        glm::vec3 wm = ggxSampleVisibleNormal(wo, p.alpha, u);
        float r = fresnelDielectric(glm::dot(wo, wm), p.etap);
        if (uLobe < r)
        {
            wi = glm::reflect(-wo, wm);
            if (wi.z <= 0.0f)
            {
                return false;
            }
        }
        else
        {
            float etap;
            if (!refractDirection(wo, wm, p.etap, etap, wi) || wi.z >= 0.0f)
            {
                return false;
            }
            s.isTransmission = true;
        }
        glm::vec3 fcos = evalBSDF(p, frame, woWorld, frame.toWorld(wi), s.pdf);
        if (s.pdf <= 0.0f)
        {
            return false;
        }
        s.weight = fcos / s.pdf;
        break;
    }
    default:
        return false;
    }

    s.wi = glm::normalize(frame.toWorld(wi));
    return true;
}
