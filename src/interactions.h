#pragma once

#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>

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

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the throughput of the path in place.
 */
__host__ __device__ inline void scatterRay(
    PathSegment& pathSegment,
    const SurfaceHit& hit,
    const Material& m,
    glm::vec2 u)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 newDirection;
    if (m.type == MATERIAL_SPECULAR)
    {
        // Perfect mirror: the BSDF is a delta distribution, so f * cos / pdf
        // reduces to the specular tint.
        newDirection = glm::reflect(pathSegment.ray.direction, hit.normal);
    }
    else
    {
        // Ideal diffuse: f = albedo / pi and the cosine-weighted pdf is
        // cos / pi, so f * cos / pdf reduces to the albedo.
        newDirection = calculateRandomDirectionInHemisphere(hit.normal, u);
    }

    pathSegment.throughput *= m.color;
    pathSegment.ray.origin = hit.position + hit.normal * RAY_EPSILON;
    pathSegment.ray.direction = glm::normalize(newDirection);
    pathSegment.remainingBounces--;
}
