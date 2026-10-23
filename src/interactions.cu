#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
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
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 wo = pathSegment.ray.direction;
    glm::vec3 wi;
    if (m.hasRefractive > 0.0f)
    {
        // the normal always faces the incoming ray
        float eta = outside ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
        float cosI = -glm::dot(wo, normal);
        float k = 1.0f - eta * eta * (1.0f - cosI * cosI);
        if (k < 0.0f)
        {
            // total internal reflection, glm::refract gives NaN here
            wi = glm::reflect(wo, normal);
        }
        else
        {
            // schlick uses the angle on the air side
            float cosTheta = outside ? cosI : sqrtf(k);
            float r0 = (1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction);
            r0 = r0 * r0;
            float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);

            // pick reflection with probability fresnel, so the weight is 1
            thrust::uniform_real_distribution<float> u01(0, 1);
            wi = u01(rng) < fresnel ? glm::reflect(wo, normal) : glm::refract(wo, normal, eta);
        }
    }
    else if (m.hasReflective > 0.0f)
    {
        // perfect mirror
        wi = glm::reflect(wo, normal);
    }
    else
    {
        // cosine pdf cancels cos / pi, so only the albedo is left
        wi = calculateRandomDirectionInHemisphere(normal, rng);
    }
    pathSegment.color *= m.color;
    // push the origin to the side the new ray leaves from
    float side = glm::dot(wi, normal) > 0.0f ? 1.0f : -1.0f;
    pathSegment.ray.origin = intersect + normal * side * 0.001f;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.remainingBounces--;
}
