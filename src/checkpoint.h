#pragma once

#include "scene.h"
#include "utilities.h"

#include <glm/glm.hpp>

#include <string>
#include <vector>

// Restartable rendering. A checkpoint stores everything needed to continue a
// render later without the original assets: the complete scene (geometry,
// BVHs, materials, textures, environment map, light list), the camera and
// interactive orbit state, the render settings and the accumulation buffers.

struct OrbitState
{
    float zoom;
    float theta;
    float phi;
    glm::vec3 lookAt;
    glm::vec3 originalLookAt;
};

struct Checkpoint
{
    int iteration = 0;
    RenderSettings settings;
    OrbitState orbit{};
    std::vector<glm::vec3> image;   // accumulated radiance (sum over iterations)
    std::vector<glm::vec3> albedo;  // accumulated denoiser features
    std::vector<glm::vec3> normal;
};

bool saveCheckpoint(const std::string& path, const Scene& scene, const Checkpoint& checkpoint);

// Returns a new scene restored from `path` (nullptr on failure) and fills
// `checkpoint` with the render progress.
Scene* loadCheckpoint(const std::string& path, Checkpoint& checkpoint);
