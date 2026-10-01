#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "utilities.h"

// Narkowicz's fitted ACES filmic curve.
__host__ __device__ inline glm::vec3 acesFilm(glm::vec3 x)
{
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Maps an averaged linear HDR radiance value to a displayable [0, 1] color.
// Shared by the OpenGL preview kernel and by the image writer so that saved
// PNGs look exactly like the preview window.
__host__ __device__ inline glm::vec3 displayTransform(glm::vec3 c, const RenderSettings& s)
{
    c *= s.exposure;
    if (s.toneMap == TONEMAP_REINHARD)
    {
        c = c / (glm::vec3(1.0f) + c);
    }
    else if (s.toneMap == TONEMAP_ACES)
    {
        c = acesFilm(c);
    }
    c = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
    if (s.gammaCorrect)
    {
        c = glm::pow(c, glm::vec3(1.0f / 2.2f));
    }
    return c;
}
