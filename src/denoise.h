#pragma once

#include <glm/glm.hpp>

#include <vector>

// open image denoise with albedo and normal guides, false if it was not built in
bool denoiseImage(
    const std::vector<glm::vec3>& color,
    const std::vector<glm::vec3>& albedo,
    const std::vector<glm::vec3>& normal,
    int width,
    int height,
    std::vector<glm::vec3>& output);
