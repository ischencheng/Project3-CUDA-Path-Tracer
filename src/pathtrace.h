#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
// Clears the accumulated image (e.g. after the camera moved) without
// re-uploading the scene.
void pathtraceResetImage();
// Copies the accumulated (unnormalized) radiance into scene->state.image.
void pathtraceCopyImageToHost();
// Renders one sample per pixel. `pbo` may be NULL when running headless.
void pathtrace(uchar4 *pbo, int frame, int iteration);

// Denoises the current accumulation (average of `iteration` samples).
// Returns false if denoising is unavailable.
bool pathtraceDenoise(int iteration);
// Copies the last denoised image into `out` (averaged radiance).
void pathtraceCopyDenoisedToHost(std::vector<glm::vec3>& out);
// Copies the averaged albedo and normal features into the given vectors.
void pathtraceCopyFeaturesToHost(int iteration, std::vector<glm::vec3>& albedo, std::vector<glm::vec3>& normal);
const char* pathtraceDenoiserName();

// Accumulation buffers (sums over iterations) for checkpoints.
void pathtraceGetAccumulation(std::vector<glm::vec3>& image, std::vector<glm::vec3>& albedo,
    std::vector<glm::vec3>& normal);
void pathtraceSetAccumulation(const std::vector<glm::vec3>& image, const std::vector<glm::vec3>& albedo,
    const std::vector<glm::vec3>& normal);
