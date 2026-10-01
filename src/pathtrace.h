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
