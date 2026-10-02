#pragma once

#include <string>

// Thin wrapper around Intel Open Image Denoise. All image pointers are CUDA
// device pointers to width * height float3 pixels. When the project is built
// without OIDN (USE_OIDN undefined) every call reports failure.

bool denoiserAvailable();

// Creates the OIDN device and filters bound to the given device buffers.
// `albedo`/`normal` are the auxiliary feature images; `output` receives the
// denoised color. Returns false if no device could be created.
bool denoiserInit(int width, int height, float* color, float* albedo, float* normal, float* output);
void denoiserFree();

// Runs the filter. With `useAux` the albedo and normal features guide the
// denoiser; with `prefilterAux` those features are denoised first so the
// final pass can treat them as noise free.
bool denoiserRun(bool useAux, bool prefilterAux, bool highQuality);

// Builds and commits the filters for these settings ahead of time (loading
// weights and compiling kernels takes far longer than one execution).
void denoiserPrepare(bool useAux, bool prefilterAux, bool highQuality);

std::string denoiserDeviceName();
