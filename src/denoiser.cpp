#include "denoiser.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#ifdef USE_OIDN

#include <OpenImageDenoise/oidn.h>

namespace
{

struct DenoiserState
{
    OIDNDevice device = nullptr;
    bool sharedMemory = false;      // OIDN runs on our CUDA buffers directly
    int width = 0;
    int height = 0;
    float* devColor = nullptr;
    float* devAlbedo = nullptr;
    float* devNormal = nullptr;
    float* devOutput = nullptr;
    OIDNBuffer color = nullptr;
    OIDNBuffer albedo = nullptr;
    OIDNBuffer normal = nullptr;
    OIDNBuffer output = nullptr;
    // Committed filters are cached: committing loads the network weights and
    // (on GPU devices) builds kernels, which costs far more than executing.
    OIDNFilter beautyFilter = nullptr;
    OIDNFilter albedoFilter = nullptr;
    OIDNFilter normalFilter = nullptr;
    int filterConfig = -1;
    std::string deviceName = "none";
};


DenoiserState state;

void releaseFilters()
{
    if (state.beautyFilter) oidnReleaseFilter(state.beautyFilter);
    if (state.albedoFilter) oidnReleaseFilter(state.albedoFilter);
    if (state.normalFilter) oidnReleaseFilter(state.normalFilter);
    state.beautyFilter = state.albedoFilter = state.normalFilter = nullptr;
    state.filterConfig = -1;
}

bool checkError(const char* what)
{
    const char* message = nullptr;
    if (oidnGetDeviceError(state.device, &message) != OIDN_ERROR_NONE)
    {
        fprintf(stderr, "OIDN error (%s): %s\n", what, message ? message : "unknown");
        return false;
    }
    return true;
}

size_t imageBytes()
{
    return (size_t)state.width * state.height * 3 * sizeof(float);
}

OIDNBuffer makeBuffer(float* devPtr)
{
    if (state.sharedMemory)
    {
        return oidnNewSharedBuffer(state.device, devPtr, imageBytes());
    }
    return oidnNewBuffer(state.device, imageBytes());
}

// Copies a CUDA buffer into an OIDN host buffer (CPU fallback only).
void upload(OIDNBuffer buffer, const float* devPtr)
{
    if (!state.sharedMemory)
    {
        cudaMemcpy(oidnGetBufferData(buffer), devPtr, imageBytes(), cudaMemcpyDeviceToHost);
    }
}

OIDNFilter makeFilter(OIDNBuffer colorInput, OIDNBuffer albedoInput, OIDNBuffer normalInput,
    OIDNBuffer outputBuffer, bool hdr, bool cleanAux, bool highQuality)
{
    OIDNFilter filter = oidnNewFilter(state.device, "RT");
    const int w = state.width;
    const int h = state.height;
    if (colorInput)
    {
        oidnSetFilterImage(filter, "color", colorInput, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0);
    }
    if (albedoInput)
    {
        oidnSetFilterImage(filter, "albedo", albedoInput, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0);
    }
    if (normalInput)
    {
        oidnSetFilterImage(filter, "normal", normalInput, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0);
    }
    oidnSetFilterImage(filter, "output", outputBuffer, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0);
    if (colorInput)
    {
        oidnSetFilterBool(filter, "hdr", hdr);
        oidnSetFilterBool(filter, "cleanAux", cleanAux);
    }
    oidnSetFilterInt(filter, "quality", highQuality ? OIDN_QUALITY_HIGH : OIDN_QUALITY_BALANCED);
    oidnCommitFilter(filter);
    return filter;
}

} // namespace

bool denoiserAvailable()
{
    return true;
}

bool denoiserInit(int width, int height, float* color, float* albedo, float* normal, float* output)
{
    denoiserFree();
    state.width = width;
    state.height = height;
    state.devColor = color;
    state.devAlbedo = albedo;
    state.devNormal = normal;
    state.devOutput = output;

    // Prefer a CUDA device on our default stream so that OIDN reads the
    // accumulation buffers in place; fall back to the CPU device.
    int cudaDevice = 0;
    cudaGetDevice(&cudaDevice);
    if (oidnIsCUDADeviceSupported(cudaDevice))
    {
        cudaStream_t stream = 0;
        state.device = oidnNewCUDADevice(&cudaDevice, &stream, 1);
        oidnCommitDevice(state.device);
        if (checkError("CUDA device"))
        {
            state.sharedMemory = true;
            state.deviceName = "CUDA";
        }
        else
        {
            oidnReleaseDevice(state.device);
            state.device = nullptr;
        }
    }
    if (!state.device)
    {
        state.device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
        oidnCommitDevice(state.device);
        if (!checkError("CPU device"))
        {
            oidnReleaseDevice(state.device);
            state.device = nullptr;
            return false;
        }
        state.sharedMemory = false;
        state.deviceName = "CPU";
    }

    state.color = makeBuffer(color);
    state.albedo = makeBuffer(albedo);
    state.normal = makeBuffer(normal);
    state.output = makeBuffer(output);
    return checkError("buffers");
}

void denoiserFree()
{
    if (!state.device)
    {
        return;
    }
    releaseFilters();
    oidnReleaseBuffer(state.color);
    oidnReleaseBuffer(state.albedo);
    oidnReleaseBuffer(state.normal);
    oidnReleaseBuffer(state.output);
    oidnReleaseDevice(state.device);
    state = DenoiserState();
}

static void prepareFilters(bool useAux, bool prefilterAux, bool highQuality)
{
    int config = (useAux ? 1 : 0) | (prefilterAux ? 2 : 0) | (highQuality ? 4 : 0);
    if (config == state.filterConfig)
    {
        return;
    }
    releaseFilters();
    if (useAux && prefilterAux)
    {
        // denoise the guides in place, then treat them as noise free
        state.albedoFilter = makeFilter(nullptr, state.albedo, nullptr, state.albedo, false, false, highQuality);
        state.normalFilter = makeFilter(nullptr, nullptr, state.normal, state.normal, false, false, highQuality);
    }
    state.beautyFilter = makeFilter(state.color, useAux ? state.albedo : nullptr, useAux ? state.normal : nullptr,
        state.output, true, useAux && prefilterAux, highQuality);
    state.filterConfig = config;
}

void denoiserPrepare(bool useAux, bool prefilterAux, bool highQuality)
{
    if (state.device)
    {
        prepareFilters(useAux, prefilterAux, highQuality);
        checkError("prepare");
    }
}

bool denoiserRun(bool useAux, bool prefilterAux, bool highQuality)
{
    if (!state.device)
    {
        return false;
    }
    upload(state.color, state.devColor);
    if (useAux)
    {
        upload(state.albedo, state.devAlbedo);
        upload(state.normal, state.devNormal);
    }

    prepareFilters(useAux, prefilterAux, highQuality);
    if (state.albedoFilter)
    {
        oidnExecuteFilter(state.albedoFilter);
        oidnExecuteFilter(state.normalFilter);
    }
    oidnExecuteFilter(state.beautyFilter);

    if (!state.sharedMemory)
    {
        cudaMemcpy(state.devOutput, oidnGetBufferData(state.output), imageBytes(), cudaMemcpyHostToDevice);
    }
    return checkError("execute");
}

std::string denoiserDeviceName()
{
    return state.deviceName;
}

#else // USE_OIDN

bool denoiserAvailable() { return false; }
bool denoiserInit(int, int, float*, float*, float*, float*) { return false; }
void denoiserFree() {}
bool denoiserRun(bool, bool, bool) { return false; }
void denoiserPrepare(bool, bool, bool) {}
std::string denoiserDeviceName() { return "unavailable (built without OIDN)"; }

#endif // USE_OIDN
