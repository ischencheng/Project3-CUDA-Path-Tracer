#include "denoise.h"

#include <chrono>
#include <cstdio>

#ifdef USE_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

bool denoiseImage(
    const std::vector<glm::vec3>& color,
    const std::vector<glm::vec3>& albedo,
    const std::vector<glm::vec3>& normal,
    int width,
    int height,
    std::vector<glm::vec3>& output)
{
#ifdef USE_OIDN
    // ask for the cuda device by name, the default pick also tries the intel gpu and fails
    static oidn::DeviceRef device;
    const char* message;
    if (!device)
    {
        device = oidn::newDevice(oidn::DeviceType::CUDA);
        if (!device)
        {
            device = oidn::newDevice(oidn::DeviceType::CPU);
        }
        device.commit();
        if (device.getError(message) != oidn::Error::None)
        {
            printf("OIDN error: %s\n", message);
            device = nullptr;
            return false;
        }
    }

    size_t bytes = width * height * sizeof(glm::vec3);
    oidn::BufferRef colorBuffer = device.newBuffer(bytes);
    oidn::BufferRef albedoBuffer = device.newBuffer(bytes);
    oidn::BufferRef normalBuffer = device.newBuffer(bytes);
    oidn::BufferRef outputBuffer = device.newBuffer(bytes);
    colorBuffer.write(0, bytes, color.data());
    albedoBuffer.write(0, bytes, albedo.data());
    normalBuffer.write(0, bytes, normal.data());

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", colorBuffer, oidn::Format::Float3, width, height);
    filter.setImage("albedo", albedoBuffer, oidn::Format::Float3, width, height);
    filter.setImage("normal", normalBuffer, oidn::Format::Float3, width, height);
    filter.setImage("output", outputBuffer, oidn::Format::Float3, width, height);
    // the image can go above 1 near the lights
    filter.set("hdr", true);
    filter.commit();

    auto start = std::chrono::high_resolution_clock::now();
    filter.execute();
    auto end = std::chrono::high_resolution_clock::now();

    if (device.getError(message) != oidn::Error::None)
    {
        printf("OIDN error: %s\n", message);
        return false;
    }
    printf("OIDN filter took %.1f ms\n", std::chrono::duration<double, std::milli>(end - start).count());

    output.resize(width * height);
    outputBuffer.read(0, bytes, output.data());
    return true;
#else
    printf("Built without Open Image Denoise, see README\n");
    return false;
#endif
}
