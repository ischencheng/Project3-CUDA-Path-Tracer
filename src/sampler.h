#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include <cstdint>

// Stateless sample generation: every random number is a pure function of
// (pixel, sample index, dimension). This keeps PathSegment small, makes the
// result independent of the order in which paths are processed (so sorting and
// compaction do not change the image) and allows swapping in low-discrepancy
// sequences per dimension.

enum SamplerType
{
    SAMPLER_RANDOM = 0,     // hashed white noise
    SAMPLER_COUNT
};

// Dimension layout of one path sample.
enum SampleDimension
{
    DIM_PIXEL = 0,          // 2D sub-pixel position
    DIM_LENS = 2,           // 2D aperture position
    DIM_TIME = 4,           // 1D shutter time
    DIM_BOUNCE_BASE = 5,
};

// Offsets relative to DIM_BOUNCE_BASE + depth * DIMS_PER_BOUNCE.
enum BounceDimension
{
    BDIM_LOBE = 0,          // 1D BSDF lobe selection
    BDIM_BSDF = 1,          // 2D BSDF direction
    BDIM_LIGHT_SELECT = 3,  // 1D light selection
    BDIM_LIGHT = 4,         // 2D point on the light
    BDIM_RR = 6,            // 1D russian roulette
    BDIM_EXTRA = 7,         // 1D spare (medium sampling)
    DIMS_PER_BOUNCE = 8
};

__host__ __device__ inline uint32_t pcgHash(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__host__ __device__ inline uint32_t hashCombine(uint32_t seed, uint32_t v)
{
    return pcgHash(seed ^ (v + 0x9e3779b9u + (seed << 6) + (seed >> 2)));
}

// Uses the top 24 bits so that the result is strictly less than one.
__host__ __device__ inline float uintToUnitFloat(uint32_t x)
{
    return (x >> 8) * (1.0f / 16777216.0f);
}

struct SampleContext
{
    uint32_t pixel;
    uint32_t sampleIndex;
    int type;
};

__host__ __device__ inline SampleContext makeSampleContext(int pixel, int iteration, int type)
{
    SampleContext ctx;
    ctx.pixel = (uint32_t)pixel;
    ctx.sampleIndex = (uint32_t)(iteration - 1);
    ctx.type = type;
    return ctx;
}

__host__ __device__ inline int bounceDimension(int depth, int offset)
{
    return DIM_BOUNCE_BASE + depth * DIMS_PER_BOUNCE + offset;
}

__host__ __device__ inline float randomSample(const SampleContext& ctx, uint32_t dim)
{
    uint32_t h = hashCombine(hashCombine(pcgHash(ctx.pixel), ctx.sampleIndex), dim);
    return uintToUnitFloat(h);
}

__host__ __device__ inline float sample1D(const SampleContext& ctx, int dim)
{
    return randomSample(ctx, (uint32_t)dim);
}

__host__ __device__ inline glm::vec2 sample2D(const SampleContext& ctx, int dim)
{
    return glm::vec2(randomSample(ctx, (uint32_t)dim), randomSample(ctx, (uint32_t)dim + 1u));
}
