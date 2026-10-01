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
    SAMPLER_SOBOL,          // Owen-scrambled Sobol, shuffled per pixel and dimension pair
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

// ---------------------------------------------------------------------------
// Owen-scrambled Sobol (Burley 2020, "Practical Hash-based Owen Scrambling").
// Each 2D sample of a path uses the first two Sobol dimensions, which form a
// (0,2)-sequence. Decorrelation between pixels and between dimension pairs
// comes from shuffling the sample index and scrambling the point with hashes
// seeded by (pixel, dimension).

__host__ __device__ inline uint32_t reverseBits(uint32_t x)
{
#ifdef __CUDA_ARCH__
    return __brev(x);
#else
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
#endif
}

// Hash-based approximation of a random base-2 Owen permutation of the bits
// above each bit, applied in reversed bit order (Laine-Karras style).
__host__ __device__ inline uint32_t laineKarrasPermutation(uint32_t x, uint32_t seed)
{
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return x;
}

__host__ __device__ inline uint32_t nestedUniformScramble(uint32_t x, uint32_t seed)
{
    x = reverseBits(x);
    x = laineKarrasPermutation(x, seed);
    return reverseBits(x);
}

// First Sobol dimension: the van der Corput sequence in base 2.
__host__ __device__ inline uint32_t sobolDim0(uint32_t index)
{
    return reverseBits(index);
}

// Second Sobol dimension (primitive polynomial x + 1, all m_i = 1); its
// direction numbers follow v_k = v_{k-1} ^ (v_{k-1} >> 1).
__host__ __device__ inline uint32_t sobolDim1(uint32_t index)
{
    uint32_t result = 0;
    uint32_t v = 1u << 31;
    for (; index != 0; index >>= 1)
    {
        if (index & 1u)
        {
            result ^= v;
        }
        v ^= v >> 1;
    }
    return result;
}

__host__ __device__ inline glm::vec2 sobolSample2D(const SampleContext& ctx, uint32_t dim)
{
    uint32_t seed = hashCombine(pcgHash(ctx.pixel), dim);
    uint32_t index = nestedUniformScramble(ctx.sampleIndex, seed);
    uint32_t x = nestedUniformScramble(sobolDim0(index), hashCombine(seed, 0x68bc21ebu));
    uint32_t y = nestedUniformScramble(sobolDim1(index), hashCombine(seed, 0x02e5be93u));
    return glm::vec2(uintToUnitFloat(x), uintToUnitFloat(y));
}

__host__ __device__ inline float sobolSample1D(const SampleContext& ctx, uint32_t dim)
{
    uint32_t seed = hashCombine(pcgHash(ctx.pixel), dim);
    uint32_t index = nestedUniformScramble(ctx.sampleIndex, seed);
    return uintToUnitFloat(nestedUniformScramble(sobolDim0(index), hashCombine(seed, 0x68bc21ebu)));
}

__host__ __device__ inline float sample1D(const SampleContext& ctx, int dim)
{
    if (ctx.type == SAMPLER_SOBOL)
    {
        return sobolSample1D(ctx, (uint32_t)dim);
    }
    return randomSample(ctx, (uint32_t)dim);
}

__host__ __device__ inline glm::vec2 sample2D(const SampleContext& ctx, int dim)
{
    if (ctx.type == SAMPLER_SOBOL)
    {
        return sobolSample2D(ctx, (uint32_t)dim);
    }
    return glm::vec2(randomSample(ctx, (uint32_t)dim), randomSample(ctx, (uint32_t)dim + 1u));
}
