#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "postprocess.h"
#include "sampler.h"
#include "mathUtils.h"

// Synchronizing after every kernel makes errors easy to attribute but
// serializes the CPU and the GPU, so only do it in debug builds.
#ifdef NDEBUG
#define ERRORCHECK_SYNC 0
#else
#define ERRORCHECK_SYNC 1
#endif

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK_SYNC
    cudaDeviceSynchronize();
#endif
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image,
    RenderSettings settings)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = displayTransform(image[index] / (float)iter, settings);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
// Second copies of the path/intersection buffers: compaction and sorting
// write into these and the pointers are swapped afterwards.
static PathSegment* dev_pathsAlt = NULL;
static ShadeableIntersection* dev_intersectionsAlt = NULL;
static int* dev_sortKeys = NULL;
static int* dev_sortKeysAlt = NULL;
static int* dev_sortIndices = NULL;
static int* dev_sortIndicesAlt = NULL;
static int* dev_numSelected = NULL;
static void* dev_cubScratch = NULL;
static size_t cubScratchBytes = 0;
static int sortKeyBits = 1;

static cudaEvent_t iterStartEvent = NULL;
static cudaEvent_t iterStopEvent = NULL;
static cudaEvent_t stageStartEvent = NULL;
static cudaEvent_t stageStopEvent = NULL;

struct IsPathAlive
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces > 0;
    }
};

struct IsPathDead
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces <= 0;
    }
};

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_pathsAlt, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersectionsAlt, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_sortKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortKeysAlt, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortIndices, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortIndicesAlt, pixelcount * sizeof(int));
    cudaMalloc(&dev_numSelected, sizeof(int));

    // Only as many key bits as needed to represent every material id plus the
    // "miss" key are sorted.
    int keyRange = (int)scene->materials.size() + 1;
    sortKeyBits = 1;
    while ((1 << sortKeyBits) < keyRange)
    {
        sortKeyBits++;
    }

    // Size the CUB scratch buffer once for the largest problem so that no
    // allocation happens while rendering.
    size_t selectBytes = 0;
    size_t sortBytes = 0;
    cub::DeviceSelect::If(NULL, selectBytes, dev_paths, dev_pathsAlt, dev_numSelected, pixelcount, IsPathAlive());
    cub::DeviceRadixSort::SortPairs(NULL, sortBytes, dev_sortKeys, dev_sortKeysAlt,
        dev_sortIndices, dev_sortIndicesAlt, pixelcount, 0, sortKeyBits);
    cubScratchBytes = std::max(selectBytes, sortBytes);
    cudaMalloc(&dev_cubScratch, cubScratchBytes);

    cudaEventCreate(&iterStartEvent);
    cudaEventCreate(&iterStopEvent);
    cudaEventCreate(&stageStartEvent);
    cudaEventCreate(&stageStopEvent);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_pathsAlt);
    cudaFree(dev_intersectionsAlt);
    cudaFree(dev_sortKeys);
    cudaFree(dev_sortKeysAlt);
    cudaFree(dev_sortIndices);
    cudaFree(dev_sortIndicesAlt);
    cudaFree(dev_numSelected);
    cudaFree(dev_cubScratch);
    dev_image = NULL;
    dev_paths = dev_pathsAlt = NULL;
    dev_geoms = NULL;
    dev_materials = NULL;
    dev_intersections = dev_intersectionsAlt = NULL;
    dev_sortKeys = dev_sortKeysAlt = dev_sortIndices = dev_sortIndicesAlt = NULL;
    dev_numSelected = NULL;
    dev_cubScratch = NULL;

    if (iterStartEvent)
    {
        cudaEventDestroy(iterStartEvent);
        cudaEventDestroy(iterStopEvent);
        cudaEventDestroy(stageStartEvent);
        cudaEventDestroy(stageStopEvent);
        iterStartEvent = iterStopEvent = stageStartEvent = stageStopEvent = NULL;
    }

    checkCUDAError("pathtraceFree");
}

void pathtraceResetImage()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    if (guiData != NULL)
    {
        guiData->stats.reset();
    }
}

void pathtraceCopyImageToHost()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments,
    RenderSettings settings)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment segment;
        SampleContext sampler = makeSampleContext(index, iter, settings.samplerType);

        segment.ray.origin = cam.position;
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        // Stochastic sampled antialiasing: every iteration shoots the ray
        // through a uniformly random point of the pixel footprint, so the
        // running average integrates the pixel box filter.
        glm::vec2 jitter(0.0f);
        if (settings.antialiasing)
        {
            jitter = sample2D(sampler, DIM_PIXEL) - glm::vec2(0.5f);
        }

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitter.x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitter.y - (float)cam.resolution.y * 0.5f)
        );

        // Thin lens depth of field: every ray through this pixel converges on
        // the same point of the focal plane, but starts from a random point
        // of the aperture disk.
        if (cam.lensRadius > 0.0f)
        {
            glm::vec2 lens = cam.lensRadius * concentricSampleDisk(sample2D(sampler, DIM_LENS));
            float ft = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 focusPoint = cam.position + ft * segment.ray.direction;
            segment.ray.origin = cam.position + cam.right * lens.x + cam.up * lens.y;
            segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.mediumMaterial = -1;
        segment.lastPdf = 0.0f;
        segment.flags = PATH_FLAG_DELTA_BOUNCE;     // camera rays behave like a delta bounce
        pathSegments[index] = segment;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        const PathSegment& pathSegment = pathSegments[path_index];
        ShadeableIntersection isect;
        if (pathSegment.remainingBounces <= 0)
        {
            // Only reachable when stream compaction is disabled.
            isect.t = -1.0f;
            isect.materialId = -1;
            isect.geomId = -1;
        }
        else
        {
            // naive parse through global geoms
            intersectScene(geoms, geoms_size, pathSegment.ray, isect);
        }
        intersections[path_index] = isect;
    }
}

// Sort key used to make paths that hit the same material contiguous in
// memory. Misses (and dead paths) are moved to the end of the array.
__global__ void computeMaterialKeys(int num_paths, const ShadeableIntersection* intersections,
    int missKey, int* keys, int* indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        const ShadeableIntersection& isect = intersections[idx];
        keys[idx] = isect.t > 0.0f ? isect.materialId : missKey;
        if (indices != NULL)
        {
            indices[idx] = idx;
        }
    }
}

// Applies the permutation produced by the key sort.
__global__ void gatherSortedPaths(int num_paths, const int* order,
    const PathSegment* pathsIn, const ShadeableIntersection* isectsIn,
    PathSegment* pathsOut, ShadeableIntersection* isectsOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        int src = order[idx];
        pathsOut[idx] = pathsIn[src];
        isectsOut[idx] = isectsIn[src];
    }
}

__device__ inline void addToImage(glm::vec3* image, int pixel, glm::vec3 radiance)
{
    // A single NaN would poison the running average forever.
    if (isfinite(radiance.x) && isfinite(radiance.y) && isfinite(radiance.z))
    {
        // Each pixel owns exactly one path per iteration, so no atomics are
        // needed as long as one kernel adds at most once per path.
        image[pixel] += radiance;
    }
}

// Everything the shading kernel needs besides the per-path buffers.
struct ShadeParams
{
    int iter;
    int depth;
    RenderSettings settings;
    glm::vec3 background;
    const Geom* geoms;
    const Material* materials;
    glm::vec3* image;
};

// Shades one bounce: accumulates emission into the image, evaluates the BSDF
// and spawns the continuation ray. Paths that miss the scene, hit a light, or
// run out of bounces are marked as terminated (remainingBounces = 0) so that
// they can be stream compacted away.
__global__ void shadeMaterial(
    ShadeParams params,
    int num_paths,
    const int* order,   // optional permutation (material-sorted indices)
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }
    if (order != NULL)
    {
        idx = order[idx];
    }

    PathSegment path = pathSegments[idx];
    if (path.remainingBounces <= 0)
    {
        return;
    }

    const int depth = params.depth;
    ShadeableIntersection intersection = shadeableIntersections[idx];

    // Beer-Lambert absorption along the segment travelled inside a medium.
    if (path.mediumMaterial >= 0 && intersection.t > 0.0f)
    {
        glm::vec3 sigma = params.materials[path.mediumMaterial].absorption;
        path.throughput *= glm::exp(-sigma * intersection.t);
    }

    if (intersection.t <= 0.0f)
    {
        // The ray escaped the scene.
        addToImage(params.image, path.pixelIndex, path.throughput * params.background);
        path.remainingBounces = 0;
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    const Material material = params.materials[intersection.materialId];
    SurfaceHit hit = computeSurfaceHit(params.geoms, path.ray, intersection);

    if (material.emission.x > 0.0f || material.emission.y > 0.0f || material.emission.z > 0.0f)
    {
        addToImage(params.image, path.pixelIndex, path.throughput * material.emission);
    }
    if (material.type == MATERIAL_EMITTING)
    {
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    SampleContext sampler = makeSampleContext(path.pixelIndex, params.iter, params.settings.samplerType);
    BSDFParams bsdf;
    bsdf.type = material.type;
    bsdf.color = material.color;
    bsdf.alpha = roughnessToAlpha(material.roughness);
    bsdf.metallic = material.metallic;
    bsdf.etap = hit.frontFace ? material.ior : 1.0f / material.ior;
    Frame frame = makeFrame(hit.normal);
    glm::vec3 wo = -path.ray.direction;

    BSDFSample bs;
    float uLobe = sample1D(sampler, bounceDimension(depth, BDIM_LOBE));
    glm::vec2 uDir = sample2D(sampler, bounceDimension(depth, BDIM_BSDF));
    bool scattered = sampleBSDF(bsdf, frame, wo, uLobe, uDir, bs)
        && maxComponent(bs.weight) > 0.0f && isFiniteVec(bs.weight);

    path.remainingBounces--;
    if (!scattered)
    {
        path.remainingBounces = 0;
    }
    else
    {
        path.throughput *= bs.weight;
        path.lastPdf = bs.pdf;
        path.flags = bs.isDelta ? PATH_FLAG_DELTA_BOUNCE : 0;
        if (bs.isTransmission)
        {
            path.mediumMaterial = hit.frontFace ? intersection.materialId : -1;
        }
        // Offset along the geometric normal to the side the new ray leaves on.
        bool leavesFront = glm::dot(bs.wi, hit.normal) > 0.0f;
        path.ray.origin = hit.position + (leavesFront ? hit.normal : -hit.normal) * RAY_EPSILON;
        path.ray.direction = bs.wi;

        // Russian roulette: once the path is a few bounces deep, terminate it
        // with probability 1 - max(throughput) and reweight the survivors so
        // the estimator stays unbiased.
        if (params.settings.russianRoulette && depth + 1 >= params.settings.rrStartDepth
            && path.remainingBounces > 0)
        {
            float survive = maxComponent(path.throughput);
            if (survive < 1.0f)
            {
                if (sample1D(sampler, bounceDimension(depth, BDIM_RR)) >= survive)
                {
                    path.remainingBounces = 0;
                }
                else
                {
                    path.throughput /= survive;
                }
            }
        }
    }

    pathSegments[idx] = path;
}

static inline void stageBegin(bool profile)
{
    if (profile)
    {
        cudaEventRecord(stageStartEvent);
    }
}

static inline void stageEnd(bool profile, RenderStage stage)
{
    if (profile)
    {
        cudaEventRecord(stageStopEvent);
        cudaEventSynchronize(stageStopEvent);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, stageStartEvent, stageStopEvent);
        if (guiData->stats.iterationsSeen >= guiData->stats.warmupIterations)
        {
            guiData->stats.stageMs[stage] += ms;
        }
    }
}

// Reorders paths (and their intersections) so that equal materials are
// contiguous before shading. Returns the permutation the shading kernel must
// read through, or NULL if the data itself was reordered.
static const int* sortPathsByMaterial(int mode, int num_paths, int blockSize)
{
    const int blocks = (num_paths + blockSize - 1) / blockSize;
    const int missKey = (int)hst_scene->materials.size();
    if (mode == SORT_THRUST)
    {
        computeMaterialKeys<<<blocks, blockSize>>>(num_paths, dev_intersections, missKey, dev_sortKeys, NULL);
        thrust::sort_by_key(thrust::device, dev_sortKeys, dev_sortKeys + num_paths,
            thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections)));
    }
    else
    {
        computeMaterialKeys<<<blocks, blockSize>>>(num_paths, dev_intersections, missKey,
            dev_sortKeys, dev_sortIndices);
        size_t bytes = cubScratchBytes;
        cub::DeviceRadixSort::SortPairs(dev_cubScratch, bytes, dev_sortKeys, dev_sortKeysAlt,
            dev_sortIndices, dev_sortIndicesAlt, num_paths, 0, sortKeyBits);
        if (mode == SORT_CUB_INDIRECT)
        {
            checkCUDAError("sort by material");
            return dev_sortIndicesAlt;
        }
        gatherSortedPaths<<<blocks, blockSize>>>(num_paths, dev_sortIndicesAlt,
            dev_paths, dev_intersections, dev_pathsAlt, dev_intersectionsAlt);
        std::swap(dev_paths, dev_pathsAlt);
        std::swap(dev_intersections, dev_intersectionsAlt);
    }
    checkCUDAError("sort by material");
    return NULL;
}

// Removes terminated paths from the active range and returns the new count.
static int compactPaths(int mode, int num_paths)
{
    if (mode == COMPACT_THRUST)
    {
        PathSegment* aliveEnd = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, IsPathDead());
        return (int)(aliveEnd - dev_paths);
    }

    size_t bytes = cubScratchBytes;
    cub::DeviceSelect::If(dev_cubScratch, bytes, dev_paths, dev_pathsAlt, dev_numSelected, num_paths, IsPathAlive());
    std::swap(dev_paths, dev_pathsAlt);
    int alive = 0;
    cudaMemcpy(&alive, dev_numSelected, sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("compact paths");
    return alive;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const RenderSettings settings = guiData->settings;
    RenderStats& stats = guiData->stats;
    const bool profile = stats.profileStages;
    const bool recordStats = stats.iterationsSeen >= stats.warmupIterations;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    cudaEventRecord(iterStartEvent);

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.
    //
    // Terminated paths add their radiance to the image directly from the
    // shading kernel, so compaction can simply drop them and no final gather
    // pass over all pixels is needed.

    // TODO: perform one iteration of path tracing

    stageBegin(profile);
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, settings);
    checkCUDAError("generate camera ray");
    stageEnd(profile, STAGE_GENERATE);

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        if (profile && recordStats && depth < MAX_TRACKED_DEPTH)
        {
            // Without compaction the active range still contains dead paths,
            // so count the live ones explicitly for the statistics.
            int alive = settings.compactionMode != COMPACT_OFF ? num_paths
                : (int)thrust::count_if(thrust::device, dev_paths, dev_paths + num_paths, IsPathAlive());
            stats.alivePaths[depth] += alive;
        }

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        stageBegin(profile);
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        stageEnd(profile, STAGE_INTERSECT);

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        const int* shadeOrder = NULL;
        if (settings.sortMode != SORT_OFF)
        {
            stageBegin(profile);
            shadeOrder = sortPathsByMaterial(settings.sortMode, num_paths, blockSize1d);
            stageEnd(profile, STAGE_SORT);
        }

        stageBegin(profile);
        ShadeParams shadeParams;
        shadeParams.iter = iter;
        shadeParams.depth = depth;
        shadeParams.settings = settings;
        shadeParams.background = hst_scene->state.backgroundColor;
        shadeParams.geoms = dev_geoms;
        shadeParams.materials = dev_materials;
        shadeParams.image = dev_image;
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            shadeParams,
            num_paths,
            shadeOrder,
            dev_intersections,
            dev_paths
        );
        checkCUDAError("shade");
        stageEnd(profile, STAGE_SHADE);
        depth++;

        if (settings.compactionMode != COMPACT_OFF)
        {
            stageBegin(profile);
            num_paths = compactPaths(settings.compactionMode, num_paths);
            stageEnd(profile, STAGE_COMPACT);
        }

        iterationComplete = num_paths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    cudaEventRecord(iterStopEvent);
    cudaEventSynchronize(iterStopEvent);
    float iterationMs = 0.0f;
    cudaEventElapsedTime(&iterationMs, iterStartEvent, iterStopEvent);
    stats.lastIterationMs = iterationMs;
    if (recordStats)
    {
        stats.totalIterationMs += iterationMs;
        stats.sampledIterations++;
        stats.numBounces = std::max(stats.numBounces, depth);
    }
    stats.iterationsSeen++;

    // Send results to OpenGL buffer for rendering
    if (pbo != NULL)
    {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image, settings);
    }

    checkCUDAError("pathtrace");
}
