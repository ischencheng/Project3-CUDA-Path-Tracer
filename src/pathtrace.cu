#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/random.h>
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

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
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
static int* dev_materialKeys = NULL;     // sort keys for material sorting
static cudaEvent_t iterStartEvent = NULL;
static cudaEvent_t iterStopEvent = NULL;
static cudaEvent_t stageStartEvent = NULL;
static cudaEvent_t stageStopEvent = NULL;

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
    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));

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
    cudaFree(dev_materialKeys);
    dev_image = NULL;
    dev_paths = NULL;
    dev_geoms = NULL;
    dev_materials = NULL;
    dev_intersections = NULL;
    dev_materialKeys = NULL;

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
    bool antialiasing)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.radiance = glm::vec3(0.0f);

        // TODO: implement antialiasing by jittering the ray
        // Stochastic sampled antialiasing: every iteration shoots the ray
        // through a uniformly random point of the pixel footprint, so the
        // running average integrates the pixel box filter.
        float jitterX = 0.0f;
        float jitterY = 0.0f;
        if (antialiasing)
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, -1);
            thrust::uniform_real_distribution<float> u01(0, 1);
            jitterX = u01(rng) - 0.5f;
            jitterY = u01(rng) - 0.5f;
        }

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
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
        PathSegment pathSegment = pathSegments[path_index];
        if (pathSegment.remainingBounces <= 0)
        {
            // Only reachable when stream compaction is disabled.
            intersections[path_index].t = -1.0f;
            return;
        }

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// Sort key used to make paths that hit the same material contiguous in
// memory. Misses are moved to the end of the array.
__global__ void computeMaterialKeys(int num_paths, const ShadeableIntersection* intersections, int* keys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        const ShadeableIntersection& isect = intersections[idx];
        keys[idx] = isect.t > 0.0f ? isect.materialId : INT_MAX;
    }
}

// Shades one bounce: accumulates emission, evaluates the BSDF and spawns the
// continuation ray. Paths that miss the scene, hit a light, or run out of
// bounces are marked as terminated (remainingBounces = 0) so that they can be
// stream compacted away.
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment path = pathSegments[idx];
    if (path.remainingBounces <= 0)
    {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t <= 0.0f)
    {
        // The ray escaped the scene.
        path.radiance += path.throughput * BACKGROUND_COLOR;
        path.remainingBounces = 0;
    }
    else
    {
        Material material = materials[intersection.materialId];
        if (material.type == MATERIAL_EMITTING)
        {
            path.radiance += path.throughput * material.color * material.emittance;
            path.remainingBounces = 0;
        }
        else
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, depth);
            glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;
            scatterRay(path, hitPoint, intersection.surfaceNormal, material, rng);
        }
    }

    pathSegments[idx] = path;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        glm::vec3 radiance = iterationPath.radiance;
        // A single NaN would poison the running average forever.
        if (!(isfinite(radiance.x) && isfinite(radiance.y) && isfinite(radiance.z)))
        {
            return;
        }
        image[iterationPath.pixelIndex] += radiance;
    }
}

struct IsPathAlive
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces > 0;
    }
};

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

    // TODO: perform one iteration of path tracing

    stageBegin(profile);
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths,
        settings.antialiasing);
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
            int alive = settings.streamCompaction ? num_paths
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

        if (settings.sortByMaterial)
        {
            stageBegin(profile);
            computeMaterialKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths, dev_intersections, dev_materialKeys);
            thrust::sort_by_key(thrust::device, dev_materialKeys, dev_materialKeys + num_paths,
                thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections)));
            checkCUDAError("sort by material");
            stageEnd(profile, STAGE_SORT);
        }

        stageBegin(profile);
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        checkCUDAError("shade");
        stageEnd(profile, STAGE_SHADE);
        depth++;

        if (settings.streamCompaction)
        {
            // Partition instead of remove so that terminated paths (and their
            // gathered radiance) stay in the buffer for the final gather.
            stageBegin(profile);
            PathSegment* aliveEnd = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, IsPathAlive());
            num_paths = aliveEnd - dev_paths;
            stageEnd(profile, STAGE_COMPACT);
        }

        iterationComplete = num_paths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    stageBegin(profile);
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);
    checkCUDAError("final gather");
    stageEnd(profile, STAGE_GATHER);

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
