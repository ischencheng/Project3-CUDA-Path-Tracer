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
#include "textures.h"
#include "lights.h"
#include "denoiser.h"

// Synchronizing after every kernel makes errors easy to attribute but
// serializes the CPU and the GPU, so only do it in debug builds.
#ifdef NDEBUG
#define ERRORCHECK_SYNC 0
#else
#define ERRORCHECK_SYNC 1
#endif

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

// Wavefront path tracing: queue index used for paths that missed the scene.
#define QUEUE_MISS MATERIAL_TYPE_COUNT
#define QUEUE_COUNT (MATERIAL_TYPE_COUNT + 1)
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
// `scale` converts the stored value into average radiance (1 / iterations
// for accumulation buffers, 1 for the denoised image). Feature buffers are
// shown without tone mapping.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, float scale, const glm::vec3* image,
    RenderSettings settings, bool rawDisplay)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] * scale;
        if (rawDisplay)
        {
            pix = glm::clamp(pix, glm::vec3(0.0f), glm::vec3(1.0f));
        }
        else
        {
            pix = displayTransform(pix, settings);
        }

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

__global__ void showNormals(uchar4* pbo, glm::ivec2 resolution, float scale, const glm::vec3* normals)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 c = glm::clamp(normals[index] * scale * 0.5f + glm::vec3(0.5f), glm::vec3(0.0f), glm::vec3(1.0f));
        pbo[index] = make_uchar4((unsigned char)(c.x * 255.0f), (unsigned char)(c.y * 255.0f), (unsigned char)(c.z * 255.0f), 0);
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
static int* dev_aliveCounter = NULL;
// Wavefront material queues
static int* dev_queues = NULL;
static int* dev_queueCounts = NULL;
// Triangle meshes and their BVHs
static MeshInfo* dev_meshes = NULL;
static BVHNode* dev_bvhNodes = NULL;
static TriangleGeom* dev_triGeoms = NULL;
static Triangle* dev_triangles = NULL;
static glm::vec3* dev_vertexNormals = NULL;
static glm::vec2* dev_vertexUVs = NULL;
static glm::vec4* dev_vertexTangents = NULL;
// Textures: one CUDA array + texture object per scene texture
static std::vector<cudaArray_t> textureArrays;
static std::vector<cudaTextureObject_t> textureObjects;
static cudaTextureObject_t* dev_textures = NULL;
// Lights and environment
static Light* dev_lights = NULL;
static float* dev_lightCdf = NULL;
static int* dev_triLightIndex = NULL;
static ShadowRay* dev_shadowRays = NULL;
// Denoiser: accumulated first-hit features and the averaged inputs/output
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
static glm::vec3* dev_dnColor = NULL;
static glm::vec3* dev_dnAlbedo = NULL;
static glm::vec3* dev_dnNormal = NULL;
static glm::vec3* dev_denoised = NULL;
static bool denoiserReady = false;
static bool denoisedValid = false;
static cudaArray_t envArray = NULL;
static cudaTextureObject_t envTexture = 0;
static float* dev_envFunc = NULL;
static float* dev_envMarginal = NULL;
static float* dev_envConditional = NULL;
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

template <typename T>
static void uploadVector(T*& dev, const std::vector<T>& host)
{
    dev = NULL;
    if (!host.empty())
    {
        cudaMalloc(&dev, host.size() * sizeof(T));
        cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
}

// Uploads RGBA8 textures into CUDA arrays and creates bilinear, wrapping
// texture objects returning normalized floats.
static void uploadTextures(const std::vector<TextureData>& textures)
{
    for (const TextureData& tex : textures)
    {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t array;
        cudaMallocArray(&array, &desc, tex.width, tex.height);
        cudaMemcpy2DToArray(array, 0, 0, tex.rgba.data(), tex.width * 4, tex.width * 4, tex.height,
            cudaMemcpyHostToDevice);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
        cudaTextureObject_t object = 0;
        cudaCreateTextureObject(&object, &resDesc, &texDesc, NULL);

        textureArrays.push_back(array);
        textureObjects.push_back(object);
    }
    uploadVector(dev_textures, textureObjects);
}

static void uploadEnvironment(const EnvironmentMap& env)
{
    if (env.width <= 0)
    {
        return;
    }
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&envArray, &desc, env.width, env.height);
    cudaMemcpy2DToArray(envArray, 0, 0, env.rgba.data(), env.width * sizeof(float4), env.width * sizeof(float4),
        env.height, cudaMemcpyHostToDevice);
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;   // longitude wraps around
    texDesc.addressMode[1] = cudaAddressModeClamp;  // latitude stops at the poles
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    cudaCreateTextureObject(&envTexture, &resDesc, &texDesc, NULL);
    uploadVector(dev_envFunc, env.func);
    uploadVector(dev_envMarginal, env.marginalCdf);
    uploadVector(dev_envConditional, env.conditionalCdf);
}

static EnvironmentView makeEnvironmentView()
{
    const EnvironmentMap& env = hst_scene->envMap;
    EnvironmentView view = {};
    view.hasMap = env.width > 0 ? 1 : 0;
    view.color = hst_scene->state.backgroundColor;
    view.enabled = view.hasMap || view.color.x > 0.0f || view.color.y > 0.0f || view.color.z > 0.0f;
    view.intensity = env.intensity;
    view.rotation = env.rotation;
    view.texture = envTexture;
    view.width = env.width;
    view.height = env.height;
    view.func = dev_envFunc;
    view.marginalCdf = dev_envMarginal;
    view.conditionalCdf = dev_envConditional;
    view.integral = env.integral;
    return view;
}

static LightsView makeLightsView()
{
    LightsView view;
    view.lights = dev_lights;
    view.cdf = dev_lightCdf;
    view.count = (int)hst_scene->lights.size();
    view.envSelectProb = hst_scene->envSampleProb;
    return view;
}

static void freeTextures()
{
    for (cudaTextureObject_t t : textureObjects)
    {
        cudaDestroyTextureObject(t);
    }
    for (cudaArray_t a : textureArrays)
    {
        cudaFreeArray(a);
    }
    textureObjects.clear();
    textureArrays.clear();
    cudaFree(dev_textures);
    dev_textures = NULL;
    if (envTexture)
    {
        cudaDestroyTextureObject(envTexture);
        cudaFreeArray(envArray);
        envTexture = 0;
        envArray = NULL;
    }
    cudaFree(dev_envFunc);
    cudaFree(dev_envMarginal);
    cudaFree(dev_envConditional);
    dev_envFunc = dev_envMarginal = dev_envConditional = NULL;
}

static SceneView makeSceneView(const RenderSettings& settings)
{
    SceneView view;
    view.geoms = dev_geoms;
    view.geomCount = (int)hst_scene->geoms.size();
    view.materials = dev_materials;
    view.meshes = dev_meshes;
    view.bvhNodes = dev_bvhNodes;
    view.triGeoms = dev_triGeoms;
    view.triangles = dev_triangles;
    view.normals = dev_vertexNormals;
    view.uvs = dev_vertexUVs;
    view.tangents = dev_vertexTangents;
    view.textures = dev_textures;
    view.triLightIndex = dev_triLightIndex;
    view.lights = dev_lights;
    view.useBVH = settings.useBVH ? 1 : 0;
    view.cullBounds = settings.cullBounds ? 1 : 0;
    return view;
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
    uploadVector(dev_meshes, scene->meshes);
    uploadVector(dev_bvhNodes, scene->bvhNodes);
    uploadVector(dev_triGeoms, scene->triGeoms);
    uploadVector(dev_triangles, scene->triangles);
    uploadVector(dev_vertexNormals, scene->vertexNormals);
    uploadVector(dev_vertexUVs, scene->vertexUVs);
    uploadVector(dev_vertexTangents, scene->vertexTangents);
    uploadTextures(scene->textures);
    uploadEnvironment(scene->envMap);
    uploadVector(dev_lights, scene->lights);
    uploadVector(dev_lightCdf, scene->lightCdf);
    if (!scene->lights.empty())
    {
        uploadVector(dev_triLightIndex, scene->triLightIndex);
    }
    cudaMalloc(&dev_shadowRays, pixelcount * sizeof(ShadowRay));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_dnColor, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_dnAlbedo, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_dnNormal, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));
    denoiserReady = denoiserInit(cam.resolution.x, cam.resolution.y, (float*)dev_dnColor,
        (float*)dev_dnAlbedo, (float*)dev_dnNormal, (float*)dev_denoised);
    denoisedValid = false;
    if (denoiserReady)
    {
        printf("Open Image Denoise ready (%s device)\n", denoiserDeviceName().c_str());
        const RenderSettings& s = guiData->settings;
        if (s.denoise)
        {
            denoiserPrepare(s.denoiseAux, s.denoisePrefilter, s.denoiseHighQuality);
        }
    }

    cudaMalloc(&dev_pathsAlt, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersectionsAlt, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_sortKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortKeysAlt, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortIndices, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortIndicesAlt, pixelcount * sizeof(int));
    cudaMalloc(&dev_numSelected, sizeof(int));
    cudaMalloc(&dev_aliveCounter, sizeof(int));
    cudaMalloc(&dev_queues, (size_t)QUEUE_COUNT * pixelcount * sizeof(int));
    cudaMalloc(&dev_queueCounts, QUEUE_COUNT * sizeof(int));

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
    cudaFree(dev_aliveCounter);
    dev_aliveCounter = NULL;
    cudaFree(dev_queues);
    cudaFree(dev_queueCounts);
    dev_queues = NULL;
    dev_queueCounts = NULL;
    cudaFree(dev_cubScratch);
    cudaFree(dev_meshes);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_triGeoms);
    cudaFree(dev_triangles);
    cudaFree(dev_vertexNormals);
    cudaFree(dev_vertexUVs);
    cudaFree(dev_vertexTangents);
    freeTextures();
    cudaFree(dev_lights);
    cudaFree(dev_lightCdf);
    cudaFree(dev_triLightIndex);
    cudaFree(dev_shadowRays);
    denoiserFree();
    denoiserReady = false;
    cudaFree(dev_albedo);
    cudaFree(dev_normal);
    cudaFree(dev_dnColor);
    cudaFree(dev_dnAlbedo);
    cudaFree(dev_dnNormal);
    cudaFree(dev_denoised);
    dev_albedo = dev_normal = dev_dnColor = dev_dnAlbedo = dev_dnNormal = dev_denoised = NULL;
    dev_lights = NULL;
    dev_lightCdf = NULL;
    dev_triLightIndex = NULL;
    dev_shadowRays = NULL;
    dev_meshes = NULL;
    dev_bvhNodes = NULL;
    dev_triGeoms = NULL;
    dev_triangles = NULL;
    dev_vertexNormals = NULL;
    dev_vertexUVs = NULL;
    dev_vertexTangents = NULL;
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
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));
    denoisedValid = false;
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

        // Motion blur: each path sees the scene at one random shutter time.
        segment.ray.time = settings.motionBlur ? sample1D(sampler, DIM_TIME) : 0.0f;

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
    SceneView scene,
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
            isect.primId = -1;
        }
        else
        {
            intersectScene(scene, pathSegment.ray, isect);
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
    SceneView scene;
    LightsView lights;
    EnvironmentView env;
    glm::vec3* image;
    glm::vec3* aovAlbedo;   // first-hit features for the denoiser
    glm::vec3* aovNormal;
    int* aliveCounter;      // number of paths that continue after this bounce
};

// Counts the calling threads whose path survives, with one atomic per warp.
__device__ inline void countAlive(int* counter, bool alive)
{
    unsigned int active = __activemask();
    unsigned int ballot = __ballot_sync(active, alive);
    if ((threadIdx.x & 31) == __ffs(active) - 1 && ballot != 0)
    {
        atomicAdd(counter, __popc(ballot));
    }
}

// Records the denoiser features once per path, at the first surface that is
// not a perfect mirror/glass (those show the next surface instead).
__device__ inline void recordFeatures(const ShadeParams& p, PathSegment& path, glm::vec3 albedo, glm::vec3 normal)
{
    if (path.flags & PATH_FLAG_AOV_DONE)
    {
        return;
    }
    p.aovAlbedo[path.pixelIndex] += glm::clamp(albedo, glm::vec3(0.0f), glm::vec3(1.0f));
    p.aovNormal[path.pixelIndex] += normal;
    path.flags |= PATH_FLAG_AOV_DONE;
}

// Whether next event estimation can contribute for these BSDF parameters.
__device__ inline bool hasNonDeltaLobe(const BSDFParams& p)
{
    switch (p.type)
    {
    case MATERIAL_DIFFUSE:
        return true;
    case MATERIAL_PBR:
        return p.alpha >= DELTA_ALPHA || p.metallic < 1.0f;
    case MATERIAL_SPECULAR:
    case MATERIAL_DIELECTRIC:
        return p.alpha >= DELTA_ALPHA;
    default:
        return false;
    }
}

// Weight of emission found by BSDF sampling. With MIS it is balanced against
// the light sampling pdf of the same direction; with NEE only, emission from
// samplable lights is skipped (light sampling already accounted for it).
__device__ inline float bsdfHitWeight(const ShadeParams& p, const PathSegment& path, float lightPdf)
{
    if (!p.settings.nextEventEstimation || (path.flags & PATH_FLAG_DELTA_BOUNCE))
    {
        return 1.0f;
    }
    if (!p.settings.multipleImportance)
    {
        return lightPdf > 0.0f ? 0.0f : 1.0f;
    }
    return powerHeuristic(path.lastPdf, lightPdf);
}

// Shades one bounce: accumulates emission into the image, samples a light
// (next event estimation) and the BSDF, and spawns the continuation ray.
// Paths that miss the scene, hit a light, or run out of bounces are marked as
// terminated (remainingBounces = 0) so that they can be stream compacted away.
//
// TYPE is a MaterialType when every path handed to this function is known to
// hit that material type (wavefront queues): the BSDF switch statements then
// fold away at compile time. TYPE = -1 handles any material (megakernel).
template <int TYPE>
__device__ inline void shadePath(
    const ShadeParams& params,
    int idx,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    ShadowRay* shadowRays)
{
    PathSegment path = pathSegments[idx];
    if (path.remainingBounces <= 0)
    {
        return;
    }

    const int depth = params.depth;
    ShadeableIntersection intersection = shadeableIntersections[idx];

    // In the specialized queue kernels these are compile-time constants, so
    // the miss and emitter kernels only contain their own small code paths.
    const bool isMiss = TYPE == QUEUE_MISS || (TYPE < 0 && intersection.t <= 0.0f);

    if (isMiss)
    {
        // The ray escaped the scene and sees the environment.
        if (params.env.enabled)
        {
            glm::vec3 dir = path.ray.direction;
            float lightPdf = params.lights.envSelectProb > 0.0f
                ? params.lights.envSelectProb * environmentPdf(params.env, dir) : 0.0f;
            float w = bsdfHitWeight(params, path, lightPdf);
            glm::vec3 radiance = environmentRadiance(params.env, dir);
            addToImage(params.image, path.pixelIndex, path.throughput * radiance * w);
            recordFeatures(params, path, radiance, glm::vec3(0.0f));
        }
        recordFeatures(params, path, glm::vec3(0.0f), glm::vec3(0.0f));
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    // Beer-Lambert absorption along the segment travelled inside a medium.
    if (path.mediumMaterial >= 0)
    {
        glm::vec3 sigma = params.scene.materials[path.mediumMaterial].absorption;
        path.throughput *= glm::exp(-sigma * intersection.t);
    }

    const Material material = params.scene.materials[intersection.materialId];
    const bool isEmitter = TYPE >= 0 ? TYPE == MATERIAL_EMITTING : material.type == MATERIAL_EMITTING;
    SurfaceHit hit = computeSurfaceHit(params.scene, path.ray, intersection);
    glm::vec3 wo = -path.ray.direction;
    // Interpolated normals can face away from the viewer at silhouettes; fall
    // back to the geometric normal there.
    if (glm::dot(wo, hit.shadingNormal) <= 0.0f)
    {
        hit.shadingNormal = hit.normal;
    }
    MaterialEval mat;
    if (isEmitter)
    {
        mat.emission = material.emission;
    }
    else
    {
        mat = evaluateMaterial(material, hit, params.scene.textures, wo);
        if (TYPE >= 0 && TYPE < MATERIAL_TYPE_COUNT)
        {
            mat.bsdf.type = TYPE;
        }
    }

    // Emission found by following the BSDF sample of the previous bounce.
    if (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f)
    {
        int lightIndex = intersection.primId >= 0
            ? (params.scene.triLightIndex ? params.scene.triLightIndex[intersection.primId] : -1)
            : params.scene.geoms[intersection.geomId].lightIndex;
        float lightPdf = 0.0f;
        if (lightIndex >= 0)
        {
            const Light& light = params.scene.lights[lightIndex];
            if (light.geomId == intersection.geomId)
            {
                lightPdf = areaLightPdf(params.scene, light, path.ray.origin, path.ray.direction,
                    intersection.t, hit.normal, path.ray.time);
            }
        }
        float w = bsdfHitWeight(params, path, lightPdf);
        addToImage(params.image, path.pixelIndex, path.throughput * mat.emission * w);
    }
    if (isEmitter)
    {
        recordFeatures(params, path, mat.emission, hit.normal);
        pathSegments[idx].remainingBounces = 0;
        return;
    }
    if (hasNonDeltaLobe(mat.bsdf) || depth >= 2)
    {
        recordFeatures(params, path, mat.albedo, mat.shadingNormal);
    }

    SampleContext sampler = makeSampleContext(path.pixelIndex, params.iter, params.settings.samplerType);
    const BSDFParams& bsdf = mat.bsdf;
    Frame frame = makeFrame(mat.shadingNormal);

    // --- Next event estimation: sample a point on a light (or a direction of
    // the environment) and queue a shadow ray carrying its contribution.
    const bool canSampleLights = params.lights.count > 0 || params.lights.envSelectProb > 0.0f;
    if (params.settings.nextEventEstimation && canSampleLights && hasNonDeltaLobe(bsdf))
    {
        LightSample ls;
        bool valid = false;
        float uSelect = sample1D(sampler, bounceDimension(depth, BDIM_LIGHT_SELECT));
        glm::vec2 uLight = sample2D(sampler, bounceDimension(depth, BDIM_LIGHT));
        float pEnv = params.lights.envSelectProb;
        if (uSelect < pEnv)
        {
            float pdf;
            ls.wi = sampleEnvironment(params.env, uLight, pdf);
            ls.pdf = pEnv * pdf;
            ls.distance = FLT_MAX;
            ls.radiance = environmentRadiance(params.env, ls.wi);
            valid = ls.pdf > 0.0f;
        }
        else if (params.lights.count > 0)
        {
            float u = (uSelect - pEnv) / (1.0f - pEnv);
            int li = selectLight(params.lights, u);
            valid = sampleAreaLight(params.scene, params.lights.lights[li], hit.position, uLight, path.ray.time, ls);
        }

        if (valid && maxComponent(ls.radiance) > 0.0f)
        {
            float bsdfPdf;
            glm::vec3 fcos = evalBSDF(bsdf, frame, wo, ls.wi, bsdfPdf);
            bool transmits = glm::dot(ls.wi, hit.normal) < 0.0f;
            // reflection must stay above the geometric surface, transmission below
            bool sideOk = !transmits || bsdf.type == MATERIAL_DIELECTRIC;
            if (sideOk && maxComponent(fcos) > 0.0f)
            {
                float w = params.settings.multipleImportance ? powerHeuristic(ls.pdf, bsdfPdf) : 1.0f;
                glm::vec3 contribution = path.throughput * fcos * ls.radiance * (w / ls.pdf);
                if (isFiniteVec(contribution) && maxComponent(contribution) > 0.0f)
                {
                    ShadowRay sr;
                    sr.origin = hit.position + (transmits ? -hit.normal : hit.normal) * RAY_EPSILON;
                    sr.direction = ls.wi;
                    sr.maxT = ls.distance == FLT_MAX ? FLT_MAX : ls.distance * (1.0f - 1e-3f) - RAY_EPSILON;
                    sr.contribution = contribution;
                    sr.pixelIndex = path.pixelIndex;
                    sr.time = path.ray.time;
                    shadowRays[idx] = sr;
                }
            }
        }
    }

    // --- BSDF sampling for the continuation ray.
    BSDFSample bs;
    float uLobe = sample1D(sampler, bounceDimension(depth, BDIM_LOBE));
    glm::vec2 uDir = sample2D(sampler, bounceDimension(depth, BDIM_BSDF));
    bool scattered = sampleBSDF(bsdf, frame, wo, uLobe, uDir, bs)
        && maxComponent(bs.weight) > 0.0f && isFiniteVec(bs.weight);
    // With shading normals a sampled direction can end up on the wrong side
    // of the actual surface; such samples would leak light, so drop them.
    if (scattered && ((glm::dot(bs.wi, hit.normal) > 0.0f) == bs.isTransmission))
    {
        scattered = false;
    }

    path.remainingBounces--;
    if (!scattered)
    {
        path.remainingBounces = 0;
    }
    else
    {
        path.throughput *= bs.weight;
        path.lastPdf = bs.pdf;
        path.flags = (path.flags & PATH_FLAG_AOV_DONE) | (bs.isDelta ? PATH_FLAG_DELTA_BOUNCE : 0);
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
    countAlive(params.aliveCounter, path.remainingBounces > 0);
}

// Megakernel: shades every path whatever its material.
__global__ void shadeMaterial(
    ShadeParams params,
    int num_paths,
    const int* order,   // optional permutation (material-sorted indices)
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    ShadowRay* shadowRays)
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
    shadowRays[idx].pixelIndex = -1;
    shadePath<-1>(params, idx, shadeableIntersections, pathSegments, shadowRays);
}

// Appends every live path to the queue of the material type it hit. Lanes of
// a warp that go to the same queue reserve their slots with one atomic.
__global__ void classifyPaths(int num_paths, const PathSegment* paths, const ShadeableIntersection* isects,
    const Material* materials, int* queues, int capacity, int* counts, ShadowRay* shadowRays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }
    shadowRays[idx].pixelIndex = -1;
    if (paths[idx].remainingBounces <= 0)
    {
        return;
    }
    const ShadeableIntersection& isect = isects[idx];
    int queue = isect.t > 0.0f ? materials[isect.materialId].type : QUEUE_MISS;

    unsigned int active = __activemask();
    unsigned int peers = __match_any_sync(active, queue);
    int lane = threadIdx.x & 31;
    int leader = __ffs(peers) - 1;
    int rank = __popc(peers & ((1u << lane) - 1u));
    int base = 0;
    if (lane == leader)
    {
        base = atomicAdd(&counts[queue], __popc(peers));
    }
    base = __shfl_sync(peers, base, leader);
    queues[queue * capacity + base + rank] = idx;
}

// Shades the paths of one queue with a kernel specialized for its material.
template <int TYPE>
__global__ void shadeQueue(
    ShadeParams params,
    const int* queue,
    int count,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    ShadowRay* shadowRays)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    shadePath<TYPE>(params, queue[i], shadeableIntersections, pathSegments, shadowRays);
}

// Traces the shadow rays queued by the shading kernel and adds the light
// contribution of the unoccluded ones.
__global__ void traceShadowRays(int num_paths, const ShadowRay* shadowRays, SceneView scene, glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }
    const ShadowRay sr = shadowRays[idx];
    if (sr.pixelIndex < 0)
    {
        return;
    }
    Ray ray;
    ray.origin = sr.origin;
    ray.direction = sr.direction;
    ray.time = sr.time;
    if (!occluded(scene, ray, sr.maxT))
    {
        addToImage(image, sr.pixelIndex, sr.contribution);
    }
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

template <int TYPE>
static void launchQueue(const ShadeParams& params, const int* counts, int capacity, int blockSize)
{
    int count = counts[TYPE];
    if (count > 0)
    {
        shadeQueue<TYPE><<<(count + blockSize - 1) / blockSize, blockSize>>>(
            params, dev_queues + (size_t)TYPE * capacity, count, dev_intersections, dev_paths, dev_shadowRays);
    }
}

// Wavefront shading: bucket paths by material type, then run one specialized
// kernel per non-empty bucket. The classification counts as sorting time.
static void shadeWavefront(const ShadeParams& params, int num_paths, int blockSize, bool profile)
{
    const int capacity = hst_scene->state.camera.resolution.x * hst_scene->state.camera.resolution.y;
    stageBegin(profile);
    cudaMemsetAsync(dev_queueCounts, 0, QUEUE_COUNT * sizeof(int));
    classifyPaths<<<(num_paths + blockSize - 1) / blockSize, blockSize>>>(num_paths, dev_paths, dev_intersections,
        dev_materials, dev_queues, capacity, dev_queueCounts, dev_shadowRays);
    int counts[QUEUE_COUNT];
    cudaMemcpy(counts, dev_queueCounts, sizeof(counts), cudaMemcpyDeviceToHost);
    checkCUDAError("classify paths");
    stageEnd(profile, STAGE_SORT);

    stageBegin(profile);
    launchQueue<MATERIAL_DIFFUSE>(params, counts, capacity, blockSize);
    launchQueue<MATERIAL_SPECULAR>(params, counts, capacity, blockSize);
    launchQueue<MATERIAL_DIELECTRIC>(params, counts, capacity, blockSize);
    launchQueue<MATERIAL_PBR>(params, counts, capacity, blockSize);
    launchQueue<MATERIAL_EMITTING>(params, counts, capacity, blockSize);
    launchQueue<QUEUE_MISS>(params, counts, capacity, blockSize);
    checkCUDAError("shade queues");
    stageEnd(profile, STAGE_SHADE);
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

    const SceneView sceneView = makeSceneView(settings);
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
            sceneView,
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
        const bool regroup = depth >= settings.coherenceStartDepth;
        if (settings.sortMode != SORT_OFF && !settings.wavefront && regroup)
        {
            stageBegin(profile);
            shadeOrder = sortPathsByMaterial(settings.sortMode, num_paths, blockSize1d);
            stageEnd(profile, STAGE_SORT);
        }

        ShadeParams shadeParams;
        shadeParams.iter = iter;
        shadeParams.depth = depth;
        shadeParams.settings = settings;
        shadeParams.scene = sceneView;
        shadeParams.lights = makeLightsView();
        shadeParams.env = makeEnvironmentView();
        shadeParams.image = dev_image;
        shadeParams.aovAlbedo = dev_albedo;
        shadeParams.aovNormal = dev_normal;
        shadeParams.aliveCounter = dev_aliveCounter;
        cudaMemsetAsync(dev_aliveCounter, 0, sizeof(int));
        if (settings.wavefront && regroup)
        {
            shadeWavefront(shadeParams, num_paths, blockSize1d, profile);
        }
        else
        {
            stageBegin(profile);
            shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
                shadeParams,
                num_paths,
                shadeOrder,
                dev_intersections,
                dev_paths,
                dev_shadowRays
            );
            checkCUDAError("shade");
            stageEnd(profile, STAGE_SHADE);
        }

        if (settings.nextEventEstimation)
        {
            stageBegin(profile);
            traceShadowRays<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths, dev_shadowRays, sceneView, dev_image);
            checkCUDAError("shadow rays");
            stageEnd(profile, STAGE_SHADOW);
        }
        depth++;

        if (settings.compactionMode != COMPACT_OFF)
        {
            // Compacting copies every live path, which only pays off when a
            // good fraction of them terminated (e.g. not in closed scenes).
            stageBegin(profile);
            int alive = 0;
            cudaMemcpy(&alive, dev_aliveCounter, sizeof(int), cudaMemcpyDeviceToHost);
            if (alive == 0)
            {
                num_paths = 0;
            }
            else if (alive <= settings.compactThreshold * num_paths)
            {
                num_paths = compactPaths(settings.compactionMode, num_paths);
            }
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

    denoisedValid = false;
    if (settings.denoise && denoiserReady && pbo != NULL
        && (iter % std::max(1, settings.denoiseInterval) == 0 || iter == (int)hst_scene->state.iterations))
    {
        pathtraceDenoise(iter);
    }

    // Send results to OpenGL buffer for rendering
    if (pbo != NULL)
    {
        const glm::vec3* source = dev_image;
        float scale = 1.0f / iter;
        bool raw = false;
        if (settings.displayMode == DISPLAY_ALBEDO || settings.displayMode == DISPLAY_NORMAL)
        {
            source = settings.displayMode == DISPLAY_ALBEDO ? dev_albedo : dev_normal;
            raw = true;
        }
        else if (settings.denoise && denoisedValid)
        {
            source = dev_denoised;
            scale = 1.0f;
        }
        if (settings.displayMode == DISPLAY_BVH_COST)
        {
            std::vector<glm::vec2> cost;
            pathtraceBvhCost(pbo, cost);
        }
        else if (settings.displayMode == DISPLAY_NORMAL)
        {
            // map [-1, 1] to [0, 1] for viewing
            showNormals<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, 1.0f / iter, dev_normal);
        }
        else
        {
            sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, scale, source, settings, raw);
        }
    }

    checkCUDAError("pathtrace");
}

// Cost of the camera ray through each pixel center (BVH debug view).
__global__ void bvhCostKernel(Camera cam, SceneView scene, glm::vec2* cost)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y)
    {
        return;
    }
    Ray ray;
    ray.origin = cam.position;
    ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
    ray.time = 0.0f;
    TraversalCounters counters = { 0, 0 };
    ShadeableIntersection isect;
    intersectScene(scene, ray, isect, &counters);
    cost[x + y * cam.resolution.x] = glm::vec2((float)counters.boxTests, (float)counters.triangleTests);
}

// Sequential single-hue ramp (light = cheap, dark = expensive) for the cost
// of a ray: box tests plus triangle tests, saturating at `maxCost`.
__global__ void showBvhCost(uchar4* pbo, glm::ivec2 resolution, const glm::vec2* cost, float maxCost)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y)
    {
        return;
    }
    const glm::vec3 ramp[5] = {
        glm::vec3(0xfc, 0xfc, 0xfb), glm::vec3(0xb7, 0xd3, 0xf6), glm::vec3(0x55, 0x98, 0xe7),
        glm::vec3(0x25, 0x6a, 0xbf), glm::vec3(0x0d, 0x36, 0x6b) };
    int index = x + y * resolution.x;
    float t = glm::clamp((cost[index].x + cost[index].y) / maxCost, 0.0f, 1.0f) * 4.0f;
    int i = glm::min((int)t, 3);
    glm::vec3 c = glm::mix(ramp[i], ramp[i + 1], t - i);
    pbo[index] = make_uchar4((unsigned char)c.x, (unsigned char)c.y, (unsigned char)c.z, 0);
}

void pathtraceBvhCost(uchar4* pbo, std::vector<glm::vec2>& cost)
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    glm::vec2* dev_cost = NULL;
    cudaMalloc(&dev_cost, pixelcount * sizeof(glm::vec2));
    const dim3 block(8, 8);
    const dim3 grid((cam.resolution.x + 7) / 8, (cam.resolution.y + 7) / 8);
    bvhCostKernel<<<grid, block>>>(cam, makeSceneView(guiData->settings), dev_cost);
    if (pbo != NULL)
    {
        showBvhCost<<<grid, block>>>(pbo, cam.resolution, dev_cost, 160.0f);
    }
    cost.resize(pixelcount);
    cudaMemcpy(cost.data(), dev_cost, pixelcount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    cudaFree(dev_cost);
    checkCUDAError("bvh cost");
}

// Averages the accumulation buffers into the denoiser inputs.
__global__ void prepareDenoiserInputs(int n, float scale, const glm::vec3* image, const glm::vec3* albedo,
    const glm::vec3* normal, glm::vec3* outColor, glm::vec3* outAlbedo, glm::vec3* outNormal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        outColor[i] = image[i] * scale;
        outAlbedo[i] = albedo[i] * scale;
        outNormal[i] = normal[i] * scale;
    }
}

bool pathtraceDenoise(int iter)
{
    if (!denoiserReady || iter <= 0)
    {
        return false;
    }
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const RenderSettings& settings = guiData->settings;
    cudaEventRecord(stageStartEvent);
    const int blockSize = 256;
    prepareDenoiserInputs<<<(pixelcount + blockSize - 1) / blockSize, blockSize>>>(pixelcount, 1.0f / iter,
        dev_image, dev_albedo, dev_normal, dev_dnColor, dev_dnAlbedo, dev_dnNormal);
    checkCUDAError("prepare denoiser inputs");
    bool ok = denoiserRun(settings.denoiseAux, settings.denoisePrefilter, settings.denoiseHighQuality);
    cudaEventRecord(stageStopEvent);
    cudaEventSynchronize(stageStopEvent);
    cudaEventElapsedTime(&guiData->stats.lastDenoiseMs, stageStartEvent, stageStopEvent);
    denoisedValid = ok;
    return ok;
}

void pathtraceCopyDenoisedToHost(std::vector<glm::vec3>& out)
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    out.resize(pixelcount);
    cudaMemcpy(out.data(), dev_denoised, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void pathtraceCopyFeaturesToHost(int iter, std::vector<glm::vec3>& albedo, std::vector<glm::vec3>& normal)
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    albedo.resize(pixelcount);
    normal.resize(pixelcount);
    cudaMemcpy(albedo.data(), dev_albedo, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(normal.data(), dev_normal, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < pixelcount; i++)
    {
        albedo[i] /= (float)iter;
        normal[i] /= (float)iter;
    }
}

void pathtraceGetAccumulation(std::vector<glm::vec3>& image, std::vector<glm::vec3>& albedo,
    std::vector<glm::vec3>& normal)
{
    const Camera& cam = hst_scene->state.camera;
    const size_t pixelcount = (size_t)cam.resolution.x * cam.resolution.y;
    image.resize(pixelcount);
    albedo.resize(pixelcount);
    normal.resize(pixelcount);
    cudaMemcpy(image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(albedo.data(), dev_albedo, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(normal.data(), dev_normal, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void pathtraceSetAccumulation(const std::vector<glm::vec3>& image, const std::vector<glm::vec3>& albedo,
    const std::vector<glm::vec3>& normal)
{
    const Camera& cam = hst_scene->state.camera;
    const size_t pixelcount = (size_t)cam.resolution.x * cam.resolution.y;
    if (image.size() == pixelcount)
    {
        cudaMemcpy(dev_image, image.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    if (albedo.size() == pixelcount && normal.size() == pixelcount)
    {
        cudaMemcpy(dev_albedo, albedo.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_normal, normal.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    denoisedValid = false;
    checkCUDAError("restore accumulation");
}

const char* pathtraceDenoiserName()
{
    static std::string name;
    name = denoiserDeviceName();
    return name.c_str();
}
