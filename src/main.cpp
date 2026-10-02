#include "checkpoint.h"
#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "postprocess.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Command line options
struct CommandLineOptions
{
    bool headless = false;
    int spp = -1;               // overrides ITERATIONS from the scene file
    int depth = -1;             // overrides DEPTH from the scene file
    int resX = -1, resY = -1;   // overrides RES from the scene file
    int warmup = 0;             // iterations excluded from timing statistics
    std::string output;         // overrides FILE from the scene file
    std::string statsFile;      // appends a CSV line of timing statistics
    bool savePfm = false;       // also write the raw linear average as .pfm
    bool saveFeatures = false;  // also write the albedo/normal denoiser features
    bool saveState = false;     // headless: write a checkpoint at the end
    int checkpointEvery = 0;    // headless: write a checkpoint every N iterations
    bool bvhOverride = false;   // BVH build settings given on the command line
    BVHBuildSettings bvh;
};
static CommandLineOptions options;

static Checkpoint resumeState;
static bool resuming = false;

// Forward declarations for window loop and interactivity
void runCuda();
void saveImage();
void saveState();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}

static const char* stageNames[STAGE_COUNT] = {
    "Generate", "Intersect", "Sort", "Shade", "Compact", "Shadow"
};

void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    RenderSettings& settings = imguiData->settings;
    RenderStats& stats = imguiData->stats;
    bool resetNeeded = false;

    ImGui::Begin("Path Tracer Analytics");

    ImGui::Text("Iteration %d / %d", iteration, renderState->iterations);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Path tracing: %.2f ms/iter (avg %.2f ms)", stats.lastIterationMs, stats.avgIterationMs());
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    if (ImGui::CollapsingHeader("Integrator", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const char* compactModes[COMPACT_MODE_COUNT] = { "Off", "thrust::remove_if", "CUB select (ping-pong)" };
        const char* sortModes[SORT_MODE_COUNT] = { "Off", "thrust::sort_by_key", "CUB radix + gather", "CUB radix, indirect" };
        resetNeeded |= ImGui::Combo("Stream compaction", &settings.compactionMode, compactModes, COMPACT_MODE_COUNT);
        resetNeeded |= ImGui::Checkbox("Wavefront material queues", &settings.wavefront);
        resetNeeded |= ImGui::SliderInt("Regroup from bounce", &settings.coherenceStartDepth, 0, 8);
        if (!settings.wavefront)
        {
            resetNeeded |= ImGui::Combo("Material sort", &settings.sortMode, sortModes, SORT_MODE_COUNT);
        }
        const char* samplers[] = { "Random (hashed)", "Sobol (Owen scrambled)" };
        resetNeeded |= ImGui::Combo("Sampler", &settings.samplerType, samplers, 2);
        resetNeeded |= ImGui::Checkbox("Stochastic antialiasing", &settings.antialiasing);
        resetNeeded |= ImGui::Checkbox("Motion blur", &settings.motionBlur);
        resetNeeded |= ImGui::Checkbox("Next event estimation", &settings.nextEventEstimation);
        if (settings.nextEventEstimation)
        {
            resetNeeded |= ImGui::Checkbox("Multiple importance sampling", &settings.multipleImportance);
        }
        resetNeeded |= ImGui::Checkbox("Russian roulette", &settings.russianRoulette);
        if (settings.russianRoulette)
        {
            resetNeeded |= ImGui::SliderInt("RR start depth", &settings.rrStartDepth, 1, 16);
        }
        resetNeeded |= ImGui::SliderInt("Max depth", &renderState->traceDepth, 1, MAX_TRACKED_DEPTH);
    }

    if (ImGui::CollapsingHeader("Acceleration", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("%zu triangles, %zu meshes, %zu BVH nodes", scene->triangles.size(),
            scene->meshes.size(), scene->bvhNodes.size());
        resetNeeded |= ImGui::Checkbox("Mesh BVH", &settings.useBVH);
        resetNeeded |= ImGui::Checkbox("Bounding box culling", &settings.cullBounds);
    }

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Camera& cam = renderState->camera;
        resetNeeded |= ImGui::SliderFloat("Lens radius", &cam.lensRadius, 0.0f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        resetNeeded |= ImGui::SliderFloat("Focal distance", &cam.focalDistance, 0.1f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    }

    if (ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Open Image Denoise: %s", pathtraceDenoiserName());
        ImGui::Checkbox("Denoise", &settings.denoise);
        ImGui::SliderInt("Every N iterations", &settings.denoiseInterval, 1, 256);
        ImGui::Checkbox("Albedo + normal guides", &settings.denoiseAux);
        ImGui::Checkbox("Prefilter guides", &settings.denoisePrefilter);
        ImGui::Checkbox("High quality", &settings.denoiseHighQuality);
        ImGui::Text("Last denoise: %.2f ms", stats.lastDenoiseMs);
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const char* displayModes[DISPLAY_MODE_COUNT] = { "Render", "Albedo feature", "Normal feature" };
        ImGui::Combo("Show", &settings.displayMode, displayModes, DISPLAY_MODE_COUNT);
        const char* toneMaps[TONEMAP_COUNT] = { "Linear clamp", "Reinhard", "ACES" };
        ImGui::Combo("Tone map", &settings.toneMap, toneMaps, TONEMAP_COUNT);
        ImGui::Checkbox("Gamma 2.2", &settings.gammaCorrect);
        ImGui::SliderFloat("Exposure", &settings.exposure, 0.05f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::Button("Save image (S)"))
        {
            saveImage();
        }
        ImGui::SameLine();
        if (ImGui::Button("Save checkpoint (P)"))
        {
            saveState();
        }
    }

    if (ImGui::CollapsingHeader("Profiling"))
    {
        if (ImGui::Checkbox("Per-stage timing (adds syncs)", &stats.profileStages))
        {
            resetNeeded = true;
        }
        if (stats.profileStages && stats.sampledIterations > 0)
        {
            for (int s = 0; s < STAGE_COUNT; ++s)
            {
                ImGui::Text("%-10s %8.3f ms", stageNames[s], stats.stageMs[s] / stats.sampledIterations);
            }
            float alive[MAX_TRACKED_DEPTH];
            for (int d = 0; d < stats.numBounces; ++d)
            {
                alive[d] = (float)(stats.alivePaths[d] / stats.sampledIterations);
            }
            ImGui::PlotHistogram("Alive paths", alive, stats.numBounces, 0, NULL, 0.0f,
                (float)(width * height), ImVec2(0, 80));
        }
    }

    ImGui::End();

    if (resetNeeded)
    {
        camchanged = true;  // restarts accumulation
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

static bool parseBool(const std::string& v)
{
    return v == "1" || v == "true" || v == "on" || v == "yes";
}

static void printUsage(const char* exe)
{
    printf("Usage: %s SCENEFILE.json|CHECKPOINT.ptstate [options]\n", exe);
    printf("  --headless          render without a window, save the image and exit\n");
    printf("  --spp N             number of iterations (overrides ITERATIONS)\n");
    printf("  --depth N           maximum path depth (overrides DEPTH)\n");
    printf("  --res WxH           image resolution (overrides RES)\n");
    printf("  --out NAME          output file prefix (overrides FILE)\n");
    printf("  --warmup N          iterations excluded from timing statistics\n");
    printf("  --profile           collect per-stage timings and alive-path counts\n");
    printf("  --stats FILE        append timing statistics as CSV to FILE\n");
    printf("  --pfm               also save the raw linear radiance as a .pfm file\n");
    printf("  --denoise 0|1       run Open Image Denoise on the final image (saved as *.denoised.png)\n");
    printf("  --denoise-aux 0|1, --denoise-prefilter 0|1   denoiser feature options\n");
    printf("  --save-features     also save the albedo and normal feature images\n");
    printf("  --save-state        headless: save a checkpoint (*.ptstate) at the end\n");
    printf("  --checkpoint N      headless: save a checkpoint every N iterations\n");
    printf("Pass a .ptstate file instead of a scene to resume a saved render; its settings are restored.\n");
    printf("  --sort off|thrust|cub       sort paths by material before shading\n");
    printf("  --compact off|thrust|cub    stream compact terminated paths\n");
    printf("  --aa 0|1            stochastic sampled antialiasing\n");
    printf("  --motion 0|1        motion blur (objects with MOTION move during the exposure)\n");
    printf("  --rr 0|1            russian roulette path termination, --rr-depth N first bounce\n");
    printf("  --nee 0|1           next event estimation (direct light sampling)\n");
    printf("  --wavefront 0|1     shade with per-material queues and specialized kernels\n");
    printf("  --regroup-depth N   sort/queue paths only from bounce N on\n");
    printf("  --sampler random|sobol  random numbers for all sample dimensions\n");
    printf("  --mis 0|1           multiple importance sampling of light and BSDF samples\n");
    printf("  --bvh 0|1           traverse mesh BVHs (0 = test every triangle)\n");
    printf("  --cull 0|1          test object bounding boxes before their geometry\n");
    printf("  --bvh-leaf N, --bvh-depth N, --bvh-bins N   BVH build parameters\n");
    printf("  --tonemap linear|reinhard|aces, --gamma 0|1, --exposure F\n");
}

// Returns false if the program should exit.
static bool parseCommandLine(int argc, char** argv, RenderSettings& settings, RenderStats& stats)
{
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&](void) -> std::string {
            if (i + 1 >= argc)
            {
                printf("Missing value for %s\n", arg.c_str());
                exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (arg == "--headless") options.headless = true;
        else if (arg == "--spp") options.spp = std::stoi(next());
        else if (arg == "--depth") options.depth = std::stoi(next());
        else if (arg == "--res")
        {
            std::string v = next();
            size_t x = v.find('x');
            options.resX = std::stoi(v.substr(0, x));
            options.resY = std::stoi(v.substr(x + 1));
        }
        else if (arg == "--out") options.output = next();
        else if (arg == "--warmup") options.warmup = std::stoi(next());
        else if (arg == "--profile") stats.profileStages = true;
        else if (arg == "--stats") options.statsFile = next();
        else if (arg == "--pfm") options.savePfm = true;
        else if (arg == "--denoise") settings.denoise = parseBool(next());
        else if (arg == "--denoise-aux") settings.denoiseAux = parseBool(next());
        else if (arg == "--denoise-prefilter") settings.denoisePrefilter = parseBool(next());
        else if (arg == "--save-features") options.saveFeatures = true;
        else if (arg == "--save-state") options.saveState = true;
        else if (arg == "--checkpoint") options.checkpointEvery = std::stoi(next());
        else if (arg == "--sort")
        {
            std::string v = next();
            settings.sortMode = v == "thrust" ? SORT_THRUST : v == "indirect" ? SORT_CUB_INDIRECT
                : (v == "cub" || parseBool(v)) ? SORT_CUB : SORT_OFF;
        }
        else if (arg == "--compact")
        {
            std::string v = next();
            settings.compactionMode = v == "thrust" ? COMPACT_THRUST
                : (v == "cub" || parseBool(v)) ? COMPACT_CUB : COMPACT_OFF;
        }
        else if (arg == "--aa") settings.antialiasing = parseBool(next());
        else if (arg == "--motion") settings.motionBlur = parseBool(next());
        else if (arg == "--rr") settings.russianRoulette = parseBool(next());
        else if (arg == "--nee") settings.nextEventEstimation = parseBool(next());
        else if (arg == "--wavefront") settings.wavefront = parseBool(next());
        else if (arg == "--regroup-depth") settings.coherenceStartDepth = std::stoi(next());
        else if (arg == "--sampler") settings.samplerType = next() == "random" ? 0 : 1;
        else if (arg == "--mis") settings.multipleImportance = parseBool(next());
        else if (arg == "--rr-depth") settings.rrStartDepth = std::stoi(next());
        else if (arg == "--bvh") settings.useBVH = parseBool(next());
        else if (arg == "--cull") settings.cullBounds = parseBool(next());
        else if (arg == "--bvh-leaf") { options.bvhOverride = true; options.bvh.maxLeafSize = std::stoi(next()); }
        else if (arg == "--bvh-depth") { options.bvhOverride = true; options.bvh.maxDepth = std::stoi(next()); }
        else if (arg == "--bvh-bins") { options.bvhOverride = true; options.bvh.numBins = std::stoi(next()); }
        else if (arg == "--gamma") settings.gammaCorrect = parseBool(next());
        else if (arg == "--exposure") settings.exposure = std::stof(next());
        else if (arg == "--tonemap")
        {
            std::string v = next();
            settings.toneMap = v == "aces" ? TONEMAP_ACES : v == "reinhard" ? TONEMAP_REINHARD : TONEMAP_NONE;
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return false;
        }
        else
        {
            printf("Unknown option %s\n", arg.c_str());
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

static void writeStats(const char* sceneFile)
{
    const RenderStats& stats = guiData->stats;
    const RenderSettings& s = guiData->settings;
    printf("Rendered %d iterations, %.3f ms/iteration (averaged over %d after %d warmup)\n",
        iteration, stats.avgIterationMs(), stats.sampledIterations, stats.warmupIterations);
    if (stats.profileStages && stats.sampledIterations > 0)
    {
        for (int st = 0; st < STAGE_COUNT; ++st)
        {
            printf("  %-10s %8.3f ms\n", stageNames[st], stats.stageMs[st] / stats.sampledIterations);
        }
        printf("  alive paths per bounce:");
        for (int d = 0; d < stats.numBounces; ++d)
        {
            printf(" %.0f", stats.alivePaths[d] / stats.sampledIterations);
        }
        printf("\n");
    }

    if (options.statsFile.empty())
    {
        return;
    }
    std::ofstream out(options.statsFile, std::ios::app);
    out << sceneFile << "," << renderState->imageName << "," << iteration << "," << renderState->traceDepth
        << "," << s.compactionMode << "," << s.sortMode << "," << s.antialiasing
        << "," << s.russianRoulette << "," << s.useBVH << "," << s.cullBounds
        << "," << s.nextEventEstimation << "," << s.multipleImportance << "," << s.samplerType << "," << s.wavefront
        << "," << stats.avgIterationMs();
    for (int st = 0; st < STAGE_COUNT; ++st)
    {
        out << "," << (stats.sampledIterations ? stats.stageMs[st] / stats.sampledIterations : 0.0);
    }
    out << ",\"";
    for (int d = 0; d < stats.numBounces; ++d)
    {
        out << (d ? " " : "") << (stats.sampledIterations ? stats.alivePaths[d] / stats.sampledIterations : 0.0);
    }
    out << "\"\n";
}

// Recomputes the camera basis from the orbit parameters (zoom, theta, phi)
// around the current look-at point.
static void updateCamera()
{
    Camera& cam = renderState->camera;
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.y = zoom * cos(theta);
    cameraPosition.z = zoom * cos(phi) * sin(theta);

    cam.view = -glm::normalize(cameraPosition);
    glm::vec3 v = cam.view;
    glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
    glm::vec3 r = glm::normalize(glm::cross(v, u));
    cam.up = glm::normalize(glm::cross(r, v));
    cam.right = r;

    cameraPosition += cam.lookAt;
    cam.position = cameraPosition;
}

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();
    if (!parseCommandLine(argc, argv, guiData->settings, guiData->stats))
    {
        return 0;
    }
    guiData->stats.warmupIterations = options.warmup;

    std::string sceneArg = sceneFile;
    resuming = sceneArg.size() > 8 && sceneArg.substr(sceneArg.size() - 8) == ".ptstate";
    if (resuming)
    {
        // Restore scene, settings and accumulation from a checkpoint.
        scene = loadCheckpoint(sceneArg, resumeState);
        if (!scene)
        {
            return 1;
        }
        guiData->settings = resumeState.settings;
    }
    else
    {
        // Load scene file
        scene = new Scene(sceneFile, options.bvhOverride ? &options.bvh : nullptr);
        if (options.resX > 0)
        {
            scene->setResolution(options.resX, options.resY);
        }
    }

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    if (options.spp > 0) renderState->iterations = options.spp;
    if (options.depth > 0) renderState->traceDepth = options.depth;
    if (!options.output.empty()) renderState->imageName = options.output;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    // compute the orbit parameters of the camera around the look-at point:
    // theta is measured from +Y, phi around +Y starting at +Z
    ogLookAt = cam.lookAt;
    glm::vec3 offset = cam.position - cam.lookAt;
    zoom = glm::length(offset);
    theta = glm::acos(glm::clamp(offset.y / zoom, -1.0f, 1.0f));
    phi = glm::atan(offset.x, offset.z);
    if (resuming)
    {
        zoom = resumeState.orbit.zoom;
        theta = resumeState.orbit.theta;
        phi = resumeState.orbit.phi;
        cam.lookAt = resumeState.orbit.lookAt;
        ogLookAt = resumeState.orbit.originalLookAt;
    }

    InitImguiData(guiData);
    InitDataContainer(guiData);

    if (options.headless)
    {
        updateCamera();
        pathtraceInit(scene);
        pathtraceResetImage();
        int first = 1;
        if (resuming)
        {
            pathtraceSetAccumulation(resumeState.image, resumeState.albedo, resumeState.normal);
            first = resumeState.iteration + 1;
        }
        for (iteration = first; iteration <= (int)renderState->iterations; ++iteration)
        {
            pathtrace(NULL, 0, iteration);
            if (options.checkpointEvery > 0 && iteration % options.checkpointEvery == 0)
            {
                saveState();
            }
        }
        iteration = std::max((int)renderState->iterations, first - 1);
        saveImage();
        if (options.saveState)
        {
            saveState();
        }
        writeStats(sceneFile);
        pathtraceFree();
        return 0;
    }

    // Initialize CUDA and GL components
    init();
    pathtraceInit(scene);
    if (resuming)
    {
        updateCamera();
        camchanged = false;
        pathtraceResetImage();
        pathtraceSetAccumulation(resumeState.image, resumeState.albedo, resumeState.normal);
        iteration = resumeState.iteration;
    }

    // GLFW main loop
    mainLoop();

    pathtraceFree();
    return 0;
}

// Writes a checkpoint that can be passed back on the command line.
void saveState()
{
    if (iteration <= 0)
    {
        return;
    }
    Checkpoint cp;
    cp.iteration = iteration;
    cp.settings = guiData->settings;
    cp.orbit.zoom = zoom;
    cp.orbit.theta = theta;
    cp.orbit.phi = phi;
    cp.orbit.lookAt = renderState->camera.lookAt;
    cp.orbit.originalLookAt = ogLookAt;
    pathtraceGetAccumulation(cp.image, cp.albedo, cp.normal);
    std::ostringstream ss;
    ss << renderState->imageName << "." << iteration << "samp.ptstate";
    saveCheckpoint(ss.str(), *scene, cp);
}

void saveImage()
{
    if (iteration <= 0)
    {
        return;
    }
    float samples = iteration;
    pathtraceCopyImageToHost();

    // output image file
    Image img(width, height);
    Image raw(options.savePfm ? width : 1, options.savePfm ? height : 1);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, displayTransform(glm::vec3(pix) / samples, guiData->settings));
            if (options.savePfm)
            {
                raw.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
            }
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    if (options.headless)
    {
        // deterministic names make scripted comparisons easier
        ss << filename << "." << samples << "samp";
    }
    else
    {
        ss << filename << "." << startTimeString << "." << samples << "samp";
    }
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    if (options.savePfm)
    {
        raw.savePFM(filename);
    }

    if (guiData->settings.denoise && pathtraceDenoise(iteration))
    {
        std::vector<glm::vec3> denoised;
        pathtraceCopyDenoisedToHost(denoised);
        Image dn(width, height);
        Image dnRaw(options.savePfm ? width : 1, options.savePfm ? height : 1);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                glm::vec3 pix = denoised[x + y * width];
                dn.setPixel(width - 1 - x, y, displayTransform(pix, guiData->settings));
                if (options.savePfm)
                {
                    dnRaw.setPixel(width - 1 - x, y, pix);
                }
            }
        }
        dn.savePNG(filename + ".denoised");
        if (options.savePfm)
        {
            dnRaw.savePFM(filename + ".denoised");
        }
        printf("Denoised in %.2f ms (%s)\n", guiData->stats.lastDenoiseMs, pathtraceDenoiserName());
    }
    if (options.saveFeatures)
    {
        std::vector<glm::vec3> albedo, normal;
        pathtraceCopyFeaturesToHost(iteration, albedo, normal);
        Image a(width, height), n(width, height);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                a.setPixel(width - 1 - x, y, albedo[x + y * width]);
                n.setPixel(width - 1 - x, y, normal[x + y * width] * 0.5f + glm::vec3(0.5f));
            }
        }
        a.savePNG(filename + ".albedo");
        n.savePNG(filename + ".normal");
    }
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        updateCamera();
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        // Only the accumulation buffer needs to be cleared; the scene data on
        // the GPU stays valid when the camera moves.
        pathtraceResetImage();
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                saveImage();
                break;
            case GLFW_KEY_P:
                saveState();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
