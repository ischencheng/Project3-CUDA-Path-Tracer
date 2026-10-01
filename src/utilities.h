#pragma once

#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f
#define RAY_EPSILON       0.0005f   // offset applied to spawned ray origins

#define MAX_TRACKED_DEPTH 64

enum ToneMapMode
{
    TONEMAP_NONE = 0,   // linear clamp, matches the reference Cornell render
    TONEMAP_REINHARD,
    TONEMAP_ACES,
    TONEMAP_COUNT
};

// Settings that the renderer reads every iteration. Most of them are exposed
// through ImGui and the command line so that features can be toggled for
// performance comparisons.
struct RenderSettings
{
    bool sortByMaterial = false;
    bool streamCompaction = true;
    bool antialiasing = true;

    // display / output
    int toneMap = TONEMAP_NONE;
    bool gammaCorrect = false;
    float exposure = 1.0f;
};

enum RenderStage
{
    STAGE_GENERATE = 0,
    STAGE_INTERSECT,
    STAGE_SORT,
    STAGE_SHADE,
    STAGE_COMPACT,
    STAGE_GATHER,
    STAGE_COUNT
};

// Statistics measured on the GPU with CUDA events, accumulated since the last
// reset of the accumulation buffer. Per-stage timing inserts a synchronization
// after every stage, so it is only collected when `profileStages` is set.
struct RenderStats
{
    bool profileStages = false;
    int warmupIterations = 0;       // iterations ignored by the accumulators
    int iterationsSeen = 0;
    int sampledIterations = 0;
    float lastIterationMs = 0.0f;
    double totalIterationMs = 0.0;
    double stageMs[STAGE_COUNT] = {};
    // number of paths still alive at the start of each bounce
    double alivePaths[MAX_TRACKED_DEPTH] = {};
    int numBounces = 0;

    void reset()
    {
        bool p = profileStages;
        int w = warmupIterations;
        *this = RenderStats();
        profileStages = p;
        warmupIterations = w;
    }

    float avgIterationMs() const
    {
        return sampledIterations ? float(totalIterationMs / sampledIterations) : 0.0f;
    }
};

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
    RenderSettings settings;
    RenderStats stats;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
