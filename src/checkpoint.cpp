#include "checkpoint.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

namespace
{

const char MAGIC[8] = { 'C', 'I', 'S', '5', '6', '5', '0', 'P' };
const uint32_t VERSION = 1;

// All serialized types are plain structs of floats and ints (the bundled GLM
// declares copy constructors, so they are not formally trivially copyable,
// but they are safe to copy byte-wise).
class Writer
{
public:
    explicit Writer(std::ofstream& out) : out(out) {}

    template <typename T>
    void pod(const T& value)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    void vec(const std::vector<T>& v)
    {
        uint64_t n = v.size();
        pod(n);
        if (n)
        {
            out.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
        }
    }

    void str(const std::string& s)
    {
        uint64_t n = s.size();
        pod(n);
        out.write(s.data(), n);
    }

private:
    std::ofstream& out;
};

class Reader
{
public:
    explicit Reader(std::ifstream& in) : in(in) {}

    template <typename T>
    void pod(T& value)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    template <typename T>
    void vec(std::vector<T>& v)
    {
        uint64_t n = 0;
        pod(n);
        if (!in || n > (1ull << 34) / sizeof(T))
        {
            in.setstate(std::ios::failbit);
            return;
        }
        v.resize(n);
        if (n)
        {
            in.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
        }
    }

    void str(std::string& s)
    {
        uint64_t n = 0;
        pod(n);
        if (!in || n > (1u << 20))
        {
            in.setstate(std::ios::failbit);
            return;
        }
        s.resize(n);
        in.read(&s[0], n);
    }

private:
    std::ifstream& in;
};

// The scene part is written and read by the same template so that the two
// directions cannot drift apart.
template <typename IO, typename SceneT>
void serializeScene(IO& io, SceneT& scene)
{
    io.str(scene.sceneFile);
    io.str(scene.sceneDir);
    io.pod(scene.state.camera);
    io.pod(scene.state.iterations);
    io.pod(scene.state.traceDepth);
    io.str(scene.state.imageName);
    io.pod(scene.state.backgroundColor);

    io.vec(scene.geoms);
    io.vec(scene.materials);
    io.pod(scene.bvhSettings);
    io.vec(scene.meshes);
    io.vec(scene.bvhNodes);
    io.vec(scene.triGeoms);
    io.vec(scene.triangles);
    io.vec(scene.vertexNormals);
    io.vec(scene.vertexUVs);
    io.vec(scene.vertexTangents);

    io.vec(scene.lights);
    io.vec(scene.lightCdf);
    io.vec(scene.triLightIndex);
    io.pod(scene.envSampleProb);

    auto& env = scene.envMap;
    io.pod(env.width);
    io.pod(env.height);
    io.vec(env.rgba);
    io.vec(env.func);
    io.vec(env.marginalCdf);
    io.vec(env.conditionalCdf);
    io.pod(env.integral);
    io.pod(env.intensity);
    io.pod(env.rotation);
}

} // namespace

bool saveCheckpoint(const std::string& path, const Scene& scene, const Checkpoint& checkpoint)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        std::cout << "Couldn't write checkpoint " << path << std::endl;
        return false;
    }
    Writer w(out);
    out.write(MAGIC, sizeof(MAGIC));
    w.pod(VERSION);

    serializeScene(w, scene);
    uint64_t textureCount = scene.textures.size();
    w.pod(textureCount);
    for (const TextureData& t : scene.textures)
    {
        w.pod(t.width);
        w.pod(t.height);
        w.str(t.name);
        w.vec(t.rgba);
    }

    w.pod(checkpoint.iteration);
    w.pod(checkpoint.settings);
    w.pod(checkpoint.orbit);
    w.vec(checkpoint.image);
    w.vec(checkpoint.albedo);
    w.vec(checkpoint.normal);
    bool ok = (bool)out;
    std::cout << (ok ? "Saved checkpoint " : "Failed to write checkpoint ") << path
              << " (" << checkpoint.iteration << " iterations)" << std::endl;
    return ok;
}

Scene* loadCheckpoint(const std::string& path, Checkpoint& checkpoint)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::cout << "Couldn't open checkpoint " << path << std::endl;
        return nullptr;
    }
    char magic[8];
    in.read(magic, sizeof(magic));
    uint32_t version = 0;
    Reader r(in);
    r.pod(version);
    if (!in || memcmp(magic, MAGIC, sizeof(MAGIC)) != 0 || version != VERSION)
    {
        std::cout << path << " is not a compatible checkpoint" << std::endl;
        return nullptr;
    }

    Scene* scene = new Scene();
    serializeScene(r, *scene);
    uint64_t textureCount = 0;
    r.pod(textureCount);
    for (uint64_t i = 0; i < textureCount && in; i++)
    {
        TextureData t;
        r.pod(t.width);
        r.pod(t.height);
        r.str(t.name);
        r.vec(t.rgba);
        scene->textures.push_back(std::move(t));
    }

    r.pod(checkpoint.iteration);
    r.pod(checkpoint.settings);
    r.pod(checkpoint.orbit);
    r.vec(checkpoint.image);
    r.vec(checkpoint.albedo);
    r.vec(checkpoint.normal);
    if (!in)
    {
        std::cout << "Checkpoint " << path << " is truncated" << std::endl;
        delete scene;
        return nullptr;
    }
    const Camera& cam = scene->state.camera;
    scene->state.image.assign((size_t)cam.resolution.x * cam.resolution.y, glm::vec3(0.0f));
    std::cout << "Restored checkpoint " << path << ": " << scene->triangles.size() << " triangles, "
              << scene->textures.size() << " textures, " << checkpoint.iteration << " iterations" << std::endl;
    return scene;
}
