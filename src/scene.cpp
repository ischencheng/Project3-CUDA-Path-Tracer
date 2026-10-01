#include "scene.h"

#include "meshLoader.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <stb_image.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

static glm::vec3 readVec3(const json& j, const char* key, glm::vec3 fallback)
{
    if (!j.contains(key))
    {
        return fallback;
    }
    const auto& v = j[key];
    if (v.is_number())
    {
        return glm::vec3(v.get<float>());
    }
    return glm::vec3(v[0], v[1], v[2]);
}

// Material description in the scene file:
//   "TYPE": Diffuse | Specular (Metal) | Refractive (Dielectric, Glass) | PBR | Emitting
//   "RGB": albedo / specular color / glass tint / base color
//   "ROUGHNESS", "METALLIC", "IOR" (default 1.5)
//   "EMITTANCE" scales "RGB" for Emitting materials; other materials may set
//   "EMISSIVE": [r,g,b] and "EMISSIVE_STRENGTH" to glow and still scatter.
//   Glass absorption: "ABSORPTION": [r,g,b] per unit length, or
//   "ATTENUATION_COLOR" + "ATTENUATION_DISTANCE" (glTF KHR_materials_volume).
static int procedurals(const std::string& name)
{
    if (name == "checker") return PROCEDURAL_CHECKER;
    if (name == "marble") return PROCEDURAL_MARBLE;
    if (name == "wood") return PROCEDURAL_WOOD;
    if (name == "noise") return PROCEDURAL_NOISE;
    cout << "Unknown procedural texture " << name << endl;
    return PROCEDURAL_NONE;
}

// Texture related material fields:
//   "TEXTURE": base color image, "NORMAL_MAP": tangent space normal map,
//   "UV_SCALE": [su, sv] tiling,
//   "PROCEDURAL": { "TYPE": checker|marble|wood|noise, "COLOR2": [...], "SCALE": s },
//   "BUMP": { "STRENGTH": a, "SCALE": s }  procedural bump mapping
static void parseTextures(Scene& scene, Material& m, const json& p)
{
    if (p.contains("TEXTURE"))
    {
        m.baseColorTex = scene.loadTextureFile(p["TEXTURE"]);
    }
    if (p.contains("NORMAL_MAP"))
    {
        m.normalTex = scene.loadTextureFile(p["NORMAL_MAP"]);
    }
    m.normalScale = p.value("NORMAL_SCALE", 1.0f);
    if (p.contains("UV_SCALE"))
    {
        const auto& s = p["UV_SCALE"];
        m.uvScale = s.is_number() ? glm::vec2(s.get<float>()) : glm::vec2(s[0], s[1]);
    }
    if (p.contains("PROCEDURAL"))
    {
        const auto& proc = p["PROCEDURAL"];
        m.procedural = procedurals(proc.value("TYPE", std::string("checker")));
        m.procColor2 = readVec3(proc, "COLOR2", glm::vec3(0.1f));
        m.procScale = proc.value("SCALE", 1.0f);
    }
    if (p.contains("BUMP"))
    {
        const auto& bump = p["BUMP"];
        m.bumpStrength = bump.value("STRENGTH", 0.1f);
        m.bumpScale = bump.value("SCALE", 1.0f);
    }
}

static Material parseMaterial(const std::string& name, const json& p)
{
    Material m = makeDefaultMaterial();
    m.color = readVec3(p, "RGB", glm::vec3(1.0f));
    m.roughness = glm::clamp(p.value("ROUGHNESS", 0.0f), 0.0f, 1.0f);
    m.metallic = glm::clamp(p.value("METALLIC", 0.0f), 0.0f, 1.0f);
    m.ior = p.value("IOR", 1.5f);
    m.emission = readVec3(p, "EMISSIVE", glm::vec3(0.0f)) * p.value("EMISSIVE_STRENGTH", 1.0f);
    m.absorption = readVec3(p, "ABSORPTION", glm::vec3(0.0f));
    if (p.contains("ATTENUATION_COLOR"))
    {
        glm::vec3 c = glm::clamp(readVec3(p, "ATTENUATION_COLOR", glm::vec3(1.0f)), glm::vec3(1e-4f), glm::vec3(1.0f));
        float d = p.value("ATTENUATION_DISTANCE", 1.0f);
        m.absorption = -glm::log(c) / d;
    }

    const std::string type = p.value("TYPE", std::string("Diffuse"));
    if (type == "Diffuse")
    {
        m.type = MATERIAL_DIFFUSE;
    }
    else if (type == "Emitting" || type == "Light")
    {
        m.type = MATERIAL_EMITTING;
        m.emission = m.color * p.value("EMITTANCE", 1.0f);
    }
    else if (type == "Specular" || type == "Metal" || type == "Conductor" || type == "Mirror")
    {
        m.type = MATERIAL_SPECULAR;
    }
    else if (type == "Refractive" || type == "Dielectric" || type == "Glass")
    {
        m.type = MATERIAL_DIELECTRIC;
    }
    else if (type == "PBR" || type == "Principled")
    {
        m.type = MATERIAL_PBR;
    }
    else
    {
        cout << "Unknown material type " << type << " for " << name << ", using Diffuse" << endl;
        m.type = MATERIAL_DIFFUSE;
    }
    return m;
}

Scene::Scene(string filename, const BVHBuildSettings* bvhOverride)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    sceneFile = filename;
    size_t slash = filename.find_last_of("/\\");
    sceneDir = slash == string::npos ? string() : filename.substr(0, slash + 1);
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename, bvhOverride);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

int Scene::addMaterial(const Material& m)
{
    materials.push_back(m);
    return (int)materials.size() - 1;
}

int Scene::addTexture(TextureData&& texture)
{
    textures.push_back(std::move(texture));
    return (int)textures.size() - 1;
}

int Scene::loadTextureFile(const std::string& file)
{
    std::string path = sceneDir + file;
    auto cached = textureCache.find(path);
    if (cached != textureCache.end())
    {
        return cached->second;
    }
    int w, h, comp;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!pixels)
    {
        cout << "Couldn't load texture " << path << ": " << stbi_failure_reason() << endl;
        return -1;
    }
    TextureData tex;
    tex.width = w;
    tex.height = h;
    tex.rgba.assign(pixels, pixels + (size_t)w * h * 4);
    tex.name = file;
    stbi_image_free(pixels);
    cout << "Loaded texture " << path << " (" << w << "x" << h << ")" << endl;
    int id = addTexture(std::move(tex));
    textureCache[path] = id;
    return id;
}

int Scene::addMesh(const MeshData& mesh)
{
    const int triCount = (int)mesh.indices.size();
    const int vertexOffset = (int)vertexNormals.size();

    // Per-triangle bounds and centroids drive the SAH build.
    std::vector<AABB> bounds(triCount);
    std::vector<glm::vec3> centroids(triCount);
    for (int i = 0; i < triCount; i++)
    {
        const glm::ivec3& t = mesh.indices[i];
        AABB b = AABB::empty();
        b.grow(mesh.positions[t.x]);
        b.grow(mesh.positions[t.y]);
        b.grow(mesh.positions[t.z]);
        bounds[i] = b;
        centroids[i] = (mesh.positions[t.x] + mesh.positions[t.y] + mesh.positions[t.z]) / 3.0f;
    }

    std::vector<BVHNode> nodes;
    std::vector<int> order;
    BVHBuildStats stats = buildBVH(bounds, centroids, bvhSettings, nodes, order);
    totalBvhBuildMs += stats.buildMs;

    MeshInfo info;
    info.triOffset = (int)triangles.size();
    info.triCount = triCount;
    info.rootNode = (int)bvhNodes.size();
    info.nodeCount = (int)nodes.size();
    info.bounds = AABB::empty();
    for (const glm::vec3& p : mesh.positions)
    {
        info.bounds.grow(p);
    }

    // Store triangles in BVH leaf order so leaves reference contiguous ranges.
    for (int i = 0; i < triCount; i++)
    {
        const glm::ivec3& t = mesh.indices[order[i]];
        glm::vec3 v0 = mesh.positions[t.x];
        glm::vec3 e1 = mesh.positions[t.y] - v0;
        glm::vec3 e2 = mesh.positions[t.z] - v0;
        TriangleGeom g;
        g.v0 = make_float4(v0.x, v0.y, v0.z, 0.0f);
        g.e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
        g.e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
        triGeoms.push_back(g);
        Triangle tri;
        tri.v[0] = vertexOffset + t.x;
        tri.v[1] = vertexOffset + t.y;
        tri.v[2] = vertexOffset + t.z;
        tri.materialId = mesh.materials[order[i]];
        triangles.push_back(tri);
    }
    for (BVHNode n : nodes)
    {
        n.leftFirst += n.triCount > 0 ? info.triOffset : info.rootNode;
        bvhNodes.push_back(n);
    }

    const size_t vertexCount = mesh.positions.size();
    for (size_t i = 0; i < vertexCount; i++)
    {
        vertexNormals.push_back(i < mesh.normals.size() ? mesh.normals[i] : glm::vec3(0.0f));
        vertexUVs.push_back(i < mesh.uvs.size() ? mesh.uvs[i] : glm::vec2(0.0f));
        vertexTangents.push_back(i < mesh.tangents.size() ? mesh.tangents[i] : glm::vec4(0.0f));
    }

    cout << "  bounds (" << info.bounds.min.x << ", " << info.bounds.min.y << ", " << info.bounds.min.z << ") - ("
         << info.bounds.max.x << ", " << info.bounds.max.y << ", " << info.bounds.max.z << ")" << endl;
    cout << "  BVH: " << triCount << " triangles, " << stats.nodeCount << " nodes, " << stats.leafCount
         << " leaves, depth " << stats.depth << ", largest leaf " << stats.maxLeafTriangles
         << ", built in " << stats.buildMs << " ms" << endl;

    meshes.push_back(info);
    return (int)meshes.size() - 1;
}

int Scene::loadMesh(const std::string& file, int defaultMaterial)
{
    std::string path = sceneDir + file;
    std::string key = path + "#" + std::to_string(defaultMaterial);
    auto cached = meshCache.find(key);
    if (cached != meshCache.end())
    {
        return cached->second;
    }

    MeshData mesh;
    std::string ext = path.substr(path.find_last_of('.') + 1);
    for (char& c : ext) c = (char)tolower(c);
    cout << "Loading mesh " << path << endl;
    bool ok = ext == "obj" ? loadOBJ(path, *this, defaultMaterial, mesh)
        : loadGLTF(path, *this, defaultMaterial, mesh);
    if (!ok)
    {
        cout << "Couldn't load mesh " << path << endl;
        exit(-1);
    }
    int id = addMesh(mesh);
    meshCache[key] = id;
    return id;
}

// World-space bounds of the transformed object-space bounds.
void Scene::computeWorldBounds(Geom& geom) const
{
    AABB local;
    if (geom.type == MESH)
    {
        local = meshes[geom.meshId].bounds;
    }
    else
    {
        local.min = glm::vec3(-0.5f);
        local.max = glm::vec3(0.5f);
    }
    AABB world = AABB::empty();
    for (int c = 0; c < 8; c++)
    {
        glm::vec3 corner((c & 1) ? local.max.x : local.min.x,
            (c & 2) ? local.max.y : local.min.y,
            (c & 4) ? local.max.z : local.min.z);
        world.grow(glm::vec3(geom.transform * glm::vec4(corner, 1.0f)));
    }
    // pad slightly so flat objects still have a usable box
    world.min -= glm::vec3(1e-4f);
    world.max += glm::vec3(1e-4f);
    geom.worldBounds = world;
}

void Scene::loadFromJSON(const std::string& jsonName, const BVHBuildSettings* bvhOverride)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        // TODO: handle materials loading differently
        MatNameToID[name] = materials.size();
        Material m = parseMaterial(name, p);
        parseTextures(*this, m, p);
        materials.emplace_back(m);
    }

    if (data.contains("BVH"))
    {
        const auto& b = data["BVH"];
        bvhSettings.maxLeafSize = b.value("MAX_LEAF_SIZE", bvhSettings.maxLeafSize);
        bvhSettings.maxDepth = b.value("MAX_DEPTH", bvhSettings.maxDepth);
        bvhSettings.numBins = b.value("BINS", bvhSettings.numBins);
    }
    if (bvhOverride != nullptr)
    {
        bvhSettings = *bvhOverride;    // command line wins over the scene file
    }

    // Fallback material for meshes whose primitives do not specify one.
    Material defaultMat = makeDefaultMaterial();
    int defaultMaterial = -1;

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom{};
        newGeom.meshId = -1;
        newGeom.materialid = -1;
        if (p.contains("MATERIAL"))
        {
            auto it = MatNameToID.find(p["MATERIAL"]);
            if (it == MatNameToID.end())
            {
                cout << "Unknown material " << p["MATERIAL"] << endl;
                exit(-1);
            }
            newGeom.materialid = it->second;
        }
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
            if (defaultMaterial < 0)
            {
                defaultMaterial = addMaterial(defaultMat);
            }
            newGeom.meshId = loadMesh(p["FILE"], defaultMaterial);
        }
        else
        {
            newGeom.type = SPHERE;
        }
        if (newGeom.type != MESH && newGeom.materialid < 0)
        {
            cout << "Object of type " << type << " needs a MATERIAL" << endl;
            exit(-1);
        }
        newGeom.translation = readVec3(p, "TRANS", glm::vec3(0.0f));
        newGeom.rotation = readVec3(p, "ROTAT", glm::vec3(0.0f));
        newGeom.scale = readVec3(p, "SCALE", glm::vec3(1.0f));
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        computeWorldBounds(newGeom);

        geoms.push_back(newGeom);
    }
    if (!triangles.empty())
    {
        cout << "Scene has " << triangles.size() << " triangles in " << meshes.size() << " meshes, "
             << bvhNodes.size() << " BVH nodes (" << totalBvhBuildMs << " ms)" << endl;
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    // The view direction has to be known before the right vector is derived.
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // Thin lens: "APERTURE" is the lens radius, "FOCAL_DISTANCE" defaults to
    // the distance to the look-at point.
    camera.lensRadius = cameraData.value("APERTURE", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", glm::length(camera.lookAt - camera.position));

    state.backgroundColor = glm::vec3(0.0f);
    if (data.contains("Environment"))
    {
        const auto& env = data["Environment"];
        state.backgroundColor = readVec3(env, "COLOR", glm::vec3(0.0f)) * env.value("INTENSITY", 1.0f);
    }

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::setResolution(int width, int height)
{
    Camera& camera = state.camera;
    float yscaled = 0.5f * camera.pixelLength.y * camera.resolution.y;
    camera.resolution = glm::ivec2(width, height);
    float xscaled = (yscaled * width) / height;
    camera.fov.x = (atan(xscaled) * 180) / PI;
    camera.pixelLength = glm::vec2(2 * xscaled / (float)width, 2 * yscaled / (float)height);
    state.image.assign(width * height, glm::vec3(0.0f));
}
