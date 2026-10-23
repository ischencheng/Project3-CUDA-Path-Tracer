#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        // meshes can use their own gltf materials
        newGeom.materialid = p.contains("MATERIAL") ? (int)MatNameToID[p["MATERIAL"]] : -1;
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (newGeom.type == MESH)
        {
            // the path is relative to the scene file
            std::string dir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            newGeom.triStart = triangles.size();
            loadGLTF(dir + p["FILE"].get<std::string>(), newGeom.transform, newGeom.materialid);
            newGeom.triCount = triangles.size() - newGeom.triStart;
            newGeom.bboxMin = glm::vec3(FLT_MAX);
            newGeom.bboxMax = glm::vec3(-FLT_MAX);
            for (int i = newGeom.triStart; i < triangles.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    newGeom.bboxMin = glm::min(newGeom.bboxMin, triangles[i].v[k]);
                    newGeom.bboxMax = glm::max(newGeom.bboxMax, triangles[i].v[k]);
                }
            }
            cout << "Loaded " << newGeom.triCount << " triangles from " << p["FILE"] << endl;
        }

        geoms.push_back(newGeom);
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
    // optional thin lens for depth of field
    camera.lensRadius = cameraData.value("APERTURE", 0.0f);
    camera.focalDistance = cameraData.value("FOCUS", glm::length(camera.lookAt - camera.position));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

static glm::mat4 nodeTransform(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        glm::mat4 m;
        for (int i = 0; i < 16; i++)
        {
            m[i / 4][i % 4] = (float)node.matrix[i];
        }
        return m;
    }
    glm::mat4 m(1.0f);
    if (node.translation.size() == 3)
    {
        m = glm::translate(m, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
    }
    if (node.rotation.size() == 4)
    {
        // gltf stores x y z w, glm::quat takes w first
        m = m * glm::mat4_cast(glm::quat((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]));
    }
    if (node.scale.size() == 3)
    {
        m = glm::scale(m, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
    }
    return m;
}

// element i of a float attribute (vec2 or vec3)
template <typename T>
static T readAttribute(const tinygltf::Model& model, int accessorIndex, int i)
{
    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const unsigned char* data = model.buffers[view.buffer].data.data()
        + view.byteOffset + accessor.byteOffset + i * accessor.ByteStride(view);
    T value;
    memcpy(&value, data, sizeof(T));
    return value;
}

static int readIndex(const tinygltf::Model& model, int accessorIndex, int i)
{
    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const unsigned char* data = model.buffers[view.buffer].data.data()
        + view.byteOffset + accessor.byteOffset + i * accessor.ByteStride(view);
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
    {
        return *data;
    }
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
    {
        return *(const uint16_t*)data;
    }
    return *(const uint32_t*)data;
}

void Scene::loadGLTF(const std::string& file, const glm::mat4& transform, int materialOverride)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ok = file.substr(file.find_last_of('.')) == ".glb"
        ? loader.LoadBinaryFromFile(&model, &err, &warn, file)
        : loader.LoadASCIIFromFile(&model, &err, &warn, file);
    if (!ok)
    {
        cout << "Couldn't load " << file << ": " << err << endl;
        exit(-1);
    }

    // one material per gltf material
    int firstMaterial = materials.size();
    for (const tinygltf::Material& m : model.materials)
    {
        Material newMaterial{};
        const auto& c = m.pbrMetallicRoughness.baseColorFactor;
        newMaterial.color = glm::vec3(c[0], c[1], c[2]);
        // gltf glass comes from the transmission extension
        auto transmission = m.extensions.find("KHR_materials_transmission");
        if (transmission != m.extensions.end() && transmission->second.Get("transmissionFactor").GetNumberAsDouble() > 0.0)
        {
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = 1.5f;
        }
        materials.push_back(newMaterial);
    }

    // walk the node tree, children get the parent transform
    std::vector<std::pair<int, glm::mat4>> stack;
    const tinygltf::Scene& gltfScene = model.scenes[model.defaultScene >= 0 ? model.defaultScene : 0];
    for (int n : gltfScene.nodes)
    {
        stack.push_back({ n, transform });
    }
    while (!stack.empty())
    {
        auto [nodeIndex, parent] = stack.back();
        stack.pop_back();
        const tinygltf::Node& node = model.nodes[nodeIndex];
        glm::mat4 world = parent * nodeTransform(node);
        for (int c : node.children)
        {
            stack.push_back({ c, world });
        }
        if (node.mesh < 0)
        {
            continue;
        }

        glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(world));
        for (const tinygltf::Primitive& prim : model.meshes[node.mesh].primitives)
        {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES)
            {
                continue;
            }
            int positions = prim.attributes.at("POSITION");
            int normals = prim.attributes.count("NORMAL") ? prim.attributes.at("NORMAL") : -1;
            int count = prim.indices >= 0 ? model.accessors[prim.indices].count : model.accessors[positions].count;
            int material = materialOverride >= 0 ? materialOverride : firstMaterial + std::max(prim.material, 0);

            for (int i = 0; i + 2 < count; i += 3)
            {
                Triangle tri;
                for (int k = 0; k < 3; k++)
                {
                    int v = prim.indices >= 0 ? readIndex(model, prim.indices, i + k) : i + k;
                    tri.v[k] = glm::vec3(world * glm::vec4(readAttribute<glm::vec3>(model, positions, v), 1.0f));
                    // zero means no normal, the flat one gets used
                    tri.n[k] = normals >= 0 ? glm::normalize(normalMatrix * readAttribute<glm::vec3>(model, normals, v)) : glm::vec3(0.0f);
                }
                tri.materialid = material;
                triangles.push_back(tri);
            }
        }
    }
}
