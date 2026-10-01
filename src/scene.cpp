#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

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
static Material parseMaterial(const std::string& name, const json& p)
{
    Material m{};
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
        // TODO: handle materials loading differently
        MatNameToID[name] = materials.size();
        materials.emplace_back(parseMaterial(name, p));
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
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
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
