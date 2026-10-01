#include "meshLoader.h"

#include "scene.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstring>
#include <iostream>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace
{

// Reads any glTF accessor as floats, honoring normalized integer formats.
std::vector<float> readAccessor(const tinygltf::Model& model, int accessorIdx, int& components)
{
    const tinygltf::Accessor& acc = model.accessors[accessorIdx];
    components = tinygltf::GetNumComponentsInType(acc.type);
    std::vector<float> out((size_t)acc.count * components, 0.0f);
    if (acc.bufferView < 0)
    {
        return out;     // all zeros (sparse-only accessors are not supported)
    }
    const tinygltf::BufferView& view = model.bufferViews[acc.bufferView];
    const unsigned char* base = model.buffers[view.buffer].data.data() + view.byteOffset + acc.byteOffset;
    const int componentSize = tinygltf::GetComponentSizeInBytes(acc.componentType);
    const int stride = acc.ByteStride(view) > 0 ? acc.ByteStride(view) : componentSize * components;

    for (size_t i = 0; i < acc.count; i++)
    {
        const unsigned char* elem = base + i * stride;
        for (int c = 0; c < components; c++)
        {
            const unsigned char* p = elem + c * componentSize;
            float v = 0.0f;
            switch (acc.componentType)
            {
            case TINYGLTF_COMPONENT_TYPE_FLOAT: { float f; memcpy(&f, p, 4); v = f; break; }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: v = *p / (acc.normalized ? 255.0f : 1.0f); break;
            case TINYGLTF_COMPONENT_TYPE_BYTE: v = glm::max(*(const int8_t*)p / (acc.normalized ? 127.0f : 1.0f), -1.0f); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: { uint16_t s; memcpy(&s, p, 2); v = s / (acc.normalized ? 65535.0f : 1.0f); break; }
            case TINYGLTF_COMPONENT_TYPE_SHORT: { int16_t s; memcpy(&s, p, 2); v = glm::max(s / (acc.normalized ? 32767.0f : 1.0f), -1.0f); break; }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: { uint32_t u; memcpy(&u, p, 4); v = (float)u; break; }
            default: break;
            }
            out[i * components + c] = v;
        }
    }
    return out;
}

std::vector<uint32_t> readIndices(const tinygltf::Model& model, int accessorIdx)
{
    const tinygltf::Accessor& acc = model.accessors[accessorIdx];
    const tinygltf::BufferView& view = model.bufferViews[acc.bufferView];
    const unsigned char* base = model.buffers[view.buffer].data.data() + view.byteOffset + acc.byteOffset;
    const int componentSize = tinygltf::GetComponentSizeInBytes(acc.componentType);
    const int stride = acc.ByteStride(view) > 0 ? acc.ByteStride(view) : componentSize;
    std::vector<uint32_t> out(acc.count);
    for (size_t i = 0; i < acc.count; i++)
    {
        const unsigned char* p = base + i * stride;
        if (componentSize == 1) out[i] = *p;
        else if (componentSize == 2) { uint16_t s; memcpy(&s, p, 2); out[i] = s; }
        else { memcpy(&out[i], p, 4); }
    }
    return out;
}

glm::mat4 nodeLocalMatrix(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        glm::mat4 m;
        for (int i = 0; i < 16; i++)
        {
            glm::value_ptr(m)[i] = (float)node.matrix[i];    // column major
        }
        return m;
    }
    glm::mat4 t(1.0f), r(1.0f), s(1.0f);
    if (node.translation.size() == 3)
    {
        t[3] = glm::vec4((float)node.translation[0], (float)node.translation[1], (float)node.translation[2], 1.0f);
    }
    if (node.rotation.size() == 4)
    {
        glm::quat q((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]);
        r = glm::mat4_cast(q);
    }
    if (node.scale.size() == 3)
    {
        s[0][0] = (float)node.scale[0];
        s[1][1] = (float)node.scale[1];
        s[2][2] = (float)node.scale[2];
    }
    return t * r * s;
}

glm::vec3 toVec3(const std::vector<double>& v, glm::vec3 fallback)
{
    return v.size() >= 3 ? glm::vec3((float)v[0], (float)v[1], (float)v[2]) : fallback;
}

double extensionNumber(const tinygltf::Value& ext, const char* key, double fallback)
{
    if (ext.Has(key) && ext.Get(key).IsNumber())
    {
        return ext.Get(key).GetNumberAsDouble();
    }
    return fallback;
}

// Translates a glTF metallic-roughness material (plus the transmission, IOR,
// volume and emissive-strength extensions) into a renderer material.
Material convertMaterial(const tinygltf::Material& gm)
{
    Material m{};
    const auto& pbr = gm.pbrMetallicRoughness;
    m.type = MATERIAL_PBR;
    m.color = toVec3(pbr.baseColorFactor, glm::vec3(1.0f));
    m.metallic = (float)pbr.metallicFactor;
    m.roughness = (float)pbr.roughnessFactor;
    m.ior = 1.5f;
    m.emission = toVec3(gm.emissiveFactor, glm::vec3(0.0f));

    auto ext = gm.extensions.find("KHR_materials_emissive_strength");
    if (ext != gm.extensions.end())
    {
        m.emission *= (float)extensionNumber(ext->second, "emissiveStrength", 1.0);
    }
    ext = gm.extensions.find("KHR_materials_ior");
    if (ext != gm.extensions.end())
    {
        m.ior = (float)extensionNumber(ext->second, "ior", 1.5);
    }
    ext = gm.extensions.find("KHR_materials_transmission");
    if (ext != gm.extensions.end() && extensionNumber(ext->second, "transmissionFactor", 0.0) > 0.5)
    {
        m.type = MATERIAL_DIELECTRIC;
    }
    ext = gm.extensions.find("KHR_materials_volume");
    if (ext != gm.extensions.end())
    {
        const tinygltf::Value& v = ext->second;
        double distance = extensionNumber(v, "attenuationDistance", 0.0);
        if (distance > 0.0 && v.Has("attenuationColor"))
        {
            const tinygltf::Value& c = v.Get("attenuationColor");
            glm::vec3 color((float)c.Get(0).GetNumberAsDouble(), (float)c.Get(1).GetNumberAsDouble(),
                (float)c.Get(2).GetNumberAsDouble());
            m.absorption = -glm::log(glm::clamp(color, glm::vec3(1e-4f), glm::vec3(1.0f))) / (float)distance;
        }
    }
    return m;
}

void appendPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& prim, const glm::mat4& xform,
    const std::vector<int>& materialMap, int defaultMaterial, MeshData& mesh)
{
    if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1)
    {
        return;
    }
    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end())
    {
        return;
    }

    int comps = 0;
    std::vector<float> pos = readAccessor(model, posIt->second, comps);
    const size_t vertexCount = pos.size() / 3;
    const size_t base = mesh.positions.size();
    const glm::mat3 normalXform = glm::inverseTranspose(glm::mat3(xform));
    const bool flipWinding = glm::determinant(glm::mat3(xform)) < 0.0f;

    std::vector<float> nrm, uv, tan;
    auto it = prim.attributes.find("NORMAL");
    if (it != prim.attributes.end()) nrm = readAccessor(model, it->second, comps);
    it = prim.attributes.find("TEXCOORD_0");
    if (it != prim.attributes.end()) uv = readAccessor(model, it->second, comps);
    it = prim.attributes.find("TANGENT");
    int tanComps = 4;
    if (it != prim.attributes.end()) tan = readAccessor(model, it->second, tanComps);

    // Attribute arrays are kept parallel to positions; missing attributes are
    // zero-filled (a zero normal means "use the geometric normal").
    for (size_t i = 0; i < vertexCount; i++)
    {
        glm::vec3 p(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
        mesh.positions.push_back(glm::vec3(xform * glm::vec4(p, 1.0f)));
        glm::vec3 n(0.0f);
        if (!nrm.empty())
        {
            n = glm::normalize(normalXform * glm::vec3(nrm[3 * i], nrm[3 * i + 1], nrm[3 * i + 2]));
        }
        mesh.normals.push_back(n);
        mesh.uvs.push_back(uv.empty() ? glm::vec2(0.0f) : glm::vec2(uv[2 * i], uv[2 * i + 1]));
        glm::vec4 t(0.0f);
        if (!tan.empty() && tanComps == 4)
        {
            glm::vec3 t3 = glm::mat3(xform) * glm::vec3(tan[4 * i], tan[4 * i + 1], tan[4 * i + 2]);
            float len = glm::length(t3);
            float w = tan[4 * i + 3] * (flipWinding ? -1.0f : 1.0f);
            t = len > 0.0f ? glm::vec4(t3 / len, w < 0.0f ? -1.0f : 1.0f) : glm::vec4(0.0f);
        }
        mesh.tangents.push_back(t);
    }

    std::vector<uint32_t> idx;
    if (prim.indices >= 0)
    {
        idx = readIndices(model, prim.indices);
    }
    else
    {
        idx.resize(vertexCount);
        for (size_t i = 0; i < vertexCount; i++) idx[i] = (uint32_t)i;
    }

    int material = prim.material >= 0 ? materialMap[prim.material] : defaultMaterial;
    for (size_t i = 0; i + 2 < idx.size(); i += 3)
    {
        glm::ivec3 tri((int)(base + idx[i]), (int)(base + idx[i + 1]), (int)(base + idx[i + 2]));
        if (flipWinding)
        {
            std::swap(tri.y, tri.z);
        }
        mesh.indices.push_back(tri);
        mesh.materials.push_back(material);
    }
}

void traverseNode(const tinygltf::Model& model, int nodeIdx, const glm::mat4& parent,
    const std::vector<int>& materialMap, int defaultMaterial, MeshData& mesh)
{
    const tinygltf::Node& node = model.nodes[nodeIdx];
    glm::mat4 xform = parent * nodeLocalMatrix(node);
    if (node.mesh >= 0)
    {
        for (const auto& prim : model.meshes[node.mesh].primitives)
        {
            appendPrimitive(model, prim, xform, materialMap, defaultMaterial, mesh);
        }
    }
    for (int child : node.children)
    {
        traverseNode(model, child, xform, materialMap, defaultMaterial, mesh);
    }
}

} // namespace

bool loadGLTF(const std::string& path, Scene& scene, int defaultMaterial, MeshData& mesh)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool binary = path.size() >= 4 && path.substr(path.size() - 4) == ".glb";
    bool ok = binary ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!warn.empty()) std::cout << "glTF warning: " << warn << std::endl;
    if (!ok)
    {
        std::cout << "Failed to load glTF " << path << ": " << err << std::endl;
        return false;
    }

    std::vector<int> materialMap(model.materials.size());
    for (size_t i = 0; i < model.materials.size(); i++)
    {
        materialMap[i] = scene.addMaterial(convertMaterial(model.materials[i]));
    }

    int sceneIdx = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (model.scenes.empty())
    {
        for (size_t n = 0; n < model.nodes.size(); n++)
        {
            traverseNode(model, (int)n, glm::mat4(1.0f), materialMap, defaultMaterial, mesh);
        }
    }
    else
    {
        for (int n : model.scenes[sceneIdx].nodes)
        {
            traverseNode(model, n, glm::mat4(1.0f), materialMap, defaultMaterial, mesh);
        }
    }

    bool hasTangents = false;
    for (const glm::vec4& t : mesh.tangents)
    {
        if (t.w != 0.0f) { hasTangents = true; break; }
    }
    if (!hasTangents)
    {
        computeTangents(mesh);
    }
    return !mesh.indices.empty();
}

bool loadOBJ(const std::string& path, Scene& scene, int defaultMaterial, MeshData& mesh)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string err;
    std::string dir = path.substr(0, path.find_last_of("/\\") + 1);
    bool ok = tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &err, path.c_str(), dir.c_str(), true);
    if (!err.empty()) std::cout << "OBJ: " << err << std::endl;
    if (!ok)
    {
        return false;
    }

    std::vector<int> materialMap(objMaterials.size());
    for (size_t i = 0; i < objMaterials.size(); i++)
    {
        const tinyobj::material_t& om = objMaterials[i];
        Material m{};
        m.type = MATERIAL_PBR;
        m.color = glm::vec3(om.diffuse[0], om.diffuse[1], om.diffuse[2]);
        m.roughness = om.roughness > 0.0f ? om.roughness : glm::clamp(sqrtf(2.0f / (om.shininess + 2.0f)), 0.0f, 1.0f);
        m.metallic = om.metallic;
        m.ior = om.ior > 1.0f ? om.ior : 1.5f;
        m.emission = glm::vec3(om.emission[0], om.emission[1], om.emission[2]);
        materialMap[i] = scene.addMaterial(m);
    }

    // OBJ indexes positions, normals and UVs separately, so every corner
    // becomes its own vertex.
    bool hasNormals = !attrib.normals.empty();
    for (const auto& shape : shapes)
    {
        const auto& indices = shape.mesh.indices;
        for (size_t f = 0; f + 2 < indices.size(); f += 3)
        {
            int base = (int)mesh.positions.size();
            for (int k = 0; k < 3; k++)
            {
                const tinyobj::index_t& ix = indices[f + k];
                mesh.positions.push_back(glm::vec3(attrib.vertices[3 * ix.vertex_index],
                    attrib.vertices[3 * ix.vertex_index + 1], attrib.vertices[3 * ix.vertex_index + 2]));
                glm::vec3 n(0.0f);
                if (hasNormals && ix.normal_index >= 0)
                {
                    n = glm::vec3(attrib.normals[3 * ix.normal_index], attrib.normals[3 * ix.normal_index + 1],
                        attrib.normals[3 * ix.normal_index + 2]);
                }
                mesh.normals.push_back(n);
                glm::vec2 uv(0.0f);
                if (ix.texcoord_index >= 0)
                {
                    uv = glm::vec2(attrib.texcoords[2 * ix.texcoord_index], attrib.texcoords[2 * ix.texcoord_index + 1]);
                }
                mesh.uvs.push_back(uv);
                mesh.tangents.push_back(glm::vec4(0.0f));
            }
            mesh.indices.push_back(glm::ivec3(base, base + 1, base + 2));
            int matId = f / 3 < shape.mesh.material_ids.size() ? shape.mesh.material_ids[f / 3] : -1;
            mesh.materials.push_back(matId >= 0 ? materialMap[matId] : defaultMaterial);
        }
    }
    computeTangents(mesh);
    return !mesh.indices.empty();
}

void computeTangents(MeshData& mesh)
{
    const size_t n = mesh.positions.size();
    if (mesh.uvs.size() != n || mesh.normals.size() != n)
    {
        return;
    }
    std::vector<glm::vec3> tan(n, glm::vec3(0.0f));
    std::vector<glm::vec3> bitan(n, glm::vec3(0.0f));
    for (const glm::ivec3& tri : mesh.indices)
    {
        glm::vec3 p0 = mesh.positions[tri.x], p1 = mesh.positions[tri.y], p2 = mesh.positions[tri.z];
        glm::vec2 w0 = mesh.uvs[tri.x], w1 = mesh.uvs[tri.y], w2 = mesh.uvs[tri.z];
        glm::vec3 e1 = p1 - p0, e2 = p2 - p0;
        glm::vec2 d1 = w1 - w0, d2 = w2 - w0;
        float det = d1.x * d2.y - d2.x * d1.y;
        if (fabsf(det) < 1e-12f)
        {
            continue;
        }
        float r = 1.0f / det;
        glm::vec3 sdir = (e1 * d2.y - e2 * d1.y) * r;
        glm::vec3 tdir = (e2 * d1.x - e1 * d2.x) * r;
        for (int k = 0; k < 3; k++)
        {
            tan[tri[k]] += sdir;
            bitan[tri[k]] += tdir;
        }
    }
    mesh.tangents.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        glm::vec3 nrm = mesh.normals[i];
        glm::vec3 t = tan[i] - nrm * glm::dot(nrm, tan[i]);
        float len = glm::length(t);
        if (len < 1e-12f || glm::dot(nrm, nrm) == 0.0f)
        {
            mesh.tangents[i] = glm::vec4(0.0f);
            continue;
        }
        t /= len;
        float w = glm::dot(glm::cross(nrm, t), bitan[i]) < 0.0f ? -1.0f : 1.0f;
        mesh.tangents[i] = glm::vec4(t, w);
    }
}
