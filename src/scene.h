#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& file, const glm::mat4& transform, int materialOverride);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    RenderState state;
};
