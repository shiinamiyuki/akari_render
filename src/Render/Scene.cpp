// MIT License
//
// Copyright (c) 2019 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <Akari/Core/Logger.h>
#include <Akari/Render/Material.h>
#include <Akari/Render/Mesh.h>
#include <Akari/Render/Plugins/AreaLight.h>
#include <Akari/Render/Scene.h>

namespace Akari {
    void Scene::Commit() {
        accelerator->Build(*this);

        for (auto &mesh : meshes) {
            auto meshLights = mesh->GetMeshLights();
            for (auto &light : meshLights)
                lights.emplace_back(light);
        }

        Info("Building distribution for {} lights\n", lights.size());
        {
            std::vector<Float> func;
            for (auto &light : lights) {
                func.emplace_back(light->Power());
            }
            lightDistribution = std::make_unique<Distribution1D>(func.data(), func.size());
            for (size_t i = 0; i < lights.size(); i++) {
                lightPdfMap[lights[i].get()] = lightDistribution->PdfDiscrete(i);
            }
        }
    }
    void Scene::GetTriangle(const PrimitiveHandle &handle, Triangle *triangle) const {
        meshes[handle.meshId]->GetTriangle(handle.primId, triangle);
    }
    const MaterialSlot &Scene::GetMaterialSlot(const PrimitiveHandle &handle) const {
        auto &mesh = meshes[handle.meshId];
        return mesh->GetMaterialSlot(mesh->GetPrimitiveGroup(handle.primId));
    }
    const Light * Scene::GetLight(const PrimitiveHandle &handle) const {
        return meshes[handle.meshId]->GetLight(handle.primId);
    }
} // namespace Akari
