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

#include <akari/core/logger.h>
#include <akari/render/light.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/scene.h>
namespace akari {
    void Scene::commit() {
        accelerator->build(*this);

        for (auto &mesh : meshes) {
            auto meshLights = mesh->get_mesh_lights();
            for (auto &light : meshLights)
                lights.emplace_back(light);
        }

        info("Building distribution for {} lights\n", lights.size());
        {
            std::vector<Float> func;
            for (auto &light : lights) {
                func.emplace_back(light->power());
            }
            lightDistribution = std::make_unique<Distribution1D>(func.data(), func.size());
            for (size_t i = 0; i < lights.size(); i++) {
                lightPdfMap[lights[i].get()] = lightDistribution->pdf_discrete(i);
            }
        }
    }
    const Light *Scene::sample_one_light(const Float u0, Float *pdf) const {
        if (lights.empty()) {
            return nullptr;
        }
        auto idx = lightDistribution->sample_discrete(u0, pdf);
        auto light = lights[idx].get();
        return light;
    }
    void Scene::get_triangle(const PrimitiveHandle &handle, Triangle *triangle) const {
        meshes[handle.meshId]->get_triangle(handle.primId, triangle);
    }
    const MaterialSlot &Scene::GetMaterialSlot(const PrimitiveHandle &handle) const {
        auto &mesh = meshes[handle.meshId];
        return mesh->get_material_slot(mesh->get_primitive_group(handle.primId));
    }
    const Light *Scene::GetLight(const PrimitiveHandle &handle) const {
        return meshes[handle.meshId]->get_light(handle.primId);
    }
} // namespace akari
