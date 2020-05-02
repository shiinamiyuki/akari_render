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

#ifndef AKARIRENDER_SCENE_H
#define AKARIRENDER_SCENE_H

#include "Material.h"
#include <akari/Core/Distribution.hpp>
#include <akari/Render/Accelerator.h>
#include <akari/Render/Geometry.hpp>
#include <atomic>
namespace akari {
    class Mesh;
    class Material;
    class Acceleator;
    class Light;
    struct MaterialSlot;
    class AKR_EXPORT Scene {
        std::vector<std::shared_ptr<const Mesh>> meshes;
        std::shared_ptr<Accelerator> accelerator;
        mutable std::atomic<size_t> rayCounter;
        std::unique_ptr<Distribution1D> lightDistribution;
        std::vector<std::shared_ptr<Light>> lights;
        std::unordered_map<const Light *, Float> lightPdfMap;

      public:
        void AddLight(const std::shared_ptr<Light>& light){lights.emplace_back(light);}
        void AddMesh(const std::shared_ptr<const Mesh> &mesh) { meshes.emplace_back(mesh); }
        [[nodiscard]] const std::vector<std::shared_ptr<const Mesh>> &GetMeshes() const { return meshes; }
        void SetAccelerator(std::shared_ptr<Accelerator> p) { accelerator = std::move(p); }
        void Commit();
        void ClearRayCounter() const { rayCounter = 0; }
        size_t GetRayCounter() const { return rayCounter.load(); }
        bool intersect(const Ray &ray, Intersection *intersection) const {
            rayCounter++;
            if (!accelerator->intersect(ray, intersection)) {
                return false;
            }
            intersection->p = ray.o + intersection->t * ray.d;
            return true;
        }
        bool Occlude(const Ray &ray) const {
            rayCounter++;
            return accelerator->occlude(ray);
        }
        void get_triangle(const PrimitiveHandle & handle, Triangle * triangle)const;
        const MaterialSlot& GetMaterialSlot(const PrimitiveHandle & handle)const;
        const Light * GetLight(const PrimitiveHandle & handle)const;
        const Mesh &get_mesh(const uint32_t id) const { return *meshes[id]; }
        const Light *sample_one_light(const Float u0, Float *pdf) const ;
        Float PdfLight(const Light *light) const {
            auto it = lightPdfMap.find(light);
            if (it == lightPdfMap.end())
                return 0.0f;
            return it->second;
        }

        Bounds3f GetBounds() const { return accelerator->bounds(); }
    };
} // namespace akari
#endif // AKARIRENDER_SCENE_H
