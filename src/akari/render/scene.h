// MIT License
//
// Copyright (c) 2020 椎名深雪
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

#pragma once
#include <akari/core/math.h>
#include <akari/core/distribution.h>
#include <akari/render/accelerator.h>
#include <akari/render/integrator.h>
#include <akari/render/instance.h>
#include <akari/render/sampler.h>
namespace akari::render {
    class Accelerator;
    class MeshNode;
    class Camera;
    class Light;
    class Sampler;
    class Scene {
      public:
        BufferView<MeshInstance> meshes;
        BufferView<const Light *> lights;
        const Distribution1D *light_distribution = nullptr;
        const Accelerator *accel = nullptr;
        const Camera *camera = nullptr;
        const Sampler * sampler = nullptr;
        std::optional<Intersection> intersect(const Ray &ray) const { return accel->intersect(ray); }
        bool occlude(const Ray &ray) const;
        Triangle get_triangle(int mesh_id, int prim_id) const {
            auto &mesh = meshes[mesh_id];
            Triangle trig = akari::render::get_triangle(mesh, prim_id);
            auto mat_idx = mesh.material_indices[prim_id];
            if (mat_idx != -1) {
                trig.material = mesh.materials[mat_idx];
            }
            return trig;
        }
    };

    class AKR_EXPORT SceneNode : public SceneGraphNode {
        std::vector<MeshInstance> instances;
        std::shared_ptr<CameraNode> camera;
        std::shared_ptr<SamplerNode> sampler;
        std::vector<std::shared_ptr<MeshNode>> shapes;
        std::string output;
        std::unique_ptr<Distribution1D> light_distribution;
        std::shared_ptr<AcceleratorNode> accel;
        std::shared_ptr<IntegratorNode> integrator;
        std::vector<const Light *> lights;
        Scene create_scene(Allocator<> *);

      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override;
        void commit();
        void render();
    };
} // namespace akari::render
