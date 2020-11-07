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
#include <akari/core/memory.h>
#include <akari/core/distribution.h>
#include <akari/core/containers.h>
#include <akari/render/accelerator.h>
#include <akari/render/integrator.h>
#include <akari/render/instance.h>
#include <akari/render/sampler.h>
#include <akari/render/light.h>
namespace akari::render {
    class Accelerator;
    class MeshNode;
    class Camera;
    class Light;
    class Sampler;
    template <typename T>
    struct PointerHash {
        uint64_t operator()(const T *v) { return hash(v); }
    };
    using LightPdfMap = std::unordered_map<const Light *, Float>;
    class Scene {
      public:
        std::vector<MeshInstance> meshes;
        std::vector<std::shared_ptr<const Light>> lights;
        std::unique_ptr<Distribution1D> light_distribution;
        std::shared_ptr<Accelerator> accel = nullptr;
        std::shared_ptr<const Camera> camera = nullptr;
        std::shared_ptr<Sampler> sampler;
        LightPdfMap light_pdf_map;
        std::shared_ptr<const Light> envmap;
        std::optional<Intersection> intersect(const Ray &ray) const { return accel->intersect(ray); }
        bool occlude(const Ray &ray) const { return accel->occlude(ray); }
        Triangle get_triangle(int mesh_id, int prim_id) const {
            auto &mesh = meshes[mesh_id];
            Triangle trig = akari::render::get_triangle(mesh, prim_id);
            auto mat_idx = mesh.material_indices[prim_id];
            if (mat_idx != -1) {
                trig.material = mesh.materials[mat_idx].get();
            }
            trig.light = mesh.lights[prim_id].get();
            return trig;
        }
        std::pair<const Light *, Float> select_light(const vec2 &u) const {
            if (lights.size() == 0) {
                return {nullptr, Float(0.0f)};
            }
            Float pdf;
            size_t idx = light_distribution->sample_discrete(u[0], &pdf);
            if (idx == lights.size()) {
                idx -= 1;
            }
            return {lights[idx].get(), pdf};
        }
        Float pdf_light(const Light *light) const {
            auto it = light_pdf_map.find(light);
            if (it != light_pdf_map.end()) {
                return it->second;
            }
            return 0.0;
        }
    };
    class TextureNode;

    class AKR_EXPORT SceneNode : public SceneGraphNode {
        int spp_override = 0;

        // optix for OptixDenoiser
        // oidn for OIDNDenoiser
        std::string denoiser_name;
        int super_sampling_k = 0;
        bool required_aovs_ = false;

        astd::pmr::monotonic_buffer_resource memory_arena;

        std::shared_ptr<CameraNode> camera;
        std::shared_ptr<SamplerNode> sampler;
        std::vector<std::shared_ptr<MeshNode>> shapes;
        std::string output;

        std::shared_ptr<AcceleratorNode> accel;
        std::shared_ptr<IntegratorNode> integrator;
        std::vector<std::shared_ptr<const Light>> lights;
        std::shared_ptr<LightNode> envmap;
        std::shared_ptr<Scene> scene;

        void init_scene(Allocator<> allocator);

      public:
        AKR_SER_CLASS("Scene")
        // AKR_SER(camera, sampler, shapes, accel, envmap)
        void load(InputArchive &ar);
        void save(OutputArchive &ar) const;
        SceneNode() : memory_arena(astd::pmr::get_default_resource()) {}
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override;
        void commit() override;
        void finalize() override;
        void render();
        void set_spp(int spp) { spp_override = spp; }
        void run_denosier(const std::string & s) { denoiser_name = s; }
        void super_sample(int k) { super_sampling_k = k; }
        void require_aovs(bool v) { required_aovs_ = v; }
    };
} // namespace akari::render
