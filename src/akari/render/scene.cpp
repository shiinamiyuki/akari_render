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

#include <csignal>
#include <akari/core/profiler.h>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/light.h>
namespace akari::render {
    void SceneNode::commit() {
        for (auto &shape : shapes) {
            AKR_ASSERT_THROW(shape);
            shape->commit();
        }
        AKR_ASSERT_THROW(camera);
        camera->commit();
    }
    void SceneNode::init_scene(Allocator<> *allocator) {
        lights.clear();
        light_id_map.clear();
        light_pdf_map.clear();
        scene.camera = camera->create_camera(allocator);
        for (auto &shape : shapes) {
            instances.emplace_back(shape->create_instance(allocator));
        }
        scene.meshes = {instances.data(), instances.size()};
        scene.accel = accel->create_accel(scene);
        std::vector<const AreaLight *> area_lights;
        std::vector<Float> power;
        std::unordered_map<const Texture *, std::future<Float>> ft_integrals;
        std::unordered_map<const Texture *, Float> integrals;
        for (uint32_t mesh_id = 0; mesh_id < scene.meshes.size(); mesh_id++) {
            MeshInstance &mesh = scene.meshes[mesh_id];
            for (uint32_t prim_id = 0; prim_id < mesh.indices.size() / 3; prim_id++) {
                auto triangle = scene.get_triangle(mesh_id, prim_id);
                auto material = triangle.material;
                if (!material)
                    continue;
                if (material->is_emissive()) {
                    const EmissiveMaterial *e = material->as_emissive();
                    auto color = e->color;
                    if (ft_integrals.find(color) == ft_integrals.end()) {
                        ft_integrals.emplace(color, std::async(std::launch::async, [=] { return color->integral(); }));
                    }
                    (void)e;
                    auto light = allocator->new_object<AreaLight>(triangle);
                    area_lights.emplace_back(light);
                    light_id_map.emplace(std::make_pair(mesh_id, prim_id), light);
                }
            }
        }
        const Texture *envmap_texture = nullptr;
        if (envmap) {
            envmap_texture = envmap->envmap->create_texture(allocator);
            AKR_ASSERT_THROW(envmap_texture);
            if (ft_integrals.find(envmap_texture) == ft_integrals.end()) {
                ft_integrals.emplace(envmap_texture,
                                     std::async(std::launch::async, [=] { return envmap_texture->integral(); }));
            }
        }
        for (auto &pair : ft_integrals) {
            integrals[pair.first] = pair.second.get();
        }
        for (size_t i = 0; i < area_lights.size(); i++) {
            auto light = area_lights[i];
            auto color = light->color;
            auto I = integrals[color];
            auto &triangle = light->triangle;
            Vec3 tc[3];
            for (int j = 0; j < 3; j++)
                tc[j] = Vec3(triangle.texcoords[j].x, triangle.texcoords[j].y, 0.0);
            auto tc_area = length(cross(tc[1] - tc[0], tc[2] - tc[0])) * 0.5;
            auto area = glm::length(
                glm::cross(triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0]));
            power.emplace_back(area * tc_area * I);
        }
        if (envmap) {
            scene.envmap = InfiniteAreaLight::create(scene, envmap->transform, envmap_texture);
            power.emplace_back(scene.envmap->power());
        }
        for (auto i : area_lights) {
            lights.emplace_back(i);
        }
        if (envmap) {
            lights.emplace_back(scene.envmap.get());
        }
        light_distribution = std::make_unique<Distribution1D>(power.data(), power.size());
        AKR_ASSERT(lights.size() == power.size());
        for (size_t i = 0; i < lights.size(); i++) {
            light_pdf_map[lights[i]] = light_distribution->pdf_discrete(i);
        }
        scene.light_distribution = light_distribution.get();
        scene.lights = BufferView(lights.data(), lights.size());
        scene.light_id_map = &light_id_map;
        scene.light_pdf_map = &light_pdf_map;
    }
    void SceneNode::render() {
        // Thanks to python hijacking SIGINT handler;
        /* We want to restore the SIGINT handler so that the user can interrupt the renderer */
        auto _prev_SIGINT_handler = signal(SIGINT, SIG_DFL);
        auto _restore_handler = AtScopeExit([=]() { signal(SIGINT, _prev_SIGINT_handler); });
        commit();
        if (spp_override > 0) {
            if (!integrator->set_spp(spp_override)) {
                warning("cannot override spp");
            }
        }
        info("preparing scene");
        astd::pmr::monotonic_buffer_resource resource(astd::pmr::get_default_resource());
        Allocator<> allocator(&resource);
        init_scene(&allocator);
        scene.sampler = sampler->create_sampler(&allocator);
        auto real_integrator = integrator->create_integrator(&allocator);
        auto res = scene.camera->resolution();
        auto film = Film(res);

        Timer timer;
        real_integrator->render(&scene, &film);
        info("render done ({}s)", timer.elapsed_seconds());
        film.write_image(fs::path(output));
    }

    void SceneNode::object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                                 const sdl::Value &value) {
        if (field == "camera") {
            camera = dyn_cast<CameraNode>(value.object());
            AKR_ASSERT_THROW(camera);
        } else if (field == "output") {
            output = value.get<std::string>().value();
        } else if (field == "integrator") {
            integrator = dyn_cast<IntegratorNode>(value.object());
            AKR_ASSERT_THROW(integrator);
        } else if (field == "shapes") {
            AKR_ASSERT_THROW(value.is_array());
            for (auto shape : value) {
                shapes.emplace_back(dyn_cast<MeshNode>(shape.object()));
            }
        } else if (field == "sampler") {
            sampler = dyn_cast<SamplerNode>(value.object());
            AKR_ASSERT_THROW(sampler);
        } else if (field == "accelerator") {
            accel = dyn_cast<AcceleratorNode>(value.object());
            AKR_ASSERT_THROW(accel);
        } else if (field == "envmap") {
            envmap = dyn_cast<EnvMapNode>(value.object());
            AKR_ASSERT_THROW(envmap);
        }
    }
    void EnvMapNode::object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                                  const sdl::Value &value) {
        if (field == "envmap") {
            envmap = resolve_texture(value);
            AKR_ASSERT_THROW(envmap);
        } else if (field == "rotation") {
            transform.rotation = radians(load<vec3>(value));
        }
    }
} // namespace akari::render