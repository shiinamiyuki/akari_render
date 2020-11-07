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
#include <akari/render/denoiser.h>
namespace akari::render {
    void SceneNode::commit() {
        for (auto &shape : shapes) {
            AKR_ASSERT_THROW(shape);
            shape->commit();
        }
        AKR_ASSERT_THROW(camera);
        camera->commit();
    }
    void SceneNode::init_scene(Allocator<> allocator) {
        scene = std::make_shared<Scene>();
        auto &light_pdf_map = scene->light_pdf_map;
        lights.clear();
        scene->camera = camera->create_camera(allocator);
        auto &instances = scene->meshes;
        for (auto &shape : shapes) {
            instances.emplace_back(shape->create_instance(allocator));
        }
        scene->accel = accel->create_accel(*scene);
        std::vector<const Light *> area_lights;
        std::vector<Float> power;
        for (uint32_t mesh_id = 0; mesh_id < scene->meshes.size(); mesh_id++) {
            MeshInstance &mesh = scene->meshes[mesh_id];
            for (uint32_t prim_id = 0; prim_id < mesh.indices.size() / 3; prim_id++) {
                auto triangle = scene->get_triangle(mesh_id, prim_id);
                auto material = triangle.material;
                if (!material)
                    continue;
                if (auto e = material->as_emissive()) {
                    auto light_node = e->light;
                    auto light = light_node->create(allocator, scene.get(), triangle);
                    power.emplace_back(light->power());
                    lights.emplace_back(light);
                    mesh.lights[prim_id] = light;
                }
            }
        }

        if (envmap) {
            scene->envmap = envmap->create(allocator, scene.get(), std::nullopt);
            power.emplace_back(scene->envmap->power());
            lights.emplace_back(scene->envmap);
        }
        auto light_distribution = std::make_unique<Distribution1D>(power.data(), power.size(), Allocator<>());
        AKR_ASSERT(lights.size() == power.size());
        for (size_t i = 0; i < lights.size(); i++) {
            light_pdf_map.emplace(lights[i].get(), light_distribution->pdf_discrete(i));
        }
        scene->light_distribution = std::move(light_distribution);
        scene->lights = std::move(lights);
        scene->sampler = sampler->create_sampler(allocator);
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
        ivec2 res = camera->resolution();
        if (super_sampling_k > 1 && !denoiser_name.empty()) {
            auto s = std::sqrt(super_sampling_k);
            if (s * s != super_sampling_k) {
                error("super sampling factor must be square number (got {})", super_sampling_k);
                std::exit(1);
            }
            integrator->set_spp(integrator->get_spp() / super_sampling_k);
            res *= s;
            camera->set_resolution(res);
        }
        info("preparing scene");

        Allocator<> allocator(&memory_arena);
        init_scene(allocator);

        auto real_integrator = integrator->create_integrator(allocator);
        Timer timer;
        RenderInput input;
        input.required_full_aov = required_aovs_;
        input.scene = scene.get();
        input.requested_aovs.emplace("color", AOVRequest{});
        bool run_denoiser_ = !denoiser_name.empty();
        if (run_denoiser_) {
            if (auto _ = dyn_cast<UniAOVIntegrator>(real_integrator)) {
                warning("integrator {} has no aov output, aov is rendered separately", integrator->description());
            }
            input.requested_aovs.emplace("albedo", AOVRequest{});
            input.requested_aovs.emplace("normal", AOVRequest{});
            input.requested_aovs.emplace("first_hit_albedo", AOVRequest{});
            input.requested_aovs.emplace("first_hit_normal", AOVRequest{});
        }
        auto render_out = real_integrator->render(input);
        info("render done ({}s)", timer.elapsed_seconds());
        std::optional<Image> output_image;
        output_image = render_out.aovs["color"].value->to_rgb_image();
        std::unique_ptr<ImageWriter> exr_writer;
        PluginManager<ImageWriter> exr_writers;
        auto ext_pi = exr_writers.try_load_plugin("HDRImageWriter");
        if (ext_pi) {
            exr_writer = ext_pi->make_unique();
        }
        auto write_intermediate = [&](const Film &film, const std::string &component, bool hdr) {
            auto image = film.to_rgb_image();
            auto filename = extend_filename(output, component);
            ldr_image_writer()->write(image, filename);
            if (hdr && exr_writer) {
                exr_writer->write(image, filename.replace_extension(".exr"));
            }
        };
        auto write_intermediate_img = [&](const Image &image, const char *component, bool hdr) {
            auto filename = extend_filename(output, component);
            ldr_image_writer()->write(image, filename);
            if (hdr && exr_writer) {
                exr_writer->write(image, filename.replace_extension(".exr"));
            }
        };
        if (run_denoiser_) {
            if (auto _ = dyn_cast<UniAOVIntegrator>(real_integrator)) {
                auto render_aov = [&](const char *name) {
                    auto aov_integrator_node = make_aov_integrator(std::min(64, integrator->get_spp()), name);
                    auto integrator = aov_integrator_node->create_integrator(allocator);
                    Film aov_film(res);
                    auto aov_out = integrator->render(input);
                    render_out.aovs[name].value = aov_film;
                };
                render_aov("normal");
                render_aov("albedo");
            }
            for (auto &[aov, rec] : render_out.aovs) {
                write_intermediate(*rec.value, "." + aov, true);
                if (rec.variance) {
                    write_intermediate(*rec.variance, ".var." + aov, true);
                }
            }
            PluginManager<Denoiser> denoisers;
            info("denoising...");
            const PluginInterface<Denoiser> *pi = nullptr;
            if (denoiser_name == "oidn")
                pi = denoisers.load_plugin("OIDNDenoiser");
            else if (denoiser_name == "optix")
                pi = denoisers.load_plugin("OptixDenoiser");
            else
                pi = denoisers.load_plugin(denoiser_name);
            if (pi) {
                auto denoiser = pi->make_shared();
                output_image = denoiser->denoise(scene.get(), render_out);
                AKR_ASSERT(is_rgb_image(*output_image));
                if (output_image) {
                    if (super_sampling_k > 1) {
                        int s = std::sqrt(super_sampling_k);
                        write_intermediate_img(*output_image, ".ss", false);

                        Array3D<float> avg_kernel(ivec3(1, s, s));
                        avg_kernel.fill(0.0);
                        for (int x = 0; x < s; x++) {
                            for (int y = 0; y < s; y++) {
                                avg_kernel(0, x, y) = 1.0 / (s * s);
                            }
                        }
                        Image down_sampled_image = rgb_image(output_image->resolution() / ivec2(s));
                        down_sampled_image.array3d() =
                            std::move(convolve(output_image->array3d(), avg_kernel, ivec3(1, s, s)));
                        output_image = std::move(down_sampled_image);
                    }
                }
                denoiser = nullptr;
            } else {
                error("failed to load denoiser; skip denoising");
            }
        }
        ldr_image_writer()->write(*output_image, fs::path(output));
        if (exr_writer) {
            exr_writer->write(*output_image, fs::path(output).replace_extension(".exr"));
        }
        exr_writer = nullptr;
        finalize();
    }
    void SceneNode::finalize() {
        scene.reset();
        lights.clear();
        accel->finalize();
        integrator->finalize();
        camera->finalize();
        sampler->finalize();
        if (envmap)
            envmap->finalize();
        for (auto &shape : shapes) {
            shape->finalize();
        }
    }
    void SceneNode::object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                                 const sdl::Value &value) {
        if (field == "camera") {
            camera = sg_dyn_cast<CameraNode>(value.object());
            AKR_ASSERT_THROW(camera);
        } else if (field == "output") {
            output = value.get<std::string>().value();
        } else if (field == "integrator") {
            integrator = sg_dyn_cast<IntegratorNode>(value.object());
            AKR_ASSERT_THROW(integrator);
        } else if (field == "shapes") {
            AKR_ASSERT_THROW(value.is_array());
            for (auto shape : value) {
                shapes.emplace_back(sg_dyn_cast<MeshNode>(shape.object()));
            }
        } else if (field == "sampler") {
            sampler = sg_dyn_cast<SamplerNode>(value.object());
            AKR_ASSERT_THROW(sampler);
        } else if (field == "accelerator") {
            accel = sg_dyn_cast<AcceleratorNode>(value.object());
            AKR_ASSERT_THROW(accel);
        } else if (field == "envmap") {
            envmap = sg_dyn_cast<LightNode>(value.object());
            AKR_ASSERT_THROW(envmap);
        }
    }
    void SceneNode::load(InputArchive &ar) { ar(camera, sampler, shapes, accel, envmap); }
    void SceneNode::save(OutputArchive &ar) const { ar(camera, sampler, shapes, accel, envmap); }
} // namespace akari::render