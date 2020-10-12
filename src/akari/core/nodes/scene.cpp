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
#ifdef AKR_ENABLE_PYTHON
#    include <pybind11/pybind11.h>
#    include <pybind11/embed.h>
#    include <pybind11/stl.h>
#endif
#include <akari/common/box.h>
#include <akari/core/nodes/scene.h>
#include <akari/kernel/embree.inl>
#include <akari/kernel/bvh-accelerator.h>
#include <akari/core/film.h>
#include <akari/core/profiler.h>
namespace akari {
    AKR_VARIANT void SceneNode<C>::commit() {
        for (auto &shape : shapes) {
            AKR_ASSERT_THROW(shape);
            shape->commit();
        }
        AKR_ASSERT_THROW(camera);
        camera->commit();
    }
    AKR_VARIANT Scene<C> SceneNode<C>::compile(MemoryArena<> *arena) {
        Scene<C> scene;

        scene.camera = camera->compile(arena);
        for (auto &shape : shapes) {
            instances.emplace_back(shape->compile(arena));
        }
        scene.meshes = {instances.data(), instances.size()};
        std::vector<AreaLight<C>> area_light_buffer;
        for (uint32_t mesh_id = 0; mesh_id < scene.meshes.size(); mesh_id++) {
            MeshInstance<C> &mesh = scene.meshes[mesh_id];
            for (uint32_t prim_id = 0; prim_id < mesh.indices.size() / 3; prim_id++) {
                auto triangle = scene.get_triangle(mesh_id, prim_id);
                auto material = triangle.material;
                if (!material)
                    continue;
                if (material->template isa<EmissiveMaterial<C>>()) {
                    const EmissiveMaterial<C> *e = material->template get<EmissiveMaterial<C>>();
                    (void)e;
                    area_light_buffer.emplace_back(triangle);
                }
            }
        }
        area_lights.copy(area_light_buffer.data(), area_light_buffer.size());
        scene.area_lights = area_lights.view();

        return scene;
    }
    AKR_VARIANT void SceneNode<C>::object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                                                const sdl::Value &value) {
        if (field == "camera") {
            camera = dyn_cast<CameraNode<C>>(value.object());
            AKR_ASSERT_THROW(camera);
        } else if (field == "output") {
            output = value.get<std::string>().value();
        } else if (field == "integrator") {
            integrator = dyn_cast<IntegratorNode<C>>(value.object());
            AKR_ASSERT_THROW(integrator);
        } else if (field == "shapes") {
            AKR_ASSERT_THROW(value.is_array());
            for (auto shape : value) {
                shapes.emplace_back(dyn_cast<MeshNode<C>>(shape.object()));
            }
        }
    }
    AKR_VARIANT void SceneNode<C>::render() {
        /*
        We want to restore the SIGINT handler so that the user can interrupt the renderer
        */
        auto _prev_SIGINT_handler = signal(SIGINT, SIG_DFL);
        auto _restore_handler = AtScopeExit([=]() { signal(SIGINT, _prev_SIGINT_handler); });
        commit();
        info("preparing scene");
        TrackedManagedMemoryResource resource(active_device()->managed_resource());
        auto arena = MemoryArena<>(astd::pmr::polymorphic_allocator<>(&resource));
        auto scene = compile(&arena);
        auto res = scene.camera.resolution();
        auto film = Film<C>(res);
        scene.sampler = LCGSampler<C>();
        auto gpu_accel =  Box<BVHAccelerator<C>>::make(default_resource());
        std::unique_ptr<EmbreeAccelerator<C>> embree_accel;
        if (active_device() == gpu_device() || !akari_enable_embree) {
            scene.accel = gpu_accel.get();
        } else {
            embree_accel = std::make_unique<EmbreeAccelerator<C>>();
            scene.accel = embree_accel.get();
        }
        scene.commit();
        resource.prefetch();
        auto render_cpu = [&]() {
            auto integrator_ = integrator->compile(&arena);
            integrator_->render(scene, &film);
        };
#ifdef AKR_ENABLE_GPU
        auto render_gpu = [&]() {
            auto integrator_ = integrator->compile_gpu(&arena);
            if (!integrator_) {
                fatal("integrator {} is not supported on gpu", integrator->description());
                std::exit(0);
            }
            integrator_->render(scene, &film);
            active_device()->sync();
        };
#else
        auto render_gpu = [&]() {
            fatal("gpu rendering is not supported");
            std::exit(1);
        };
#endif
        Timer timer;
        if (active_device() == cpu_device()) {
            render_cpu();
        } else {
            render_gpu();
        }
        info("render done took ({}s)", timer.elapsed_seconds());
        film.write_image(fs::path(output));
    }

    AKR_VARIANT void RegisterSceneNode<C>::register_nodes() {
        AKR_IMPORT_TYPES()
        register_node<C, SceneNode<C>>("Scene");
    }

    AKR_VARIANT void RegisterSceneNode<C>::register_python_nodes(py::module &m) {
        AKR_IMPORT_TYPES()
#ifdef AKR_ENABLE_PYTHON
        py::class_<SceneNode<C>, SceneGraphNode<C>, std::shared_ptr<SceneNode<C>>>(m, "Scene")
            .def(py::init<>())
            .def_readwrite("variant", &SceneNode<C>::variant)
            .def_readwrite("camera", &SceneNode<C>::camera)
            .def_readwrite("output", &SceneNode<C>::output)
            .def_readwrite("integrator", &SceneNode<C>::integrator)
            .def("render", &SceneNode<C>::render)
            .def("add_mesh", &SceneNode<C>::add_mesh);
#endif
    }

    AKR_RENDER_STRUCT(RegisterSceneNode)

    AKR_RENDER_CLASS(SceneNode)

} // namespace akari