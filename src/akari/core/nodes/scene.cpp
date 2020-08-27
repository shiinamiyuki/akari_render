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
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <akari/core/nodes/scene.h>
#include <akari/kernel/embree.inl>
#include <akari/core/film.h>

namespace akari {
    AKR_VARIANT void SceneNode<Float, Spectrum>::commit() {
        for (auto &shape : shapes) {
            AKR_ASSERT_THROW(shape);
            shape->commit();
        }
        AKR_ASSERT_THROW(camera);
        camera->commit();
    }
    AKR_VARIANT Scene<Float, Spectrum> SceneNode<Float, Spectrum>::compile() {
        AScene scene;
        for (auto &shape : shapes) {
            meshviews.emplace_back(shape->compile());
        }
        scene.meshes = meshviews;
        scene.camera = camera->compile();
        return scene;
    }
    AKR_VARIANT void SceneNode<Float, Spectrum>::render() {
        auto scene = compile();
        auto res = scene.camera.resolution();
        auto film = Film<Float, Spectrum>(res);
        scene.sampler = Sampler<Float, Spectrum>(RandomSampler<Float, Spectrum>());
        auto embree_scene = EmbreeAccelerator<Float, Spectrum>();
        scene.embree_scene = &embree_scene;
        scene.commit();
        auto integrator = cpu::Integrator<Float, Spectrum>(cpu::AmbientOcclusion<Float, Spectrum>());
        integrator.render(scene, &film);
        film.write_image(fs::path(output));
    }
    AKR_VARIANT void RegisterSceneNode<Float, Spectrum>::register_nodes(py::module &m) {
        AKR_IMPORT_RENDER_TYPES(SceneNode, SceneGraphNode);
        py::class_<ASceneNode, ASceneGraphNode, std::shared_ptr<ASceneNode>>(m, "Scene")
            .def(py::init<>())
            .def_readwrite("variant", &ASceneNode::variant)
            .def_readwrite("camera", &ASceneNode::camera)
            .def_readwrite("shapes", &ASceneNode::shapes)
            .def_readwrite("output", &ASceneNode::output)
            .def("commit", &ASceneNode::commit)
            .def("render", &ASceneNode::render);
    }
    AKR_RENDER_CLASS(SceneNode)
    AKR_RENDER_STRUCT(RegisterSceneNode)
} // namespace akari