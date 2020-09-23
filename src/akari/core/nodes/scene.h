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
#include <akari/core/nodes/scenegraph.h>
#include <akari/core/nodes/camera.h>
#include <akari/core/nodes/mesh.h>
#include <akari/core/nodes/material.h>
#include <akari/core/nodes/integrator.h>
#include <akari/kernel/scene.h>
namespace akari {
    AKR_VARIANT class SceneNode : public SceneGraphNode<C> {
      public:
        AKR_IMPORT_TYPES()
        Buffer<MeshInstance<C>> instances;
        std::string variant;
        std::shared_ptr<CameraNode<C>> camera;
        std::vector<std::shared_ptr<MeshNode<C>>> shapes;
        std::string output;
        std::shared_ptr<IntegratorNode<C>> integrator;
        Buffer<AreaLight<C>> area_lights;
        void commit() override;
        Scene<C> compile(MemoryArena *arena);
        void render();
        void add_mesh(const std::shared_ptr<MeshNode<C>> &mesh) { shapes.emplace_back(mesh); }
        void load(const pugi::xml_node &xml);
    };

    AKR_VARIANT struct RegisterSceneNode { static void register_nodes(py::module &m); };

} // namespace akari