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
#include <akari/core/akari.h>
#include <akari/common/fwd.h>
#include <akari/kernel/meshview.h>
// #include <akari/kernel/materials/material.h>
namespace akari {
    AKR_VARIANT class SceneGraphNode {
      public:
        AKR_IMPORT_BASIC_RENDER_TYPES()
        virtual void commit() {}
    };
    AKR_VARIANT class SceneNode;
    AKR_VARIANT class CameraNode : public SceneGraphNode<Float, Spectrum> {
      public:
        AKR_IMPORT_TYPES(Camera)
        virtual Camera compile() = 0;
    };
    AKR_VARIANT class FilmNode : public SceneGraphNode<Float, Spectrum> { public: };
    AKR_VARIANT class MaterialNode : public SceneGraphNode<Float, Spectrum> { public: };
    AKR_VARIANT class MeshNode : public SceneGraphNode<Float, Spectrum> {
      public:
        AKR_IMPORT_RENDER_TYPES(SceneNode)
        virtual MeshView compile(SceneNode &) = 0;
    };
    AKR_VARIANT class SceneNode : public SceneGraphNode<Float, Spectrum> {
      public:
        AKR_IMPORT_BASIC_RENDER_TYPES()
        std::vector<MeshView> meshviews;
        using CameraNode = CameraNode<Float, Spectrum>;
        using MeshNode = MeshNode<Float, Spectrum>;
        std::string variant;
        std::shared_ptr<CameraNode> camera;
        std::vector<std::shared_ptr<MeshNode>> shapes;
        void commit() override {
            for (auto &shape : shapes) {
                AKR_ASSERT_THROW(shape);
                shape->commit();
            }
            AKR_ASSERT_THROW(camera);
            camera->commit();
        }
        Scene compile() {
            Scene scene;
            scene.meshes = meshviews;
            scene.camera = camera->compile();
            return scene;
        }
    };

} // namespace akari