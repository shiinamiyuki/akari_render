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

#ifndef AKARIRENDER_SCENEGRAPH_H
#define AKARIRENDER_SCENEGRAPH_H

#include <Akari/Core/Component.h>
#include <Akari/Render/Accelerator.h>
#include <Akari/Render/Camera.h>
#include <Akari/Render/Integrator.h>
#include <Akari/Render/Light.h>
#include <Akari/Render/Material.h>
#include <Akari/Render/Mesh.h>
#include <Akari/Render/Sampler.h>
#include <fstream>

namespace Akari {
    struct MeshWrapper {
        fs::path file; // path to json file
        std::shared_ptr<Mesh> mesh;
//        AKR_SER(file, mesh)
        template <class Archive> void save(Archive &ar) const {
            miyuki::serialize::AutoSaveVisitor v{ar}; MYK_REFL(v, file, mesh);
            std::ofstream out(file);
            ReviveContext ctx;
            auto j = miyuki::serialize::toJson(ctx, mesh);
            out << j.dump(1) << std::endl;
        }
        template <class Archive> void load(Archive &ar) {
            miyuki::serialize::AutoLoadVisitor v{ar}; MYK_REFL(v, file, mesh);
            std::ifstream in(file);
            std::string str((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            json data = str.empty() ? json::object() : json::parse(str);
            ReviveContext ctx;
            mesh = miyuki::serialize::fromJson<std::shared_ptr<Mesh>>(ctx, data);
        }
    };

    struct AKR_EXPORT SceneGraph {
        std::shared_ptr<Camera> camera;
        std::shared_ptr<Accelerator> accelerator;
        std::shared_ptr<Sampler> sampler;
        std::shared_ptr<Integrator> integrator;
        std::vector<MeshWrapper> meshes;
        AKR_SER(camera, accelerator, sampler, meshes, integrator)
        std::shared_ptr<RenderTask> CreateRenderTask();
        void Commit();

      private:
        std::shared_ptr<Scene> scene;
    };
} // namespace Akari

#endif // AKARIRENDER_SCENEGRAPH_H
