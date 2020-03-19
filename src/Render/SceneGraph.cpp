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

#include <Akari/Core/Logger.h>
#include <Akari/Render/SceneGraph.h>

namespace Akari {
    void SceneGraph::Commit() {
        if(!sampler){
            sampler = Cast<Sampler>(CreateComponent("RandomSampler"));
        }
        if(!camera){
            camera = Cast<Camera>(CreateComponent("PerspectiveCamera"));
        }
        if(!integrator){
            integrator = Cast<Integrator>(CreateComponent("RTDebug"));
        }
        scene = std::make_shared<Scene>();
        for (auto &mesh : meshes) {
            mesh->Commit();
            scene->AddMesh(mesh);
        }
        scene->SetAccelerator(Cast<Accelerator>(CreateComponent("BVHAccelerator")));
        Info("Building Accelerator\n");
        scene->Commit();
        camera->Commit();
        sampler->Commit();
    }
    std::shared_ptr<RenderTask> SceneGraph::CreateRenderTask() {
        RenderContext ctx;
        ctx.scene = scene;
        ctx.sampler = sampler;
        ctx.camera = camera;
        return integrator->CreateRenderTask(ctx);
    }
} // namespace Akari