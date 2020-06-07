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

#include <akari/core/logger.h>
#include <akari/render/scene_graph.h>

namespace akari {
    void SceneGraph::commit_settings(RenderSetting &setting) {
        auto &camera = setting.camera;
        auto &sampler = setting.sampler;
        auto &integrator = setting.integrator;
        if (!sampler) {
            sampler = dyn_cast<Sampler>(create_component("RandomSampler"));
        }
        if (!camera) {
            camera = dyn_cast<Camera>(create_component("PerspectiveCamera"));
        }
        if (!integrator) {
            integrator = dyn_cast<Integrator>(create_component("PathTracer"));
        }
        camera->commit();
        sampler->commit();
        integrator->commit();
    }
    void SceneGraph::commit() {

        pScene = std::make_shared<Scene>();
        for (auto &mesh : scene.meshes) {
            mesh.mesh->commit();
            pScene->add_mesh(mesh.mesh);
        }
        for(auto & light : scene.lights){
            pScene->add_light(light);
        }
        if (!scene.accelerator) {
#ifdef AKR_USE_EMBREE
            scene.accelerator = dyn_cast<Accelerator>(create_component("EmbreeAccelerator"));
#else
            scene.accelerator = dyn_cast<Accelerator>(create_component("BVHAccelerator"));
#endif
        }
        pScene->set_accelerator(scene.accelerator);
        info("Building Accelerator\n");
        pScene->commit();
    }
    std::shared_ptr<RenderTask> SceneGraph::create_render_task(int settingId) {
        commit_settings(render.at(settingId));
        RenderContext ctx;
        ctx.scene = pScene;
        ctx.sampler = render.at(settingId).sampler;
        ctx.camera = render.at(settingId).camera;
        pScene->ClearRayCounter();
        return render.at(settingId).integrator->create_render_task(ctx);
    }
} // namespace akari