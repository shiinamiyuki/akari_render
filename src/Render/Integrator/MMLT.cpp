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
#include <Akari/Core/Parallel.h>
#include <Akari/Core/Plugin.h>
#include <Akari/Render/Integrator.h>
#include <Akari/Render/Plugins/CoreBidir.h>
#include <future>
#include <mutex>
#include <utility>

namespace Akari {

    struct MLTSampler : public Sampler {
        struct PrimarySample {
            Float value = 0;

            void Backup() {
                valueBackup = value;
                modifyBackup = lastModificationIteration;
            }
            void Restore() {
                value = valueBackup;
                lastModificationIteration = modifyBackup;
            }

            int64_t lastModificationIteration = 0;
            Float valueBackup = 0;
            int64_t modifyBackup = 0;
        };

        enum Stream : uint8_t { ECamera, ELight, EConnect };
        uint64_t seed;
        int streamIndex = 0;
        int sampleIndex = 0;

        Float Next1D() override { return 0; }
        std::shared_ptr<Sampler> Clone() const override { return std::shared_ptr<Sampler>(); }
        void SetSampleIndex(size_t size) override {}
        void StartNextSample() override {}
        void StartStream(Stream stream) {
            sampleIndex = 0;
            streamIndex = stream;
        }
        void EnsureReady(int index){

        }

    };
    struct RadianceRecord {
        vec2 pRaster;
        Spectrum radiance;
    };

    class MMLTRenderTask : public RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        int maxDepth;

      public:
        MMLTRenderTask(RenderContext ctx, int spp, int maxDepth) : ctx(std::move(ctx)), spp(spp), maxDepth(maxDepth) {}
        bool HasFilmUpdate() override { return false; }
        std::shared_ptr<const Film> GetFilmUpdate() override { return ctx.camera->GetFilm(); }
        bool IsDone() override { return false; }
        bool WaitEvent(Event event) override {
            if (event == Event::EFILM_AVAILABLE) {
                std::unique_lock<std::mutex> lock(mutex);

                filmAvailable.wait(lock, [=]() { return HasFilmUpdate() || IsDone(); });
                return HasFilmUpdate();
            } else if (event == Event::ERENDER_DONE) {
                std::unique_lock<std::mutex> lock(mutex);

                done.wait(lock, [=]() { return IsDone(); });
                return true;
            }
            return false;
        }
        void Wait() override {
            future.wait();
            done.notify_all();
        }

        void Start() override {
            future = std::async(std::launch::async, [=]() {
                auto beginTime = std::chrono::high_resolution_clock::now();

                auto &scene = ctx.scene;
                auto &camera = ctx.camera;
                auto &_sampler = ctx.sampler;
                auto film = camera->GetFilm();
                auto nTiles = ivec2(film->Dimension() + ivec2(TileSize - 1)) / ivec2(TileSize);

                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = (endTime - beginTime);
                Info("Rendering done in {} secs, traced {} rays, {} M rays/sec\n", elapsed.count(),
                     scene->GetRayCounter(), scene->GetRayCounter() / elapsed.count() / 1e6);
            });
        }
    };
    class MMLT : public Integrator {
        int spp = 4;
        int maxDepth = 5;

      public:
        AKR_DECL_COMP(MMLT, "MMLT")
        AKR_SER(spp, maxDepth)
        std::shared_ptr<RenderTask> CreateRenderTask(const RenderContext &ctx) override {
            return std::make_shared<MMLTRenderTask>(ctx, spp, maxDepth);
        }
    };
    AKR_EXPORT_COMP(MMLT, "Integrator")
} // namespace Akari
