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
#include <Akari/Render/Scene.h>
#include <future>
#include <mutex>
#include <random>
#include <utility>

namespace Akari {
    /// From LuxCoreRender
    static float Mutate(const float x, const float randomValue) {
        static const float s1 = 1.f / 512.f;
        static const float s2 = 1.f / 16.f;

        const float dx = s1 / (s1 / s2 + fabsf(2.f * randomValue - 1.f)) - s1 / (s1 / s2 + 1.f);

        float mutatedX = x;
        if (randomValue < .5f) {
            mutatedX += dx;
            mutatedX = (mutatedX < 1.f) ? mutatedX : (mutatedX - 1.f);
        } else {
            mutatedX -= dx;
            mutatedX = (mutatedX < 0.f) ? (mutatedX + 1.f) : mutatedX;
        }

        // mutatedX can still be 1.f due to numerical precision problems
        if (mutatedX == 1.f)
            mutatedX = 0.f;

        return mutatedX;
    }
    static inline float MutateScaled(const float x, const float range, const float randomValue) {
        static const float s1 = 32.f;

        const float dx =
            range / (s1 / (1.f + s1) + (s1 * s1) / (1.f + s1) * fabs(2.f * randomValue - 1.f)) - range / s1;

        float mutatedX = x;
        if (randomValue < .5f) {
            mutatedX += dx;
            mutatedX = (mutatedX < 1.f) ? mutatedX : (mutatedX - 1.f);
        } else {
            mutatedX -= dx;
            mutatedX = (mutatedX < 0.f) ? (mutatedX + 1.f) : mutatedX;
        }

        // mutatedX can still be 1.f due to numerical precision problems
        if (mutatedX == 1.f)
            mutatedX = 0.f;

        return mutatedX;
    }
    struct MLTSampler : public Sampler {
        AKR_DECL_COMP(MLTSampler, "MLTSampler")
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

        enum Stream : uint8_t { ECamera, ELight, EConnect, NStream };
        uint64_t seed = 0;
        int depth = 0;
        Rng rng;
        int streamIndex = 0;
        int sampleIndex = 0;
        bool largeStep = true;
        int64_t lastLargeStepIteration = 0;
        int64_t curIteration = 0;
        float largeStepProb = 0.3;
        float imageMutationScale = 0.1;
        MLTSampler() = default;
        MLTSampler(uint64_t seed, int depth, float largeStepProb = 0.3)
            : seed(seed), depth(depth), rng(seed), largeStepProb(largeStepProb) {}
        std::vector<PrimarySample> X;
        Float Next1D() override {
            auto idx = GetNextIndex();
            EnsureReady(idx);
            return X[idx].value;
        }
        std::shared_ptr<Sampler> Clone() const override { return std::shared_ptr<Sampler>(); }
        void SetSampleIndex(size_t size) override {}
        void StartNextSample() override {}
        void StartIteration() {
            curIteration++;
            sampleIndex = 0;
            streamIndex = 0;
            largeStep = rng.uniformFloat() < largeStepProb;
        }
        void StartStream(Stream stream) {
            sampleIndex = 0;
            streamIndex = stream;
        }
        void EnsureReady(size_t index) {
            if (index >= X.size()) {
                X.resize(index + 1);
            }
            auto mutateFunc = [=](float x, float u) {
                if (streamIndex == ECamera && sampleIndex < 2) {
                    return MutateScaled(x, imageMutationScale, u);
                } else {
                    return Mutate(x, u);
                }
            };

            PrimarySample &Xi = X[index];

            if (Xi.lastModificationIteration < lastLargeStepIteration) {
                Xi.value = rng.uniformFloat();
                Xi.lastModificationIteration = lastLargeStepIteration;
            }

            if (largeStep) {
                Xi.Backup();
                Xi.value = rng.uniformFloat();
            } else {
                int64_t nSmall = curIteration - Xi.lastModificationIteration;
                auto nSmallMinus = nSmall - 1;
                if (nSmallMinus > 0) {
                    auto x = Xi.value;
                    while (nSmallMinus > 0) {
                        nSmallMinus--;
                        x = mutateFunc(x, rng.uniformFloat());
                    }
                    Xi.value = x;
                    Xi.lastModificationIteration = curIteration - 1;
                }
                Xi.Backup();
                Xi.value = mutateFunc(Xi.value, rng.uniformFloat());
            }
            Xi.lastModificationIteration = curIteration;
        }

        void Reject() {
            for (auto &Xi : X) {
                if (Xi.lastModificationIteration == curIteration) {
                    Xi.Restore();
                }
            }
            --curIteration;
        }

        void Accept() {
            if (largeStep) {
                lastLargeStepIteration = curIteration;
            }
        }

        int GetNextIndex() { return (sampleIndex++) * NStream + streamIndex; }
    };
    struct RadianceRecord {
        vec2 pRaster;
        Spectrum radiance;
    };

    struct MarkovChain {
        std::vector<RadianceRecord> radianceRecords;
        std::vector<MLTSampler> samplers; // one for each depth
        std::shared_ptr<Distribution1D> depthDist;
    };

    class MMLTRenderTask : public RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        int maxDepth;
        size_t nBootstrap;
        size_t nChains;
        int directSamples = 1;
        float clamp;

      public:
        MMLTRenderTask(RenderContext ctx, int spp, int maxDepth, size_t nBootstrap, size_t nChains, int nDirect,
                       float clamp)
            : ctx(std::move(ctx)), spp(spp), maxDepth(maxDepth), nBootstrap(nBootstrap), nChains(nChains),
              directSamples(nDirect), clamp(clamp) {}
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
        RadianceRecord Radiance(MLTSampler &sampler, MemoryArena &arena, bool bootstrap) {
            if (!bootstrap) {
                sampler.StartIteration();
            }
            sampler.StartStream(MLTSampler::ECamera);
            float pFilmX = sampler.Next1D();
            float pFilmY = sampler.Next1D();
            auto dim = ctx.camera->GetFilm()->Dimension();
            ivec2 pRaster = round(vec2(pFilmX, pFilmY) * vec2(dim));
            pRaster.x = std::clamp(pRaster.x, 0, dim.x - 1);
            pRaster.y = std::clamp(pRaster.y, 0, dim.y - 1);
            RadianceRecord radianceRecord;
            radianceRecord.pRaster = pRaster;
            radianceRecord.radiance = Spectrum(0);
            auto &scene = *ctx.scene;
            auto &camera = *ctx.camera;
            int s, t, nStrategies;
            int depth = sampler.depth;
            if (directSamples == 0) {
                if (depth == 0) {
                    nStrategies = 1;
                    s = 0;
                    t = 2;
                } else {
                    nStrategies = depth + 2;
                    s = std::min((int)(sampler.Next1D() * nStrategies), nStrategies - 1);
                    t = nStrategies - s;
                }
            } else {
                if (depth <= 1) {
                    return radianceRecord;
                } else {
                    nStrategies = depth + 2;
                    s = std::min((int)(sampler.Next1D() * nStrategies), nStrategies - 1);
                    t = nStrategies - s;
                }
            }
            auto eyePath = arena.allocN<PathVertex>(t + 1);
            size_t nCamera = TraceEyePath(scene, arena, camera, pRaster, sampler, eyePath, t);
            if ((int)nCamera != t) {
                return radianceRecord;
            }
            sampler.StartStream(MLTSampler::ELight);
            auto lightPath = arena.allocN<PathVertex>(s + 1);
            size_t nLight = TraceLightPath(scene, arena, sampler, lightPath, s);
            if ((int)nLight != s) {
                return radianceRecord;
            }
            sampler.StartStream(MLTSampler::EConnect);
            radianceRecord.radiance =
                ConnectPath(scene, sampler, eyePath, t, lightPath, s, &radianceRecord.pRaster) * nStrategies;
            return radianceRecord;
        }
        static Float ScalarContributionFunction(const Spectrum &L) { return std::max(0.0f, L.Luminance()); }
        void Start() override {
            future = std::async(std::launch::async, [=]() {
                DirectLighting();
                auto beginTime = std::chrono::high_resolution_clock::now();
                auto &scene = ctx.scene;
                auto &camera = ctx.camera;
                auto &_sampler = ctx.sampler;
                auto film = camera->GetFilm();
                scene->ClearRayCounter();
                Info("Start bootstrapping\n");
                std::random_device rd;
                std::uniform_int_distribution<uint64_t> dist;
                std::uniform_real_distribution<float> distF;
                size_t nBootstrapSamples = nBootstrap * (maxDepth + 1);
                auto nCores = (int)GetConfig()->NumCore;
                std::vector<MemoryArena> arenas(nCores);
                std::vector<MarkovChain> markovChains(nChains);
                std::vector<float> depthWeight(maxDepth + 1);
                for (auto &chain : markovChains) {
                    chain.radianceRecords.resize(maxDepth + 1);
                    chain.samplers.resize(maxDepth + 1);
                }
                for (int depth = 0; depth < maxDepth + 1; depth++) {
                    std::vector<uint64_t> bootstrapSeeds(nBootstrap);
                    std::vector<MLTSampler> samplers(nBootstrap);
                    std::vector<Float> weights(nBootstrap);
                    for (auto &seed : bootstrapSeeds) {
                        seed = dist(rd);
                    }
                    for (size_t i = 0; i < nBootstrap; i++) {
                        samplers[i] = MLTSampler(bootstrapSeeds[i], depth);
                    }

                    ParallelFor(nBootstrap, [&, this](uint32_t i, uint32_t tid) {
                        auto record = Radiance(samplers[i], arenas[tid], true);
                        arenas[tid].reset();
                        weights[i] = ScalarContributionFunction(record.radiance);
                    });
                    Distribution1D distribution(weights.data(), weights.size());
                    depthWeight[depth] = distribution.Integral();
                    for (auto &chain : markovChains) {
                        auto seedIdx = distribution.SampleDiscrete(distF(rd));
                        auto seed = bootstrapSeeds[seedIdx];
                        chain.samplers[depth] = MLTSampler(seed, depth);
                    }
                }
                auto depthDist = std::make_shared<Distribution1D>(depthWeight.data(), depthWeight.size());
                Float avgLuminance = std::accumulate(depthWeight.begin(), depthWeight.end(), 0.0f);
                Info("Average luminance: {}\n", avgLuminance);
                if (avgLuminance == 0.0f) {
                    Error("Average luminance is ZERO; Improper scene setup?\n");
                    Error("Aborting...\n");
                    return;
                }
                ParallelFor(nChains, [&, this](uint32_t i, uint32_t tid) {
                    for (int depth = 0; depth < maxDepth + 1; depth++) {
                        markovChains[i].radianceRecords[depth] =
                            Radiance(markovChains[i].samplers[depth], arenas[tid], true);
                        arenas[tid].reset();
                    }
                });
                Info("Running {} Markov Chains\n", nChains);
                // done bootstrapping
                auto dim = film->Dimension();
                int64_t nTotalMutations = spp * dim.x * dim.y;
                ParallelFor(nChains, [&](uint32_t i, uint32_t tid) {
                    Rng rng(i);
                    auto &chain = markovChains[i];
                    chain.depthDist = depthDist;
                    int64_t nChainMutations = std::min<int64_t>((i + 1) * nTotalMutations / nChains, nTotalMutations) -
                                              i * nTotalMutations / nChains;
                    for (int64_t iter = 0; iter < nChainMutations; iter++) {
                        Float depthPdf = 0;
                        int depth = chain.depthDist->SampleDiscrete(rng.uniformFloat(), &depthPdf);
                        depth = std::min(maxDepth + 1, depth);
                        AKARI_ASSERT(depthPdf > 0);
                        auto &sampler = chain.samplers[depth];
                        auto record = Radiance(sampler, arenas[tid], false);
                        arenas[tid].reset();
                        auto &curRecord = chain.radianceRecords[depth];
                        auto LProposed = record.radiance;
                        auto FProposed = ScalarContributionFunction(record.radiance);
                        auto LCurrent = curRecord.radiance;
                        auto FCurrent = ScalarContributionFunction(curRecord.radiance);
                        auto accept = std::min<float>(1.0f, FProposed / FCurrent);
                        Float b = depthWeight[depth];
                        float newWeight =
                            (accept + (sampler.largeStep ? 1.0f : 0.0f)) / (FProposed / b + sampler.largeStepProb);
                        float oldWeight = (1 - accept) / (FCurrent / b + sampler.largeStepProb);
                        Spectrum Lnew = LProposed * newWeight / depthPdf;
                        Spectrum Lold = LCurrent * oldWeight / depthPdf;
                        Lnew = glm::clamp(Lnew, vec3(0), vec3(clamp));
                        Lold = glm::clamp(Lold, vec3(0), vec3(clamp));
                        if (accept > 0) {
                            film->AddSplat(Lnew, record.pRaster);
                        }
                        film->AddSplat(Lold, curRecord.pRaster);

                        if (rng.uniformFloat() < accept) {
                            sampler.Accept();
                            curRecord = record;
                        } else {
                            sampler.Reject();
                        }
                    }
                });
                film->splatScale = 1.0 / spp;
                if (directSamples > 0) {
                    film->splatScale *= (float)directSamples;
                }
                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = (endTime - beginTime);
                Info("Rendering done in {} secs, traced {} rays, {} M rays/sec\n", elapsed.count(),
                     scene->GetRayCounter(), scene->GetRayCounter() / elapsed.count() / 1e6);
            });
        }
        static Float MisWeight(Float pdfA, Float pdfB) {
            pdfA *= pdfA;
            pdfB *= pdfB;
            return pdfA / (pdfA + pdfB);
        }

        void DirectLighting() {
            if (directSamples == 0)
                return;
            ;
            std::string pathTracerSetting = fmt::format(
                R"(
{{
    "type": "PathTracer",
    "props": {{
      "spp": {},
      "enableRR": false,
      "maxDepth":1,
      "minDepth":1
    }}
}}
)",
                directSamples);
            auto setting = json::parse(pathTracerSetting);
            ReviveContext reviveContext;
            auto pathTracer = miyuki::serialize::fromJson<std::shared_ptr<Integrator>>(reviveContext, setting);
            AKARI_ASSERT(pathTracer);
            Info("Render direct samples\n");
            (void)pathTracer;
            auto task = pathTracer->CreateRenderTask(ctx);
            task->Start();
            task->Wait();
        }
    };
    class MMLT : public Integrator {
        int spp = 16;
        int maxDepth = 7;
        size_t nBootstrap = 100000u;
        size_t nChains = 100;
        int nDirect = 16;
        Float clamp = 1e5;

      public:
        AKR_DECL_COMP(MMLT, "MMLT")
        AKR_SER(spp, maxDepth, nBootstrap, nChains, nDirect, clamp)
        std::shared_ptr<RenderTask> CreateRenderTask(const RenderContext &ctx) override {
            return std::make_shared<MMLTRenderTask>(ctx, spp, maxDepth, nBootstrap, nChains, nDirect, clamp);
        }
    };
    AKR_EXPORT_COMP(MMLT, "Integrator")
} // namespace Akari
