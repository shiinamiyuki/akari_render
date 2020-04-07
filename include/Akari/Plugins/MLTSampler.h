
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

#ifndef AKARIRENDER_MLTSAMPLER_H
#define AKARIRENDER_MLTSAMPLER_H

#include <Akari/Core/Distribution.hpp>
#include <Akari/Render/Sampler.h>

namespace Akari::MLT {
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
        size_t accepts = 0, rejects = 0;
        size_t consecutiveRejects = 0;
        size_t _Xcount[NStream] = {0, 0, 0};
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
                _Xcount[streamIndex] = index + 1;
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
            consecutiveRejects++;
            rejects++;
        }

        void Accept() {
            if (largeStep) {
                lastLargeStepIteration = curIteration;
            }
            accepts++;
            consecutiveRejects = consecutiveRejects == 0 ? 0 : consecutiveRejects - 1;
        }
        size_t GetCurrentDimension() override {
            size_t dim = 0;
            for (uint8_t i = 0; i < sampleIndex; i++) {
                dim += _Xcount[i];
            }
            return dim + streamIndex;
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

} // namespace Akari::MLT

#endif // AKARIRENDER_MLTSAMPLER_H
