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

#ifndef AKARIRENDER_COREBIDIR_H
#define AKARIRENDER_COREBIDIR_H

#include <Akari/Render/EndPoint.h>
#include <Akari/Render/Integrator.h>

namespace Akari {

    struct PathVertex {
        enum Type : uint8_t { ENone, ESurface, ELight, ECamera };
        Type type = ENone;
        union {
            SurfaceInteraction si;
            EndPointInteraction ei;
        };
        Float pdfFwd = 0, pdfRev = 0;
        [[nodiscard]] Float IsInfiniteLight() const { return false; }
        [[nodiscard]] const Interaction *getInteraction() const {
            if (type == ESurface) {
                return &si;
            } else if (type == ELight || type == ECamera) {
                return &ei;
            }
            return nullptr;
        }
        [[nodiscard]] vec3 Ng() const { return getInteraction()->Ng; }
        [[nodiscard]] vec3 Ns() const {
            if (type == ESurface) {
                return si.Ns;
            } else if (type == ELight || type == ECamera) {
                return ei.Ng;
            } else {
                return vec3(0);
            }
        }

        [[nodiscard]] vec3 wo() const { return getInteraction()->wo; }

        [[nodiscard]] vec3 p() const { return getInteraction()->p; }

        Spectrum f(const PathVertex &next) {
            auto wi = normalize(next.p() - p());
            switch (type) {
            case ESurface: {
                wi = si.bsdf->WorldToLocal(wi);
                return si.bsdf->Evaluate(si.wo, wi);
            }
            default:
                AKARI_PANIC("not implemented Vertex::f()");
            }
        }

        [[nodiscard]] bool IsOnSurface() const { return all(equal(Ng(), vec3(0))); }

        Float PdfSAToArea(Float pdf, const PathVertex &next) {
            if (next.IsInfiniteLight())
                return pdf;
            auto w = next.p() - p();
            auto invDistSqr = 1.0f / dot(w, w);
            if (next.IsOnSurface()) {
                // dw = dA cos(t) / r^2
                // p(w) dw/dA = p(A)
                // p(A) = p(w) * cos(t)/r^2
                pdf *= abs(dot(next.Ng(), w * std::sqrt(invDistSqr)));
            }
            return pdf * invDistSqr;
        }

        Float PdfLight(const Scene &scene, const PathVertex &next) {
            auto *light = dynamic_cast<Light *>(ei.ep);
            auto w = next.p() - p();
            w = normalize(w);
            Float pdfPos = 0, pdfDir = 0;
            light->PdfEmission(Ray(p(), w), &pdfPos, &pdfDir);
            return pdfPos * pdfDir;
        }

        Float PdfLightOrigin(const Scene &scene, const PathVertex &next) {
            auto *light = dynamic_cast<Light *>(ei.ep);
            auto w = next.p() - p();
            w = normalize(w);
            Float pdfPos = 0, pdfDir = 0;
            light->PdfEmission(Ray(p(), w), &pdfPos, &pdfDir);
            return scene.PdfLight(light) * pdfPos * pdfDir;
        }

        Float Pdf(const Scene &scene, const PathVertex *prev, const PathVertex &next) {
            auto wiNext = normalize(next.p() - p());
            vec3 wiPrev;
            if (prev) {
                wiPrev = normalize(prev->p() - p());
            } else {
                AKARI_ASSERT(type == ECamera);
            }
            Float pdf = 0;
            {
                AKARI_ASSERT(type == ESurface);
                auto wo = si.bsdf->WorldToLocal(-wiPrev);
                auto wi = si.bsdf->WorldToLocal(wiNext);
                pdf = si.bsdf->EvaluatePdf(wo, wi);
            }
            return PdfSAToArea(pdf, next);
        }
    };

    AKR_EXPORT size_t RandomWalk(const Scene &scene, Sampler &sampler, TransportMode mode, const Ray &ray,
                                 Spectrum beta, Float pdf, PathVertex *path, size_t maxDepth);

    AKR_EXPORT size_t TraceEyePath(const Scene &scene, const Camera &camera, Sampler &sampler, PathVertex *path,
                                   size_t maxDepth);

    AKR_EXPORT size_t TraceLightPath(const Scene &scene, Sampler &sampler, PathVertex *path, size_t maxDepth);

} // namespace Akari

#endif // AKARIRENDER_COREBIDIR_H
