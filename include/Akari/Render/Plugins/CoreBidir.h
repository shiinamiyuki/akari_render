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

#include <Akari/Render/Camera.h>
#include <Akari/Render/EndPoint.h>
#include <Akari/Render/Integrator.h>
#include <Akari/Render/Light.h>
namespace Akari {

    struct PathVertex {
        enum Type : uint8_t { ENone, ESurface, ELight, ECamera };
        Type type = ENone;
        Float pdfFwd = 0, pdfRev = 0;
        union {
            SurfaceInteraction si;
            EndPointInteraction ei;
        };
        bool delta = false;
        Spectrum beta;
        PathVertex() : si() {}
        static PathVertex CreateSurfaceVertex(const MaterialSlot *materialSlot, const Spectrum beta, const vec3 &wo,
                                              const vec3 &p, const Triangle &triangle, const Intersection &intersection,
                                              Float pdf, MemoryArena &arena) {
            PathVertex vertex;
            vertex.beta = beta;
            vertex.type = ESurface;
            vertex.si = SurfaceInteraction(materialSlot, wo, p, triangle, intersection, arena);
            vertex.pdfFwd = pdf;
            return vertex;
        }
        static PathVertex CreateLightVertex(const Spectrum beta, const Light *light, const vec3 &p,
                                            const vec3 &normal) {
            PathVertex vertex;
            vertex.beta = beta;
            vertex.type = ELight;
            vertex.ei = EndPointInteraction(light, p, normal);
            return vertex;
        }
        static PathVertex CreateLightVertex(const Spectrum beta, const Light *light, const vec3 &p, const vec3 &normal,
                                            Float pdf) {
            PathVertex vertex;
            vertex.beta = beta;
            vertex.type = ELight;
            vertex.ei = EndPointInteraction(light, p, normal);
            vertex.pdfFwd = pdf;
            return vertex;
        }
        static PathVertex CreateCameraVertex(const Spectrum beta, const Camera *camera, const vec3 &p,
                                             const vec3 &normal) {
            PathVertex vertex;
            vertex.beta = beta;
            vertex.type = ECamera;
            vertex.ei = EndPointInteraction(camera, p, normal);
            return vertex;
        }
        [[nodiscard]] bool IsInfiniteLight() const { return false; }
        [[nodiscard]] bool IsDeltaLight() const { return false; }
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

        Spectrum f(const PathVertex &next, TransportMode) {
            auto wi = normalize(next.p() - p());
            switch (type) {
            case ESurface: {
                return si.bsdf->Evaluate(si.wo, wi);
            }
            default:
                AKARI_PANIC("not implemented Vertex::f()");
            }
        }

        [[nodiscard]] bool IsOnSurface() const { return !all(equal(Ng(), vec3(0))); }

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
            const Light *light = nullptr;
            if (type == ELight) {
                light = dynamic_cast<const Light *>(ei.ep);

            } else if (type == ESurface) {
                auto &handle = si.handle;
                light = scene.GetLight(handle);
            }
            if (!light) {
                return 0;
            }
            auto w = next.p() - p();
            Float invDist2 = 1 / dot(w, w);
            w /= std::sqrt(invDist2);
            Float pdf;
            Float pdfPos = 0, pdfDir = 0;
            light->PdfEmission(Ray(p(), w), &pdfPos, &pdfDir);
            pdf = pdfDir / invDist2;
            if (next.IsOnSurface()) {
                pdf *= abs(dot(next.Ng(), w));
            }
            return pdf;
        }
        Float PdfLightOrigin(const Scene &scene, const PathVertex &next) {
            const Light *light = nullptr;
            if (type == ELight) {
                light = dynamic_cast<const Light *>(ei.ep);

            } else if (type == ESurface) {
                auto &handle = si.handle;
                light = scene.GetLight(handle);
            }
            if (!light) {
                return 0.0f;
            }
            auto w = next.p() - p();
            w = normalize(w);
            Float pdfPos = 0, pdfDir = 0;
            light->PdfEmission(Ray(p(), w), &pdfPos, &pdfDir);
            return scene.PdfLight(light) * pdfPos;
        }
        [[nodiscard]] bool IsConnectible() const {
            switch (type) {
            case ENone:
                return false;
            case ESurface:
                return !delta;
            case ELight:
                return ((uint32_t) dynamic_cast<const Light *>(ei.ep)->GetLightType() &
                        (uint32_t)LightType::EDeltaDirection) != 0;
            case ECamera:
                return true;
            }
            return false;
        }
        static Spectrum G(const Scene &scene, const PathVertex &v1, const PathVertex &v2) {
            auto w = v1.p() - v2.p();
            auto invDist2 = 1.0f / dot(w, w);
            w *= std::sqrt(invDist2);
            Float g = invDist2;
            if (v1.IsOnSurface()) {
                g *= abs(dot(v1.Ng(), w));
            }
            if (v2.IsOnSurface()) {
                g *= abs(dot(v2.Ng(), w));
            }
            VisibilityTester tester(*v1.getInteraction(), *v2.getInteraction());
            return tester.Tr(scene) * g;
        }

        Spectrum Le(const Scene &scene, const PathVertex &next) {
            switch (type) {
            default:
            case ENone:
                return Spectrum(0);
            case ESurface: {
                auto wo = next.p() - p();
                wo = normalize(wo);
                return si.Le(wo);
            }
            case ELight: {
                auto *light = dynamic_cast<const Light *>(ei.ep);
                auto wo = next.p() - p();
                return light->Li(wo, ei.uv);
            }
            case ECamera:
                return Spectrum(0);
            }
        }
        Float Pdf(const Scene &scene, const PathVertex *prev, const PathVertex &next) {
            if (type == ELight) {
                return PdfLight(scene, next);
            }
            auto wiNext = normalize(next.p() - p());
            vec3 wiPrev;
            if (prev) {
                wiPrev = normalize(prev->p() - p());
            } else {
                AKARI_ASSERT(type == ECamera);
            }
            Float pdf = 0;
            if (type == ESurface) {
                auto wo = -wiPrev;
                auto wi = wiNext;
                pdf = si.bsdf->EvaluatePdf(wo, wi);
            } else if (type == ECamera) {
                auto *camera = dynamic_cast<const Camera *>(ei.ep);
                Float _;
                camera->PdfEmission(ei.SpawnRay(wiNext), &_, &pdf);
            } else {
                AKARI_PANIC("???");
            }
            return PdfSAToArea(pdf, next);
        }
    };

    AKR_EXPORT size_t RandomWalk(const Scene &scene, MemoryArena &arena, Sampler &sampler, TransportMode mode, Ray ray,
                                 Spectrum beta, Float pdf, PathVertex *path, size_t maxDepth);

    AKR_EXPORT size_t TraceEyePath(const Scene &scene, MemoryArena &arena, const Camera &camera, const ivec2 &raster,
                                   Sampler &sampler, PathVertex *path, size_t maxDepth);

    AKR_EXPORT size_t TraceLightPath(const Scene &scene, MemoryArena &arena, Sampler &sampler, PathVertex *path,
                                     size_t maxDepth);

    AKR_EXPORT Spectrum ConnectPath(const Scene &scene, Sampler &sampler, PathVertex *eyePath, size_t t,
                                    PathVertex *lightPath, size_t s, vec2 *pRaster);

    template <typename T> class ScopedAssignment {
        T *target = nullptr;
        T backup;

      public:
        ScopedAssignment() : target(nullptr), backup(T()) {}
        ScopedAssignment(T *target, T value) : target(target) {
            if (target) {
                backup = *target;
                *target = value;
            }
        }
        ~ScopedAssignment() {
            if (target)
                *target = backup;
        }
        ScopedAssignment(ScopedAssignment &&) = delete;
        ScopedAssignment(const ScopedAssignment &) = delete;
        ScopedAssignment &operator=(const ScopedAssignment &) = delete;
        ScopedAssignment &operator=(ScopedAssignment &&other) noexcept {
            if (target)
                *target = backup;
            target = other.target;
            backup = other.backup;
            other.target = nullptr;
            return *this;
        }
    };

} // namespace Akari

#endif // AKARIRENDER_COREBIDIR_H
