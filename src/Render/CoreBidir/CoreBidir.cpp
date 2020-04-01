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
#include <Akari/Render/Plugins/CoreBidir.h>

namespace Akari {
    size_t RandomWalk(const Scene &scene, MemoryArena &arena, Sampler &sampler, TransportMode mode, Ray ray,
                      Spectrum beta, Float pdf, PathVertex *path, size_t maxDepth) {
        if (maxDepth == 0) {
            return 0;
        }
        Float pdfFwd = pdf;
        Float pdfRev = 0;
        size_t depth = 0;
        while (true) {
            Intersection intersection(ray);
            bool foundIntersection = scene.Intersect(ray, &intersection);
            if (!foundIntersection) {
                break;
            }
            auto &mesh = scene.GetMesh(intersection.meshId);
            int group = mesh.GetPrimitiveGroup(intersection.primId);
            const auto &materialSlot = mesh.GetMaterialSlot(group);

            auto material = materialSlot.material;
            if (!material) {
                Debug("no material!!\n");
                break;
            }
            Triangle triangle{};
            mesh.GetTriangle(intersection.primId, &triangle);
            const auto &p = intersection.p;
            auto &vertex = path[depth];
            auto &prev = path[depth - 1];
            vertex =
                PathVertex::CreateSurfaceVertex(&materialSlot, beta, -ray.d, p, triangle, intersection, pdfFwd, arena);
            vertex.pdfFwd = prev.PdfSAToArea(vertex.pdfFwd, vertex);
            vertex.si.ComputeScatteringFunctions(arena, mode, 1.0f);
            if (++depth >= maxDepth) {
                break;
            }
            auto &si = vertex.si;
            BSDFSample bsdfSample(sampler.Next1D(), sampler.Next2D(), si);
            si.bsdf->Sample(bsdfSample);
            pdfFwd = bsdfSample.pdf;
            pdfRev = si.bsdf->EvaluatePdf(bsdfSample.wi, bsdfSample.wo);
            auto wiW = bsdfSample.wi;
            beta *= bsdfSample.f * abs(dot(wiW, si.Ns)) / bsdfSample.pdf;
            if (bsdfSample.sampledType & BSDF_SPECULAR) {
                vertex.delta = true;
                pdfFwd = pdfRev = 0;
            }
            ray = si.SpawnRay(wiW);
            prev.pdfRev = vertex.PdfSAToArea(pdfRev, prev);
        }
        return depth;
    }

    size_t TraceEyePath(const Scene &scene, MemoryArena &arena, const Camera &camera, const ivec2 &raster,
                        Sampler &sampler, PathVertex *path, size_t maxDepth) {
        if (maxDepth == 0)
            return 0;
        CameraSample cameraSample;
        camera.GenerateRay(sampler.Next2D(), sampler.Next2D(), raster, &cameraSample);
        auto beta = Spectrum(1);
        Float pdfPos, pdfDir;
        camera.PdfEmission(cameraSample.primary, &pdfPos, &pdfDir);
        if (pdfDir <= 0 || pdfPos <= 0) {
            return 0;
        }
        path[0] = PathVertex::CreateCameraVertex(beta, &camera, cameraSample.primary.o, cameraSample.normal);
        return 1 + RandomWalk(scene, arena, sampler, TransportMode ::EImportance, cameraSample.primary, beta, pdfDir,
                              path + 1, maxDepth - 1);
    }

    size_t TraceLightPath(const Scene &scene, MemoryArena &arena, Sampler &sampler, PathVertex *path, size_t maxDepth) {
        if (maxDepth == 0)
            return 0;
        Float pdfLight;
        const auto *light = scene.SampleOneLight(sampler.Next1D(), &pdfLight);
        RayEmissionSample sample;
        light->SampleEmission(sampler.Next2D(), sampler.Next2D(), &sample);
        if (pdfLight <= 0 || sample.pdfPos <= 0 || sample.pdfDir <= 0 || sample.E.IsBlack()) {
            return 0;
        }
        path[0] =
            PathVertex::CreateLightVertex(sample.E, light, sample.ray.o, sample.normal, sample.pdfDir * sample.pdfPos);
        Spectrum beta = sample.E * abs(dot(sample.ray.d, sample.normal)) / (pdfLight * sample.pdfPos * sample.pdfDir);
        return 1 + RandomWalk(scene, arena, sampler, TransportMode::ERadiance, sample.ray, beta, sample.pdfDir,
                              path + 1, maxDepth - 1);
    }
    template <int Power>
    Float MisWeight(const Scene &scene, Sampler &sampler, PathVertex *eyePath, size_t t, PathVertex *lightPath,
                    size_t s, PathVertex &sampled) {

        if (s + t == 2)
            return 1;
        auto remap0 = [](Float x) { return x != 0 ? Akari::Power<Power>(x) : 1.0f; };
        (void)remap0;
        Float sumRi = 0;

        // p_0 ... pt  qs ... q_0
        auto *pt = t > 0 ? &eyePath[t - 1] : nullptr;
        auto *qs = s > 0 ? &lightPath[s - 1] : nullptr;

        auto *ptMinus = t > 1 ? &eyePath[t - 2] : nullptr;
        auto *qsMinus = s > 1 ? &lightPath[s - 2] : nullptr;

        //        if(s == 1){
        //            printf("a %f\n",sampled.pdfFwd);
        //        }
        ScopedAssignment<PathVertex> _a1;
        if (s == 1)
            _a1 = {qs, sampled};
        else if (t == 1)
            _a1 = {pt, sampled};
        //        if(s == 1){
        //            printf("b %f\n",lightPath[s-1].pdfFwd);
        //        }
        ScopedAssignment<bool> _a2, _a3;
        if (pt)
            _a2 = {&pt->delta, false};
        if (qs)
            _a3 = {&qs->delta, false};

        // now connect pt to qs

        // we need to compute pt->pdfRev
        // segfault ?
        ScopedAssignment<float> _a4;
        if (pt) {
            Float pdfRev;
            if (s > 0) {
                pdfRev = qs->Pdf(scene, qsMinus, *pt);
            } else {
                pdfRev = pt->PdfLightOrigin(scene, *ptMinus);
            }
            _a4 = {&pt->pdfRev, pdfRev};
        }

        // now ptMinus->pdfRev
        ScopedAssignment<float> _a5;
        if (ptMinus) {
            Float pdfRev;
            if (s > 0) {
                AKARI_ASSERT(qs);
                pdfRev = pt->Pdf(scene, qs, *ptMinus);
            } else {
                AKARI_ASSERT(pt);
                pdfRev = pt->PdfLight(scene, *ptMinus);
            }
            _a5 = {&ptMinus->pdfRev, pdfRev};
        }

        // now qs
        ScopedAssignment<float> _a6;
        if (qs) {
            AKARI_ASSERT(pt);
            _a6 = {&qs->pdfRev, pt->Pdf(scene, ptMinus, *qs)};
        }
        //        printf("%f\n",sampled.pdfFwd);
        // now qsMinus
        ScopedAssignment<float> _a7;
        if (qsMinus) {
            AKARI_ASSERT(pt);
            _a7 = {&qsMinus->pdfRev, qs->Pdf(scene, pt, *qsMinus)};
        }
        Float ri = 1;
        for (int i = (int)t - 1; i > 0; i--) {
            ri *= remap0(eyePath[i].pdfRev) / remap0(eyePath[i].pdfFwd);
            if (!eyePath[i].delta && !eyePath[i - 1].delta) {
                sumRi += ri;
            }
        }
        ri = 1;
        for (int i = (int)s - 1; i >= 0; i--) {
            ri *= remap0(lightPath[i].pdfRev) / remap0(lightPath[i].pdfFwd);
            bool delta = i > 0 ? lightPath[i - 1].delta : lightPath[i].IsDeltaLight();
            if (!lightPath[i].delta && !delta) {
                sumRi += ri;
            }
        }
        return 1.0 / (1.0 + sumRi);
    }
    Spectrum ConnectPath(const Scene &scene, Sampler &sampler, PathVertex *eyePath, size_t t, PathVertex *lightPath,
                         size_t s, vec2 *pRaster) {
        if (t > 1 && s != 0 && eyePath[t - 1].type == PathVertex::ELight)
            return Spectrum(0.f);
        Spectrum L(0);
        PathVertex sampled{};
        if (s == 0) {
            // eye path is complete
            auto &pt = eyePath[t - 1];
            L = pt.Le(scene, eyePath[t - 2]) * pt.beta;
        } else if (t == 1) {
            AKARI_ASSERT(s >= 1);
            auto &cameraVertex = eyePath[0];
            auto *camera = dynamic_cast<const Camera *>(cameraVertex.ei.ep);
            AKARI_ASSERT(camera);
            auto &qs = lightPath[s - 1];
            if (qs.IsConnectible()) {
                RayIncidentSample sample;
                VisibilityTester tester;
                AKARI_ASSERT(qs.getInteraction());
                camera->SampleIncidence(sampler.Next2D(), *qs.getInteraction(), &sample, &tester);
                *pRaster = sample.pos;
                sampled =
                    PathVertex::CreateCameraVertex(sample.I / sample.pdf, camera, tester.shadowRay.o, sample.normal);
                if (sample.pdf > 0 && !sample.I.IsBlack()) {

                    L = qs.beta * qs.f(sampled, TransportMode::EImportance) * sampled.beta;
                    if (qs.IsOnSurface()) {
                        L *= abs(dot(sample.wi, qs.Ns()));
                    }
                    if (!L.IsBlack()) {
                        L *= tester.Tr(scene);
                    }
                }
            }
        } else if (s == 1) {
            auto &lightVertex = lightPath[0];
            auto &pt = eyePath[t - 1];
            if (pt.IsConnectible()) {
                AKARI_ASSERT(t >= 1);
                auto *light = dynamic_cast<const Light *>(lightVertex.ei.ep);
                RayIncidentSample sample;
                VisibilityTester tester;
                AKARI_ASSERT(light);
                AKARI_ASSERT(pt.getInteraction());
                light->SampleIncidence(sampler.Next2D(), *pt.getInteraction(), &sample, &tester);
                sampled = PathVertex::CreateLightVertex(sample.I / (scene.PdfLight(light) * sample.pdf), light,
                                                        tester.shadowRay.o, sample.normal);
                sampled.pdfFwd = sampled.PdfLightOrigin(scene, pt);
                if (sample.pdf > 0 && !sample.I.IsBlack()) {
                    L = pt.beta * pt.f(sampled, TransportMode::ERadiance) * sampled.beta;
                    if (pt.IsOnSurface()) {
                        L *= abs(dot(sample.wi, pt.Ns()));
                    }
                    if (!L.IsBlack()) {
                        L *= tester.Tr(scene);
                    }
                }
            }
        } else {
            auto &pt = eyePath[t - 1];
            auto &qs = lightPath[s - 1];
            if (pt.IsConnectible() && qs.IsConnectible()) {
                VisibilityTester tester(*pt.getInteraction(), *qs.getInteraction());
                auto g = PathVertex::G(scene, pt, qs);
                L = g * qs.f(pt, TransportMode::EImportance) * pt.f(qs, TransportMode::ERadiance) * pt.beta * qs.beta;
                if (!L.IsBlack())
                    L *= tester.Tr(scene);
            }
        }
        Float misWeight = 1.0f / (s + t);
        misWeight = MisWeight<1>(scene, sampler, eyePath, t, lightPath, s, sampled);
        AKARI_ASSERT(misWeight >= 0);
        L *= misWeight;
        return L;
    }
} // namespace Akari