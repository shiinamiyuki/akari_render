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
            vertex = PathVertex::CreateSurfaceVertex(beta, -ray.d, p, triangle, intersection, pdfFwd);
            vertex.pdfFwd = prev.PdfSAToArea(prev.pdfFwd, vertex);
            material->computeScatteringFunctions(&vertex.si, arena, mode, 1.0f);
            if (++depth >= maxDepth) {
                break;
            }
            auto &si = vertex.si;
            BSDFSample bsdfSample(sampler.Next1D(), sampler.Next2D(), si);
            si.bsdf->Sample(bsdfSample);
            pdfRev = si.bsdf->EvaluatePdf(bsdfSample.wi, bsdfSample.wo);
            auto wiW = si.bsdf->LocalToWorld(bsdfSample.wi);
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
        path[0] = PathVertex::CreateCameraVertex(beta, &camera, cameraSample.primary.o, cameraSample.normal);
        return 1 + RandomWalk(scene, arena, sampler, TransportMode ::EImportance, cameraSample.primary, beta, pdfDir,
                              path + 1, maxDepth - 1);
    }

    size_t TraceLightPath(const Scene &scene, MemoryArena &arena, Sampler &sampler, PathVertex *path, size_t maxDepth) {
        Float pdfLight;
        const auto *light = scene.SampleOneLight(sampler.Next1D(), &pdfLight);
        RayEmissionSample sample;
        light->SampleEmission(sampler.Next2D(), sampler.Next2D(), &sample);
        Spectrum beta = sample.E * abs(dot(sample.ray.d, sample.normal)) / (pdfLight * sample.pdfPos * sample.pdfDir);
        path[0] = PathVertex::CreateLightVertex(beta, light, sample.ray.o, sample.normal);
        return 1 + RandomWalk(scene, arena, sampler, TransportMode::ERadiance, sample.ray, beta, sample.pdfDir,
                              path + 1, maxDepth - 1);
    }

    Spectrum ConnectPath(const Scene &scene, Sampler &sampler, PathVertex *eyePath, size_t t, PathVertex *lightPath,
                         size_t s, vec2 *pRaster) {
        if (t > 1 && s != 0 && eyePath[t - 1].type == PathVertex::ELight)
            return Spectrum(0.f);
        Spectrum L(0);
        PathVertex sampled{};
        if (s == 0) {
            // eye path is complete
            auto &pt = lightPath[t - 1];
            L = pt.Le(scene, lightPath[t - 2]) * pt.beta;
        } else if (t == 1) {
            auto &cameraVertex = eyePath[0];
            auto *camera = dynamic_cast<const Camera *>(cameraVertex.ei.ep);
            auto &qs = lightPath[s - 1];
            RayIncidentSample sample;
            VisibilityTester tester;
            camera->SampleIncidence(sampler.Next2D(), *qs.getInteraction(), &sample, &tester);
            *pRaster = sample.pos;
            if (sample.pdf > 0 && !sample.I.IsBlack()) {
                sampled =
                    PathVertex::CreateCameraVertex(sample.I / sample.pdf, camera, tester.shadowRay.o, sample.normal);
                if (qs.IsOnSurface()) {
                    L *= abs(dot(sample.wi, qs.Ns()));
                }
                if (!L.IsBlack()) {
                    L *= tester.Tr(scene);
                }
            }
        } else if (s == 1) {
            auto &lightVertex = lightPath[0];
            auto &pt = lightPath[t - 1];
            auto *light = dynamic_cast<const Light *>(lightVertex.ei.ep);
            RayIncidentSample sample;
            VisibilityTester tester;
            light->SampleIncidence(sampler.Next2D(), *pt.getInteraction(), &sample, &tester);
            if (sample.pdf > 0 && !sample.I.IsBlack()) {
                sampled =
                    PathVertex::CreateLightVertex(sample.I / sample.pdf, light, tester.shadowRay.o, sample.normal);
                sampled.pdfFwd = sampled.PdfLightOrigin(scene, pt);
                if (pt.IsOnSurface()) {
                    L *= abs(dot(sample.wi, pt.Ns()));
                }
                if (!L.IsBlack()) {
                    L *= tester.Tr(scene);
                }
            }
        }
    }
} // namespace Akari