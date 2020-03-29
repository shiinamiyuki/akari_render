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
            vertex = PathVertex::CreateSurfaceVertex(-ray.d, p, triangle, intersection, pdfFwd);
            prev.pdfFwd = prev.PdfSAToArea(prev.pdfFwd, vertex);
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
} // namespace Akari