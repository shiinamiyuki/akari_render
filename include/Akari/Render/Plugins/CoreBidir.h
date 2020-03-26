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
            Interaction si;
            EndPointInteraction ei;
        };
        Float pdfFwd = 0, pdfRev = 0;
    };

    AKR_EXPORT size_t RandomWalk(const Scene &scene, Sampler &sampler, const Ray &ray, Spectrum beta, Float pdf,
                                 PathVertex *path, size_t maxDepth);

    AKR_EXPORT size_t TraceEyePath(const Scene &scene, const Camera &camera, Sampler &sampler, PathVertex *path,
                                   size_t maxDepth);

    AKR_EXPORT size_t TraceLightPath(const Scene &scene, Sampler &sampler, PathVertex *path, size_t maxDepth);

} // namespace Akari

#endif // AKARIRENDER_COREBIDIR_H
