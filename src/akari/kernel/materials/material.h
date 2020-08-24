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

#pragma once

#include <akari/common/variant.h>
#include <akari/common/math.h>

namespace akari {
    AKR_VARIANT struct BSDFSample {
        AKR_IMPORT_CORE_TYPES()
        Vector3f wi = Vector3f(0);
        Float pdf = 0.0;
    };
    AKR_VARIANT struct BSDFSampleContext {
        AKR_IMPORT_CORE_TYPES()
        Vector3f wi;
        Normal3f ng, ns;
        Point3f p;
    };
    AKR_VARIANT struct DiffuseClosure {};

    AKR_VARIANT struct SumBSDF { int bsdfs[2]; };

    AKR_VARIANT class BSDFClosure {};

    AKR_VARIANT class BSDF {

        AKR_IMPORT_TYPES(BSDFClosure)
        std::array<ABSDFClosure, 16> closures;
        Normal3f ng, ns;
        Frame3f frame;
        int num_closures = 0;

      public:
        explicit BSDF(const Normal3f &ng, const Normal3f &ns) : ng(ng), ns(ns) { frame = Frame3f(ns); }
        void add_closure(const ABSDFClosure &closure) { closures[num_closures++] = closure; }
    };

    AKR_VARIANT class Material {};

} // namespace akari