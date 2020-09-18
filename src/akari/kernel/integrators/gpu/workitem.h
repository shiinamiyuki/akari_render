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
#include <akari/kernel/sampler.h>
#include <akari/kernel/camera.h>
namespace akari {
    AKR_VARIANT struct PathState {
        AKR_IMPORT_TYPES()
        int pixel;
        Sampler<C> sampler;
        Spectrum L;
        Spectrum beta;
    };
    AKR_VARIANT struct CameraRayWorkItem {
        AKR_IMPORT_TYPES()
        CameraSample<C> sample;
    };
    AKR_VARIANT struct ClosestHitWorkItem {
        AKR_IMPORT_TYPES();
        int pixel;
        Intersection<C> intersection;
    };
    AKR_VARIANT struct MissWorkItem {
        AKR_IMPORT_TYPES();
        int pixel;
    };
    AKR_VARIANT struct AnyHitWorkItem {
        AKR_IMPORT_TYPES();
        int pixel;
        bool hit = false;
    };
    AKR_VARIANT struct RayWorkItem {
        AKR_IMPORT_TYPES()
        int pixel;
        Ray3f ray;
    };
    AKR_VARIANT struct ShadowRayWorkItem {
        AKR_IMPORT_TYPES()
        int pixel;
        Ray3f ray;
    };
} // namespace akari