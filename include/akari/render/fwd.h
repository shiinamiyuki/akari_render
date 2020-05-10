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

#ifndef AKARIRENDER_FWD_H
#define AKARIRENDER_FWD_H
namespace akari {
    template <typename Float, typename Spectrum> struct Ray;
    template <typename Float, typename Spectrum> struct Intersection;
    template <typename Float, typename Spectrum> struct Triangle;
    template <typename Float, typename Spectrum> struct ShadingPoint;
    template <typename Float, typename Spectrum> class BSDF;
    template <typename Float, typename Spectrum> class BSDFComponent;
    template <typename Float, typename Spectrum> class Material;
    template <typename Float, typename Spectrum> class EndPoint;
    template <typename Float, typename Spectrum> class Camera;
    template <typename Float, typename Spectrum> class Texture;
    template <typename Float, typename Spectrum> class Integrator;
    template <typename Float, typename Spectrum> struct SurfaceSample;
    template <typename Float, typename Spectrum> struct Interaction;
    template <typename Float, typename Spectrum> struct SurfaceInteraction;
    template <typename Float, typename Spectrum> struct EndPointInteraction;
    template <typename Float, typename Spectrum> struct VolumeInteraction;
    template <typename Float, typename Spectrum> struct MaterialSlot;
    template <typename Float, typename Spectrum> class Light;
} // namespace akari
#endif // AKARIRENDER_FWD_H
