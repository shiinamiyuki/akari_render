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

#ifndef AKARIRENDER_LIGHT_H
#define AKARIRENDER_LIGHT_H

#include <akari/core/component.h>
#include <akari/core/spectrum.h>
#include <akari/render/endpoint.h>
#include <akari/render/geometry.hpp>
#include <akari/render/scene.h>

namespace akari {
    template <typename Float, typename Spectrum> struct LightSample : RayIncidentSample<Float, Spectrum> {};

    template <typename Float, typename Spectrum> struct LightRaySample : RayEmissionSample<Float, Spectrum> {};

    enum class LightType : uint32_t {
        ENone = 1u,
        EDeltaPosition = 1u << 1u,
        EDeltaDirection = 1u << 2u,
    };
    template <typename Float, typename Spectrum> class Light : public EndPoint<Float, Spectrum> {
      public:
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Ray, Interaction, VisibilityTester, RayEmissionSample, RayIncidentSample)
        using LightTypeV = replace_scalar_t<Float, LightType>;
        virtual Float power() const = 0;
        virtual Spectrum Li(const Vector3f &wo, const Vector2f &uv) const = 0;
        [[nodiscard]] virtual LightTypeV get_light_type() const = 0;
        static bool is_delta(LightType type) {
            return (uint32_t)type & ((uint32_t)LightType ::EDeltaPosition | (uint32_t)LightType::EDeltaDirection);
        }
        //        virtual void SampleLe(const Point2f &u1, const Point2f &u2, LightRaySample &sample) = 0;
    };

} // namespace akari

#endif // AKARIRENDER_LIGHT_H
