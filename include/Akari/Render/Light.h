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

#include <Akari/Core/Component.h>
#include <Akari/Core/Spectrum.h>
#include <Akari/Render/EndPoint.h>
#include <Akari/Render/Geometry.hpp>
#include <Akari/Render/Scene.h>

namespace Akari {
    struct LightSample : RayIncidentSample {};

    struct LightRaySample : RayEmissionSample {};

    enum class LightType : uint32_t {
        ENone = 1u,
        EDeltaPosition = 1u << 1u,
        EDeltaDirection = 1u << 2u,
    };
    class Light : public EndPoint {
      public:
        virtual Float power() const = 0;
        virtual Spectrum Li(const vec3 &wo, const vec2 &uv) const = 0;
        virtual LightType get_light_type() const = 0;
        static bool is_delta(LightType type) {
            return (uint32_t)type & ((uint32_t)LightType ::EDeltaPosition | (uint32_t)LightType::EDeltaDirection);
        }
        //        virtual void SampleLe(const Point2f &u1, const Point2f &u2, LightRaySample &sample) = 0;
    };

} // namespace Akari

#endif // AKARIRENDER_LIGHT_H
