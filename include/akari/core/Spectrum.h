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

#ifndef AKARIRENDER_SPECTRUM_H
#define AKARIRENDER_SPECTRUM_H

#include <akari/core/math.h>
#include <json.hpp>
namespace akari {
    template <size_t N> struct CoefficientSpectrum : public vec<N, float, defaultp> {
        using vec<N, float, defaultp>::vec;
        using Base = vec<N, float, defaultp>;
        using Self = CoefficientSpectrum<N>;\

        using vec<N, float, defaultp>::operator[];

        static const int nChannel = N;
        CoefficientSpectrum(const Base &v) : Base(v) {}
        Self &operator+=(const Self &v) {
            static_cast<Base &>(*this) += Base(v);
            return *this;
        }

        Self &operator-=(const Self &v) {
            static_cast<Base &>(*this) -= Base(v);
            return *this;
        }

        Self &operator*=(const Self &v) {
            static_cast<Base &>(*this) *= Base(v);
            return *this;
        }

        Self &operator/=(const Self &v) {
            static_cast<Base &>(*this) /= Base(v);
            return *this;
        }

        Self &operator*=(const float &v) {
            static_cast<Base &>(*this) *= v;
            return *this;
        }

        Self &operator/=(const float &v) {
            static_cast<Base &>(*this) /= v;
            return *this;
        }

        friend Self operator+(const Self &lhs, const Self &rhs) {
            Self tmp = lhs;
            tmp += rhs;
            return tmp;
        }

        friend Self operator-(const Self &lhs, const Self &rhs) {
            Self tmp = lhs;
            tmp -= rhs;
            return tmp;
        }

        friend Self operator*(const Self &lhs, const Self &rhs) {
            Self tmp = lhs;
            tmp *= rhs;
            return tmp;
        }

        friend Self operator/(const Self &lhs, const Self &rhs) {
            Self tmp = lhs;
            tmp /= rhs;
            return tmp;
        }

        friend Self operator*(const Self &lhs, const float &rhs) {
            Self tmp = lhs;
            tmp *= rhs;
            return tmp;
        }

        friend Self operator/(const Self &lhs, const float &rhs) {
            Self tmp = lhs;
            tmp /= rhs;
            return tmp;
        }
        friend Self operator*(const float &rhs, const Self &lhs) {
            Self tmp = lhs;
            tmp *= rhs;
            return tmp;
        }

        friend Self operator/(const float &rhs, const Self &lhs) {
            Self tmp = lhs;
            tmp /= rhs;
            return tmp;
        }
        [[nodiscard]] float luminance() const {
            return 0.2126 * (*this)[0] + 0.7152 * (*this)[1] + 0.0722 * (*this)[2];
        }
        [[nodiscard]] Self remove_nans() const {
            Self tmp;
            for (size_t i = 0; i < N; i++) {
                auto x = (*this)[i];
                if (std::isnan(x)) {
                    tmp[i] = 0;
                } else {
                    tmp[i] = x;
                }
            }
            return tmp;
        }
        [[nodiscard]] float is_black() const { return MaxComp(*this) <= 0; }
    };

    using RGBSpectrum = CoefficientSpectrum<3>;

    inline void from_json(const nlohmann::json &j, RGBSpectrum &s) {
        for (int i = 0; i < RGBSpectrum::nChannel; i++) {
            s[i] = j[i].get<float>();
        }
    }
    inline void to_json(nlohmann::json &j, const RGBSpectrum &s) {
        for (int i = 0; i < RGBSpectrum::nChannel; i++) {
            j[i] = s[i];
        }
    }

} // namespace akari
#endif // AKARIRENDER_SPECTRUM_H
