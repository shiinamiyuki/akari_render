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

#ifndef AKARIRENDER_PROPERTY_HPP
#define AKARIRENDER_PROPERTY_HPP

#include <Akari/Core/Akari.h>
#include <Akari/Core/Component.h>
#include <Akari/Core/Math.h>
#include <Akari/Core/Spectrum.h>
#include <functional>
#include <memory>
#include <string>
#include <variant>

namespace nlohmann {
    template <> struct adl_serializer<Akari::fs::path> {
        static void from_json(const json &j,  Akari::fs::path &path) { path = Akari::fs::path(j.get<std::string>()); }

        static void to_json(json &j, const Akari::fs::path &path) { j = path.string(); }
    };
} // namespace nlohmann
namespace Akari {
    class Component;

    inline void from_json(const json &j, fs::path &path) { path = fs::path(j.get<std::string>()); }
    inline void to_json(json &j, const fs::path &path) { j = path.string(); }

    template <typename... T> using TRefVariant = std::variant<std::reference_wrapper<T>...>;
    struct Variant : std::variant<bool, int, float, ivec2, vec2, vec3, fs::path, Angle<float>, Angle<vec3>, Spectrum,
                                  std::string, fs::path, std::shared_ptr<Component>, std::vector<Variant>> {};

    struct Property {
        const char *name{};
        Variant value;
        void SetModified() { _dirty = true; }
        [[nodiscard]] bool IsModified() const { return _dirty; }

      private:
        bool _dirty = false;
    };
    class PropertyVisitor {
      public:
        virtual void visit(Property &property) = 0;
    };

} // namespace Akari
#endif // AKARIRENDER_PROPERTY_HPP
