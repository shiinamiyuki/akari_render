// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AKARIRENDER_IMAGE_HPP
#define AKARIRENDER_IMAGE_HPP

#include <list>
#include <vector>
#include <akari/array.h>

namespace akari {

    template <class T>
    class TImage {
        using Alloc = std::allocator<T>; // astd::pmr::polymorphic_allocator<T>;
        using Array = Array3D<T, Alloc>;
        Array array;

        std::vector<std::string> channel_names_;

      public:
        const T *data() const { return array.data(); }
        T *data() { return array.data(); }
        [[nodiscard]] const Array &array3d() const { return array; }
        [[nodiscard]] Array &array3d() { return array; }
        const std::string &channel_name(int ch) const { return channel_names_[ch]; }
        int channels() const { return (int)channel_names_.size(); }
        TImage(const std::vector<std::string> channel_names_, const ivec2 &dim = ivec2(1),
               Alloc allocator = Alloc())
            : array(ivec3((int)channel_names_.size(), dim.x, dim.y)), channel_names_(channel_names_) {}
        [[nodiscard]] const T &operator()(int x, int y, int ch) const {
            x = std::clamp(x, 0, resolution()[0] - 1);
            y = std::clamp(y, 0, resolution()[1] - 1);
            return array(ch, x, y);
        }

        [[nodiscard]] T &operator()(int x, int y, int ch) {
            x = std::clamp(x, 0, resolution()[0] - 1);
            y = std::clamp(y, 0, resolution()[1] - 1);
            return array(ch, x, y);
        }

        [[nodiscard]] const T &operator()(float x, float y, int ch) const { return (*this)(vec2(x, y), ch); }

        [[nodiscard]] T &operator()(float x, float y, int ch) { return (*this)(vec2(x, y), ch); }

        [[nodiscard]] const T &operator()(const ivec2 &p, int ch) const { return (*this)(p.x, p.y, ch); }

        [[nodiscard]] T &operator()(const ivec2 &p, int ch) { return (*this)(p.x, p.y, ch); }

        [[nodiscard]] const T &operator()(const vec2 &p, int ch) const {
            return (*this)(ivec2(p * vec2(resolution())), ch);
        }

        [[nodiscard]] T &operator()(const vec2 &p, int ch) { return (*this)(ivec2(p * vec2(resolution())), ch); }

        void resize(const ivec2 &size) { array.resize(ivec3(array.dimension().x, size.x, size.y)); }

        void fill(const T &v) { array.fill(v); }

        [[nodiscard]] ivec2 resolution() const { return ivec2(array.dimension().y, array.dimension().z); }
    };
    using Image = TImage<float>;
    Image array2d_to_rgb(const Array2D<Color3f> & array);
    inline Image rgba_image(const ivec2 &dim) { return Image({"R", "G", "B", "A"}, dim); }
    inline Image rgb_image(const ivec2 &dim) { return Image({"R", "G", "B"}, dim); }
    inline bool is_rgb_image(const Image &image) {
        return image.channels() == 3 && image.channel_name(0) == "R" && image.channel_name(1) == "G" &&
               image.channel_name(2) == "B";
    }
    using RGB = Color3f;
    struct alignas(16) RGBA {
        RGB rgb;
        float alpha;
        RGBA() = default;
        RGBA(vec3 rgb, float alpha) : rgb(rgb), alpha(alpha) {}
    };

    bool write_ldr(const Image &image, const fs::path &);
    bool write_hdr(const Image &image, const fs::path &);
    Image read_generic_image(const fs::path &);
} // namespace akari

#endif // AKARIRENDER_IMAGE_HPP
