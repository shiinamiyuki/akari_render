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

#ifndef AKARIRENDER_IMAGE_HPP
#define AKARIRENDER_IMAGE_HPP

#include <list>
#include <vector>
#include <akari/core/akari.h>
#include <akari/core/math.h>
#include <akari/core/color.h>
#include <akari/core/memory.h>
#include <akari/core/array.h>

namespace akari {

    template <class T>
    class TImage {
        using Alloc = astd::pmr::polymorphic_allocator<T>;
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
               Allocator<> allocator = Allocator<>())
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
    inline Image rgba_image(const ivec2 &dim) { return Image({"R", "G", "B", "A"}, dim); }
    inline Image rgb_image(const ivec2 &dim) { return Image({"R", "G", "B"}, dim); }
    inline bool is_rgb_image(const Image &image) {
        return image.channels() == 3 && image.channel_name(0) == "R" && image.channel_name(1) == "G" &&
               image.channel_name(2) == "B";
    }

    struct alignas(16) RGBA {
        RGB rgb;
        float alpha;
        RGBA() = default;
        RGBA(vec3 rgb, float alpha) : rgb(rgb), alpha(alpha) {}
    };

    class AKR_EXPORT ImageWriter {
      public:
        virtual bool write(const Image &image, const fs::path &) = 0;
        virtual ~ImageWriter() = default;
    };

    class AKR_EXPORT ImageReader {
      public:
        virtual std::shared_ptr<Image> read(const fs::path &) = 0;
        virtual ~ImageReader() = default;
    };

    AKR_EXPORT std::shared_ptr<ImageWriter> ldr_image_writer();
    AKR_EXPORT std::shared_ptr<ImageReader> default_image_reader();

} // namespace akari

#endif // AKARIRENDER_IMAGE_HPP
