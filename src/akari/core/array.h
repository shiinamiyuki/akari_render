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
#include <akari/core/memory.h>
#include <akari/core/parallel.h>

namespace akari {
    template <typename T, class Allocator = std::allocator<T>>
    class Array2D {
      protected:
        std::vector<T, Allocator> data_;
        ivec2 dimension_;

      public:
        T *data() { return data_.data(); }
        const T *data() const { return data_.data(); }
        Array2D() : Array2D(ivec2(1, 1)) {}
        Array2D(const ivec2 &size) : Array2D(size, Allocator()) {}
        Array2D(const ivec2 &size, Allocator alloc) : dimension_(size), data_(size.x * size.y, alloc) {}
        void fill(const T &v) {
            for (auto &i : data_) {
                i = v;
            }
        }
#define ARRAY_OP(op, assign_op)                                                                                        \
    Array2D &operator assign_op(const T &rhs) {                                                                        \
        AKR_ASSERT(glm::all(glm::equal(rhs.dimension_, dimension_)));                                                  \
        for (int i = 0; i < hprod(dimension_); i++) {                                                                  \
            data_[i] assign_op rhs;                                                                                    \
        }                                                                                                              \
        return *this;                                                                                                  \
    }                                                                                                                  \
    Array2D &operator assign_op(const Array2D &rhs) {                                                                  \
        AKR_ASSERT(glm::all(glm::equal(rhs.dimension_, dimension_)));                                                  \
        for (int i = 0; i < hprod(dimension_); i++) {                                                                  \
            data_[i] assign_op rhs.data_[i];                                                                           \
        }                                                                                                              \
        return *this;                                                                                                  \
    }                                                                                                                  \
    Array2D operator op(const Array2D &rhs) const {                                                                    \
        auto tmp = *this;                                                                                              \
        tmp assign_op rhs;                                                                                             \
        return tmp;                                                                                                    \
    }                                                                                                                  \
    Array2D operator op(const T &rhs) const {                                                                          \
        auto tmp = *this;                                                                                              \
        tmp assign_op rhs;                                                                                             \
        return tmp;                                                                                                    \
    }                                                                                                                  \
    friend Array2D operator op(const T &lhs, const Array2D &rhs) {                                                     \
        auto tmp = Array2D(dimension_);                                                                                \
        tmp.fill(lhs);                                                                                                 \
        tmp assign_op rhs;                                                                                             \
        return tmp;                                                                                                    \
    }
        const T &operator()(int x, int y) const { return data_[x + y * dimension().x]; }
        T &operator()(int x, int y) { return data_[x + y * dimension().x]; }
        const T &operator()(ivec2 id) const { return (*this)(id.x, id.y); }
        T &operator()(ivec2 id) { return (*this)(id.x, id.y); }
        ARRAY_OP(+, +=)
        ARRAY_OP(-, -=)
        ARRAY_OP(*, *=)
        ARRAY_OP(/, /=)
#undef ARRAY_OP
        ivec2 dimension() const { return dimension_; }
        static Array2D convolve(const Array2D &image, const Array2D &kernel, const ivec2 &stride) {
            Array2D out(image.dimension() / stride);
            bool parallel = hprod(out.dimension()) >= 128;
            tiled_for_2d(out.dimension(), parallel, [&](ivec2 id) {
                T sum = T(0.0);
                for (int y = 0; y < kernel.dimension().y; y++) {
                    for (int x = 0; x < kernel.dimension().x; x++) {
                        sum += image(stride * id + ivec2(x, y)) * kernel(x, y);
                    }
                }
                out(id) = sum;
            });
            return out;
        }
        T sum() const {
            return std::reduce(std::execution::par, data_.begin(), data_.end(), T(0.0), [](T x, T y) { return x + y; });
        }
        T prod() const {
            return std::reduce(std::execution::par, data_.begin(), data_.end(), T(0.0), [](T x, T y) { return x * y; });
        }
        T max() const {
            return std::reduce(std::execution::par, data_.begin(), data_.end(), T(0.0),
                               [](T x, T y) { return max(x, y); });
        }
        T min() const {
            return std::reduce(std::execution::par, data_.begin(), data_.end(), T(0.0),
                               [](T x, T y) { return min(x, y); });
        }
        void resize(const ivec2 &size) {
            dimension_ = size;
            data_.resize(dimension_[0] * dimension_[1]);
        }
        template <class F>
        void map_inplace(F &&f) {
            std::for_each(std::execution::par, data_.begin(), data_.end(), [&](auto &item) { item = f(item); });
        }
        template <class F>
        Array2D map(F &&f) const {
            Array2D out(dimension());
            bool parallel = hprod(out.dimension()) >= 128;
            tiled_for_2d(out.dimension(), parallel, [&](ivec2 id) { out(id) = f((*this)(id)); });
            return out;
        }
    };
} // namespace akari