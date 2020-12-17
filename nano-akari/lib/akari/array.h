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
#pragma once
#include <akari/thread.h>
namespace akari {
    template <typename T, class Allocator = std::allocator<T>>
    class Array2D {
      protected:
        ivec2 dimension_;
        std::vector<T, Allocator> data_;

      public:
        T *data() { return data_.data(); }
        const T *data() const { return data_.data(); }
        Array2D() : Array2D(ivec2(1, 1)) {}
        Array2D(const ivec2 &size) : Array2D(size, Allocator()) {}
        static Array2D ones(const ivec2 &size){
            Array2D tmp(size);
            tmp.fill(T(1.0));
            return tmp;
        }
        static Array2D zeros(const ivec2 &size){
            Array2D tmp(size);
            tmp.fill(T(0.0));
            return tmp;
        }
        Array2D(const ivec2 &size, Allocator alloc) : dimension_(size), data_(size.x * size.y, alloc) {}
        void fill(const T &v) {
            for (auto &i : data_) {
                i = v;
            }
        }
#define ARRAY_OP(op, assign_op)                                                                                        \
    Array2D &operator assign_op(const T &rhs) {                                                                        \
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
        auto tmp = Array2D(rhs.dimension());                                                                           \
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
        // template <typename = std::enable_if_t<!vec_trait<T>::is_vector>>
        Array2D safe_div(const Array2D &rhs) const {
            AKR_ASSERT(glm::all(glm::equal(rhs.dimension_, dimension_)));
            auto tmp = Array2D(rhs.dimension());
            if constexpr (!vec_trait<T>::is_vector) {
                for (int i = 0; i < hprod(dimension_); i++) {
                    tmp.data_[i] = data_[i] / (rhs.data_[i] == T(0) ? T(1) : rhs.data_[i]);
                }
            } else {
                using V = typename vec_trait<T>::value_type;
                for (int i = 0; i < hprod(dimension_); i++) {
                    tmp.data_[i] = data_[i] / T(select(glm::equal(rhs.data_[i], T(0.0)), T(1.0), rhs.data_[i]));
                }
            }
            return tmp;
        }

        ivec2 dimension() const { return dimension_; }
        T sum() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return x + y; });
        }
        T prod() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return x * y; });
        }
        T max() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return std::max(x, y); });
        }
        T min() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return std::min(x, y); });
        }
        void resize(const ivec2 &size) {
            dimension_ = size;
            data_.resize(dimension_[0] * dimension_[1]);
        }
        template <class F>
        void map_inplace(F &&f) {
            thread::parallel_for_each(data_.begin(), data_.end(), [&](auto &item) { item = f(item); });
        }
        template <class F>
        Array2D map(F &&f) const {
            Array2D out(dimension());
            thread::parallel_for(thread::blocked_range<2>(out.dimension(), ivec2(64)),
                                 [&](ivec2 id) { out(id) = f((*this)(id)); });
            return out;
        }
    };

    template <class T, class A, class A2>
    Array2D<T, A> convolve(const Array2D<T, A> &image, const Array2D<T, A2> &kernel, const ivec2 &stride) {
        Array2D<T, A> out(image.dimension() / stride);
        thread::parallel_for(thread::blocked_range<2>(out.dimension(), ivec2(64)), [&](ivec2 id, uint32_t) {
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

    template <class T, class Allocator = std::allocator<T>>
    class Array3D {
      protected:
        ivec3 dimension_;
        std::vector<T, Allocator> data_;

      public:
        T *data() { return data_.data(); }
        const T *data() const { return data_.data(); }
        Array3D() : Array3D(ivec3(1)) {}
        Array3D(const ivec3 &size) : Array3D(size, Allocator()) {}
        Array3D(const ivec3 &size, Allocator alloc) : dimension_(size), data_(size.x * size.y * size.z, alloc) {}
        template <class U, class A>
        Array3D(const Array3D<U, A> &rhs, Allocator alloc)
            : dimension_(rhs.dimension()), data_(hprod(dimension_), alloc) {
            for (int i = 0; i < hprod(dimension_); i++) {
                data_[i] = rhs.data()[i];
            }
        }
        void fill(const T &v) {
            for (auto &i : data_) {
                i = v;
            }
        }
#define ARRAY_OP(op, assign_op)                                                                                        \
    Array3D &operator assign_op(const T &rhs) {                                                                        \
        AKR_ASSERT(glm::all(glm::equal(rhs.dimension_, dimension_)));                                                  \
        for (int i = 0; i < hprod(dimension_); i++) {                                                                  \
            data_[i] assign_op rhs;                                                                                    \
        }                                                                                                              \
        return *this;                                                                                                  \
    }                                                                                                                  \
    Array3D &operator assign_op(const Array3D &rhs) {                                                                  \
        AKR_ASSERT(glm::all(glm::equal(rhs.dimension_, dimension_)));                                                  \
        for (int i = 0; i < hprod(dimension_); i++) {                                                                  \
            data_[i] assign_op rhs.data_[i];                                                                           \
        }                                                                                                              \
        return *this;                                                                                                  \
    }                                                                                                                  \
    Array3D operator op(const Array3D &rhs) const {                                                                    \
        auto tmp = *this;                                                                                              \
        tmp assign_op rhs;                                                                                             \
        return tmp;                                                                                                    \
    }                                                                                                                  \
    Array3D operator op(const T &rhs) const {                                                                          \
        auto tmp = *this;                                                                                              \
        tmp assign_op rhs;                                                                                             \
        return tmp;                                                                                                    \
    }                                                                                                                  \
    friend Array3D operator op(const T &lhs, const Array3D &rhs) {                                                     \
        auto tmp = Array3D(rhs.dimension());                                                                           \
        tmp.fill(lhs);                                                                                                 \
        tmp assign_op rhs;                                                                                             \
        return tmp;                                                                                                    \
    }
        const T &operator()(int x, int y, int z) const {
            return data_[x + y * dimension().x + z * dimension().x * dimension().y];
        }
        T &operator()(int x, int y, int z) { return data_[x + y * dimension().x + z * dimension().x * dimension().y]; }
        const T &operator()(ivec3 id) const { return (*this)(id.x, id.y, id.z); }
        T &operator()(ivec3 id) { return (*this)(id.x, id.y, id.z); }
        ARRAY_OP(+, +=)
        ARRAY_OP(-, -=)
        ARRAY_OP(*, *=)
        ARRAY_OP(/, /=)
#undef ARRAY_OP
        ivec3 dimension() const { return dimension_; }

        T sum() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return x + y; });
        }
        T prod() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return x * y; });
        }
        T max() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return std::max(x, y); });
        }
        T min() const {
            return thread::parallel_reduce(data_.begin(), data_.end(), T(0.0), [](T x, T y) { return std::min(x, y); });
        }
        void resize(const ivec3 &size) {
            dimension_ = size;
            data_.resize(dimension_[0] * dimension_[1] * dimension_[2]);
        }
        template <class F>
        void map_inplace(F &&f) {
            thread::parallel_for_each(data_.begin(), data_.end(), [&](auto &item) { item = f(item); });
        }
        template <class F>
        Array3D map(F &&f) const {
            Array3D out(dimension());
            thread::parallel_for(thread::blocked_range<3>(out.dimension(), ivec3(64)),
                                 [&](ivec3 id) { out(id) = f((*this)(id)); });
            return out;
        }
    };

    template <class T, class A, class A2>
    Array3D<T, A> convolve(const Array3D<T, A> &image, const Array3D<T, A2> &kernel, const ivec3 &stride) {
        Array3D<T, A> out(image.dimension() / stride);
        thread::parallel_for(thread::blocked_range<3>(out.dimension(), ivec3(16)), [&](ivec3 id, uint32_t) {
            T sum = T(0.0);
            for (int z = 0; z < kernel.dimension().z; z++) {
                for (int y = 0; y < kernel.dimension().y; y++) {
                    for (int x = 0; x < kernel.dimension().x; x++) {
                        sum += image(stride * id + ivec3(x, y, z)) * kernel(x, y, z);
                    }
                }
            }
            out(id) = sum;
        });
        return out;
    }

} // namespace akari