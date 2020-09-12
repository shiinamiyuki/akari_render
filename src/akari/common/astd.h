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

#include <type_traits>
#include <new>
#include <akari/common/panic.h>
#include <akari/common/def.h>
#include <math.h>
namespace akari {
    // port some nessary stl class to CUDA
    namespace astd {
        template <typename T1, typename T2>
        struct pair {
            T1 first;
            T2 second;
        };
        template <typename T1, typename T2>
        AKR_XPU pair<T1, T2> make_pair(T1 &&a, T2 &&b) {
            return pair{a, b};
        }
        struct nullopt_t {};
        inline constexpr nullopt_t nullopt{};
        template <typename T>
        class optional {
          public:
            using value_type = T;
            optional(nullopt_t) : optional() {}
            optional() = default;
            AKR_XPU
            optional(const T &v) : set(true) { new (ptr()) T(v); }
            AKR_XPU
            optional(T &&v) : set(true) { new (ptr()) T(std::move(v)); }
            AKR_XPU
            optional(const optional &v) : set(v.has_value()) {
                if (v.has_value())
                    new (ptr()) T(v.value());
            }
            AKR_XPU
            optional(optional &&v) : set(v.has_value()) {
                if (v.has_value()) {
                    new (ptr()) T(std::move(v.value()));
                    v.reset();
                }
            }

            AKR_XPU
            optional &operator=(const T &v) {
                reset();
                new (ptr()) T(v);
                set = true;
                return *this;
            }
            AKR_XPU
            optional &operator=(T &&v) {
                reset();
                new (ptr()) T(std::move(v));
                set = true;
                return *this;
            }
            AKR_XPU
            optional &operator=(const optional &v) {
                reset();
                if (v.has_value()) {
                    new (ptr()) T(v.value());
                    set = true;
                }
                return *this;
            }
            template <typename... Ts>
            AKR_XPU void emplace(Ts &&... args) {
                reset();
                new (ptr()) T(std::forward<Ts>(args)...);
                set = true;
            }
            AKR_XPU
            optional &operator=(optional &&v) {
                reset();
                if (v.has_value()) {
                    new (ptr()) T(std::move(v.value()));
                    set = true;
                    v.reset();
                }
                return *this;
            }

            AKR_XPU
            ~optional() { reset(); }

            AKR_XPU
            explicit operator bool() const { return set; }

            AKR_XPU
            T value_or(const T &alt) const { return set ? value() : alt; }

            AKR_XPU
            T *operator->() { return &value(); }
            AKR_XPU
            const T *operator->() const { return &value(); }
            AKR_XPU
            T &operator*() { return value(); }
            AKR_XPU
            const T &operator*() const { return value(); }
            AKR_XPU
            T &value() {
                AKR_CHECK(set);
                return *ptr();
            }
            AKR_XPU
            const T &value() const {
                AKR_CHECK(set);
                return *ptr();
            }

            AKR_XPU
            void reset() {
                if (set) {
                    value().~T();
                    set = false;
                }
            }

            AKR_XPU
            bool has_value() const { return set; }

          private:
#ifdef __NVCC__
            // Work-around NVCC bug
            AKR_XPU
            T *ptr() { return reinterpret_cast<T *>(&optionalValue); }
            AKR_XPU
            const T *ptr() const { return reinterpret_cast<const T *>(&optionalValue); }
#else
            AKR_XPU
            T *ptr() { return std::launder(reinterpret_cast<T *>(&optionalValue)); }
            AKR_XPU
            const T *ptr() const { return std::launder(reinterpret_cast<const T *>(&optionalValue)); }
#endif

            std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
            bool set = false;
        };

        template <int I, typename T1, typename T2>
        AKR_XPU const T1 &get(const pair<T1, T2> &p) {
            static_assert(I >= 0 && I < 2);
            if constexpr (I == 0) {
                return p.first;
            } else {
                return p.second;
            }
        }
        template <typename T, int N>
        class array {
            T _data[N] = {};

          public:
            AKR_XPU array() = default;
            AKR_XPU T *data() { return _data; }
            AKR_XPU const T *data() const { return _data; }
            AKR_XPU T &operator[](int i) { return _data[i]; }
            AKR_XPU const T &operator[](int i) const { return _data[i]; }
            AKR_XPU size_t size() const { return N; }
            AKR_XPU T *begin() const { return _data; }
            AKR_XPU T *end() const { return _data + size(); }
            AKR_XPU const T *cbegin() const { return _data; }
            AKR_XPU const T *cend() const { return _data + size(); }
        };
    } // namespace astd
} // namespace akari