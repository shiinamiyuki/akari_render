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
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <cstring>
#include <akari/common.h>
#include <akari/pmr.h>
#if defined(AKR_GPU_BACKEND_CUDA) || defined(__CUDACC__) || defined(__NVCC__)
#    include <cuda.h>
#    define GLM_FORCE_CUDA
// #else
// #    define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// #    define GLM_FORCE_PURE
// #    define GLM_FORCE_INTRINSICS
// #    define GLM_FORCE_SSE2
// #    define GLM_FORCE_SSE3
// #    define GLM_FORCE_AVX
// #    define GLM_FORCE_AVX2
#endif
#include <akari/mathutil.h>
#include <cereal/cereal.hpp>
#include <akari/common.h>

namespace akari {
    class NonCopyable {
      public:
        NonCopyable()                    = default;
        NonCopyable(const NonCopyable &) = delete;
        NonCopyable &operator=(const NonCopyable &) = delete;
        NonCopyable(NonCopyable &&)                 = default;
    };
    template <class T = astd::byte>
    using Allocator = astd::pmr::polymorphic_allocator<T>;
    template <typename T, typename... Ts>
    std::shared_ptr<T> make_pmr_shared(Allocator<> alloc, Ts &&...args) {
        return std::shared_ptr<T>(alloc.new_object<T>(std::forward<Ts>(args)...), [=](T *p) mutable {
            alloc.destroy(p);
            alloc.deallocate_object(const_cast<std::remove_const_t<T> *>(p), 1);
        });
    }
    template <class Archive, class T>
    void safe_apply(Archive &ar, const char *name, T &val) {
        try {
            ar(CEREAL_NVP_(name, val));
        } catch (cereal::Exception &e) {
            std::string_view msg(e.what());
            if (msg.find("provided NVP") == std::string_view::npos) {
                // not a name not found error
                throw e;
            }
        }
    }
} // namespace akari
namespace cereal {
    template <typename Archive, int N, typename T, glm::qualifier Q>
    void serialize(Archive &ar, glm::vec<N, T, Q> &v) {
        for (int i = 0; i < N; i++) {
            ar(v[i]);
        }
    }

    template <typename Archive, int C, int R, typename T, glm::qualifier Q>
    void serialize(Archive &ar, glm::mat<C, R, T, Q> &v) {
        for (int i = 0; i < C; i++) {
            ar(v[i]);
        }
    }
} // namespace cereal

namespace akari {
    struct TRSTransform {
        Vec3 translation;
        Vec3 rotation;
        Vec3 scale     = Vec3(1.0, 1.0, 1.0);
        TRSTransform() = default;
        AKR_XPU TRSTransform(Vec3 t, Vec3 r, Vec3 s) : translation(t), rotation(r), scale(s) {}
        AKR_XPU Transform operator()() const {
            Transform T;
            T = Transform::scale(scale);
            T = Transform::rotate_z(rotation.z) * T;
            T = Transform::rotate_x(rotation.y) * T;
            T = Transform::rotate_y(rotation.x) * T;
            T = Transform::translate(translation) * T;
            return T;
        }
        AKR_SER(translation, rotation, scale)
    };
} // namespace akari
namespace akari {
    using namespace glm;
    namespace astd {
        template <class To, class From>
        typename std::enable_if_t<
            sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>, To>
        // constexpr support needs compiler magic
        bit_cast(const From &src) noexcept {
            static_assert(std::is_trivially_constructible_v<To>,
                          "This implementation additionally requires destination type to be trivially constructible");

            To dst;
            std::memcpy(&dst, &src, sizeof(To));
            return dst;
        }

        inline constexpr size_t max(size_t a, size_t b) { return a < b ? b : a; }
        template <typename T1, typename T2>
        struct alignas(astd::max(alignof(T1), alignof(T2))) pair {
            T1 first;
            T2 second;
        };
        // template <typename T1, typename T2>
        // pair(T1 &&, T2 &&) -> pair<T1, T2>;

        // template <typename T1, typename T2>
        // AKR_XPU pair<T1, T2> make_pair(T1 a, T2 b) {
        //     return pair<T1, T2>{a, b};
        // }

        struct nullopt_t {};
        inline constexpr nullopt_t nullopt{};
        template <typename T>
        class optional {
          public:
            using value_type = T;
            AKR_XPU optional(nullopt_t) : optional() {}
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
            AKR_CPU void emplace(Ts &&...args) {
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
            // #ifdef __NVCC__
            // Work-around NVCC bug
            AKR_XPU
            T *ptr() { return reinterpret_cast<T *>(&optionalValue); }
            AKR_XPU
            const T *ptr() const { return reinterpret_cast<const T *>(&optionalValue); }
            // #else
            //         AKR_XPU
            //         T *ptr() { return std::launder(reinterpret_cast<T *>(&optionalValue)); }
            //         AKR_XPU
            //         const T *ptr() const { return std::launder(reinterpret_cast<const T *>(&optionalValue)); }
            // #endif

            std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
            bool set = false;
        };

        // template <int I, typename T1, typename T2>
        // AKR_XPU const T1 &get(const pair<T1, T2> &p) {
        //     static_assert(I >= 0 && I < 2);
        //     if constexpr (I == 0) {
        //         return p.first;
        //     } else {
        //         return p.second;
        //     }
        // }
        template <typename T>
        AKR_XPU inline void swap(T &a, T &b) {
            T tmp = std::move(a);
            a     = std::move(b);
            b     = std::move(tmp);
        }
    } // namespace astd

    template <typename... T>
    struct TypeIndex {
        template <typename U, typename Tp, typename... Rest>
        struct GetIndex_ {
            static const int value =
                std::is_same<Tp, U>::value
                    ? 0
                    : ((GetIndex_<U, Rest...>::value == -1) ? -1 : 1 + GetIndex_<U, Rest...>::value);
        };
        template <typename U, typename Tp>
        struct GetIndex_<U, Tp> {
            static const int value = std::is_same<Tp, U>::value ? 0 : -1;
        };
        template <int I, typename Tp, typename... Rest>
        struct GetType_ {
            using type = typename std::conditional<I == 0, Tp, typename GetType_<I - 1, Rest...>::type>::type;
        };

        template <int I, typename Tp>
        struct GetType_<I, Tp> {
            using type = typename std::conditional<I == 0, Tp, void>::type;
        };

        template <typename U>
        struct GetIndex {
            static const int value = GetIndex_<U, T...>::value;
        };

        template <int N>
        struct GetType {
            using type = typename GetType_<N, T...>::type;
        };
    };
    template <class T, class... Rest>
    struct FirstOf {
        using type = T;
    };

    template <typename U, typename... T>
    struct SizeOf {
        static constexpr int value = std::max<int>(sizeof(U), SizeOf<T...>::value);
    };
    template <typename T>
    struct SizeOf<T> {
        static constexpr int value = sizeof(T);
    };

    template <typename... T>
    struct Variant {
      private:
        static constexpr int nTypes                  = sizeof...(T);
        static constexpr std::size_t alignment_value = std::max({alignof(T)...});
        typename std::aligned_storage<SizeOf<T...>::value, alignment_value>::type data;
        int index = -1;

      public:
        using Index                       = TypeIndex<T...>;
        static constexpr size_t num_types = nTypes;

        template <typename U>
        AKR_XPU Variant(const U &u) {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            new (&data) U(u);
            index = Index::template GetIndex<U>::value;
        }

        AKR_XPU Variant(const Variant &v) : index(v.index) {
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                new (&data) U(item);
            });
        }
        AKR_XPU int typeindex() const { return index; }
        template <typename U>
        AKR_XPU constexpr static int indexof() {
            return Index::template GetIndex<U>::value;
        }
        AKR_XPU Variant &operator=(const Variant &v) noexcept {
            if (this == &v)
                return *this;
            if (index != -1)
                _drop();
            index = v.index;
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                new (&data) U(item);
            });
            return *this;
        }

        AKR_XPU Variant(Variant &&v) noexcept : index(v.index) {
            index   = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
        }

        AKR_XPU Variant &operator=(Variant &&v) noexcept {
            if (index != -1)
                _drop();
            index   = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
            return *this;
        }

        template <typename U>
        AKR_XPU Variant &operator=(const U &u) {
            if (index != -1) {
                _drop();
            }
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            new (&data) U(u);
            index = Index::template GetIndex<U>::value;
            return *this;
        }
        AKR_XPU bool null() const { return index == -1; }
        template <typename U>
        AKR_XPU bool isa() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value == index;
        }
        template <typename U>
        AKR_XPU U *get() {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<U *>(&data);
        }

        template <typename U>
        AKR_XPU const U *get() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<const U *>(&data);
        }

#define _GEN_CASE_N(N)                                                                                                 \
    case N:                                                                                                            \
        if constexpr (N < nTypes) {                                                                                    \
            using ty = typename Index::template GetType<N>::type;                                                      \
            if constexpr (!std::is_same_v<ty, std::monostate>) {                                                       \
                if constexpr (std::is_const_v<std::remove_pointer_t<decltype(this)>>)                                  \
                    return visitor(*reinterpret_cast<const ty *>(&data));                                              \
                else                                                                                                   \
                    return visitor(*reinterpret_cast<ty *>(&data));                                                    \
            }                                                                                                          \
        };                                                                                                             \
        break;
#define _GEN_CASES_2()                                                                                                 \
    _GEN_CASE_N(0)                                                                                                     \
    _GEN_CASE_N(1)
#define _GEN_CASES_4()                                                                                                 \
    _GEN_CASES_2()                                                                                                     \
    _GEN_CASE_N(2)                                                                                                     \
    _GEN_CASE_N(3)
#define _GEN_CASES_8()                                                                                                 \
    _GEN_CASES_4()                                                                                                     \
    _GEN_CASE_N(4)                                                                                                     \
    _GEN_CASE_N(5)                                                                                                     \
    _GEN_CASE_N(6)                                                                                                     \
    _GEN_CASE_N(7)
#define _GEN_CASES_16()                                                                                                \
    _GEN_CASES_8()                                                                                                     \
    _GEN_CASE_N(8)                                                                                                     \
    _GEN_CASE_N(9)                                                                                                     \
    _GEN_CASE_N(10)                                                                                                    \
    _GEN_CASE_N(11)                                                                                                    \
    _GEN_CASE_N(12)                                                                                                    \
    _GEN_CASE_N(13)                                                                                                    \
    _GEN_CASE_N(14)                                                                                                    \
    _GEN_CASE_N(15)
#define _GEN_DISPATCH_BODY()                                                                                           \
    using Ret = std::invoke_result_t<Visitor, typename FirstOf<T...>::type &>;                                         \
    static_assert(nTypes <= 16, "too many types");                                                                     \
    if constexpr (nTypes <= 2) {                                                                                       \
        switch (index) { _GEN_CASES_2(); }                                                                             \
    } else if constexpr (nTypes <= 4) {                                                                                \
        switch (index) { _GEN_CASES_4(); }                                                                             \
    } else if constexpr (nTypes <= 8) {                                                                                \
        switch (index) { _GEN_CASES_8(); }                                                                             \
    } else if constexpr (nTypes <= 16) {                                                                               \
        switch (index) { _GEN_CASES_16(); }                                                                            \
    }                                                                                                                  \
    if constexpr (std::is_same_v<void, Ret>) {                                                                         \
        return;                                                                                                        \
    } else {                                                                                                           \
        AKR_PANIC("No matching case");                                                                                 \
    }
        template <class Visitor>
        AKR_XPU auto dispatch(Visitor &&visitor) {
            _GEN_DISPATCH_BODY()
        }

        template <class Visitor>
        AKR_XPU auto dispatch(Visitor &&visitor) const {_GEN_DISPATCH_BODY()}

        AKR_XPU ~Variant() {
            if (index != -1)
                _drop();
        }

      private:
        AKR_XPU void _drop() {
            auto *that = this; // prevent gcc ICE
            dispatch([=](auto &&self) {
                using U = std::decay_t<decltype(self)>;
                that->template get<U>()->~U();
            });
        }
#undef _GEN_CASE_N
#define AKR_VAR_DISPATCH(method, ...)                                                                                  \
    return this->dispatch([&, this](auto &&self) {                                                                     \
        (void)this;                                                                                                    \
        return self.method(__VA_ARGS__);                                                                               \
    });
#define AKR_VAR_PTR_DISPATCH(method, ...)                                                                              \
    return this->dispatch([&, this](auto &&self) {                                                                     \
        (void)this;                                                                                                    \
        return self->method(__VA_ARGS__);                                                                              \
    });
    };
    template <class... Ts>
    struct overloaded : Ts... {
        using Ts::operator()...;
    };
    // explicit deduction guide (not needed as of C++20)
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    template <typename T>
    struct BufferView {
        BufferView() = default;
        // template <typename Allocator>
        // BufferView(std::vector<T, Allocator> &vec) : _data(vec.data()), _size(vec.size()) {}
        // template <typename Allocator>
        // BufferView(std::enable_if_t<std::is_const_v<T>, const std::vector<std::remove_const_t<T>, Allocator>> &vec)
        //     : _data(vec.data()), _size(vec.size()) {}
        BufferView(T *data, size_t size) : _data(data), _size(size) {}
        T &operator[](uint32_t i) const { return _data[i]; }
        size_t size() const { return _size; }
        T *begin() const { return _data; }
        T *end() const { return _data + _size; }
        const T *cbegin() const { return _data; }
        const T *cend() const { return _data + _size; }
        T *const &data() const { return _data; }
        bool empty() const { return size() == 0; }

      private:
        T *_data     = nullptr;
        size_t _size = 0;
    };
    template <typename T>
    BufferView(T *, size_t) -> BufferView<T>;

    template <typename T>
    struct ObjectCache {
        template <class F>
        T &get_cached_or(F &&f) {
            if (!_storage) {
                _storage = f();
            }
            return _storage.value();
        }
        void invalidate() { _storage.reset(); }

      private:
        std::optional<T> _storage;
    };
    namespace astd {
        template <class Fn>
        struct scope_exit {
            explicit scope_exit(Fn &&fn) noexcept : fn(fn) {}
            ~scope_exit() { fn(); }
            scope_exit(const scope_exit &) = delete;

          private:
            Fn fn;
        };
        template <class Fn>
        scope_exit(Fn &&) -> scope_exit<Fn>;
    } // namespace astd

    template <typename T, int N = (64 + sizeof(T) - 1) / sizeof(T)>
    class FixedVector {
        T *_data = nullptr;
        std::array<T, N> arr;
        // bool on_heap = false;
        uint32_t _size = 0;
        void ensure_size(size_t x) { AKR_ASSERT(x <= N); }

      public:
        FixedVector() { _data = &arr[0]; }
        size_t size() const { return _size; }
        const T *data() const { return _data; }
        T *data() { return _data; }
        const T &operator[](uint32_t i) const { return data()[i]; }
        T &operator[](uint32_t i) { return data()[i]; }
        void push_back(const T &v) {
            ensure_size(size() + 1);
            new (data() + size()) T(v);
            _size++;
        }
        void pop_back() {
            AKR_ASSERT(size() > 0);
            _size--;
            data()[_size].~T();
        }
        bool empty() const { return size() == 0; }
        T &back() { return data()[size() - 1]; }
        const T &back() const { return data()[size() - 1]; }
    };
} // namespace akari
