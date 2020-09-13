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
#pragma once
#include <type_traits>
#include <new>
#include <akari/common/def.h>
#include <akari/common/panic.h>
#include <akari/common/platform.h>
#include <math.h>

// port some nessary stl class to CUDA
// mostly from pbrt-v4
namespace akari::astd {
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
        array() = default;
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

    // MSVC has incomplete pmr support ...
    namespace pmr {
        class AKR_EXPORT memory_resource {
            static constexpr size_t max_align = alignof(std::max_align_t);

          public:
            virtual ~memory_resource() = default;
            void *allocate(size_t bytes, size_t alignment = max_align) { return do_allocate(bytes, alignment); }
            void deallocate(void *p, size_t bytes, size_t alignment = max_align) {
                return do_deallocate(p, bytes, alignment);
            }
            bool is_equal(const memory_resource &other) const noexcept { return do_is_equal(other); }

          private:
            virtual void *do_allocate(size_t bytes, size_t alignment) = 0;
            virtual void do_deallocate(void *p, size_t bytes, size_t alignment) = 0;
            virtual bool do_is_equal(const memory_resource &other) const noexcept = 0;
        };

        AKR_EXPORT memory_resource *new_delete_resource() noexcept;
        AKR_EXPORT memory_resource *set_default_resource(memory_resource *r) noexcept;
        AKR_EXPORT memory_resource *get_default_resource() noexcept;

        template <class Tp = std::byte>
        class polymorphic_allocator {
          public:
            using value_type = Tp;
            polymorphic_allocator() noexcept { memoryResource = new_delete_resource(); }
            polymorphic_allocator(memory_resource *r) : memoryResource(r) {}
            polymorphic_allocator(const polymorphic_allocator &other) = default;
            template <class U>
            polymorphic_allocator(const polymorphic_allocator<U> &other) noexcept : memoryResource(other.resource()) {}
            polymorphic_allocator &operator=(const polymorphic_allocator &rhs) = delete;
            [[nodiscard]] Tp *allocate(size_t n) {
                return static_cast<Tp *>(resource()->allocate(n * sizeof(Tp), alignof(Tp)));
            }
            void deallocate(Tp *p, size_t n) { resource()->deallocate(p, n); }

            void *allocate_bytes(size_t nbytes, size_t alignment = alignof(max_align_t)) {
                return resource()->allocate(nbytes, alignment);
            }
            void deallocate_bytes(void *p, size_t nbytes, size_t alignment = alignof(std::max_align_t)) {
                return resource()->deallocate(p, nbytes, alignment);
            }
            template <class T>
            T *allocate_object(size_t n = 1) {
                return static_cast<T *>(allocate_bytes(n * sizeof(T), alignof(T)));
            }
            template <class T>
            void deallocate_object(T *p, size_t n = 1) {
                deallocate_bytes(p, n * sizeof(T), alignof(T));
            }
            template <class T, class... Args>
            void construct(T *p, Args &&... args) {
                ::new ((void *)p) T(std::forward<Args>(args)...);
            }

            template <class T>
            void destroy(T *p) {
                p->~T();
            }
            memory_resource *resource() const { return memoryResource; }

          private:
            memory_resource *memoryResource;
        };
    } // namespace pmr
    AKR_XPU inline void *align(size_t _Bound, size_t _Size, void *&_Ptr, size_t &_Space) noexcept /* strengthened */ {
        // try to carve out _Size bytes on boundary _Bound
        size_t _Off = static_cast<size_t>(reinterpret_cast<uintptr_t>(_Ptr) & (_Bound - 1));
        if (_Off != 0) {
            _Off = _Bound - _Off; // number of bytes to skip
        }

        if (_Space < _Off || _Space - _Off < _Size) {
            return nullptr;
        }

        // enough room, update
        _Ptr = static_cast<char *>(_Ptr) + _Off;
        _Space -= _Off;
        return _Ptr;
    }
    [[noreturn]] AKR_XPU inline void abort() {
#ifdef AKR_GPU_CODE
        asm("trap;");
#else
        std::abort();
#endif
    }

} // namespace akari::astd