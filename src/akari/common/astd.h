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
#include <iterator>
#include <akari/common/def.h>
#include <akari/common/panic.h>
#include <akari/common/platform.h>
#include <math.h>

// port some nessary stl class to CUDA
// mostly from pbrt-v4
namespace akari::astd {
    enum class byte : unsigned char {};
    template <class _Ty>
    AKR_XPU constexpr _Ty &&forward(std::remove_reference_t<_Ty> &&_Arg) noexcept { // forward an rvalue as an rvalue
        static_assert(!std::is_lvalue_reference_v<_Ty>, "bad forward call");
        return static_cast<_Ty &&>(_Arg);
    }
    inline constexpr size_t max_size_t_(size_t a, size_t b) { return a < b ? b : a; }
    template <typename T1, typename T2>
    struct alignas(max_size_t_(alignof(T1), alignof(T2))) pair {
        T1 first;
        T2 second;
    };
    template <typename T1, typename T2>
    pair(T1 &&, T2 &&) -> pair<T1, T2>;

    template <typename T1, typename T2>
    AKR_XPU pair<T1, T2> make_pair(T1 &&a, T2 &&b) {
        return pair<T1, T2>{a, b};
    }

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
        AKR_CPU void emplace(Ts &&... args) {
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

    template <int I, typename T1, typename T2>
    AKR_XPU const T1 &get(const pair<T1, T2> &p) {
        static_assert(I >= 0 && I < 2);
        if constexpr (I == 0) {
            return p.first;
        } else {
            return p.second;
        }
    }
    template <typename T>
    AKR_XPU inline void swap(T &a, T &b) {
        T tmp = std::move(a);
        a = std::move(b);
        b = std::move(tmp);
    }

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

        template <class Tp = astd::byte>
        class polymorphic_allocator {
          public:
            using value_type = Tp;
            polymorphic_allocator() noexcept { memoryResource = new_delete_resource(); }
            polymorphic_allocator(memory_resource *r) : memoryResource(r) {}
            polymorphic_allocator(const polymorphic_allocator &other) = default;
            template <class U>
            polymorphic_allocator(const polymorphic_allocator<U> &other) noexcept : memoryResource(other.resource()) {}
            polymorphic_allocator &operator=(const polymorphic_allocator &rhs) = delete;
            polymorphic_allocator &operator=(polymorphic_allocator &&rhs) {
                this->memoryResource = rhs.memoryResource;
                rhs.memoryResource = nullptr;
                return *this;
            }
            [[nodiscard]] Tp *allocate(size_t n) {
                return static_cast<Tp *>(resource()->allocate(n * sizeof(Tp), alignof(Tp)));
            }
            template <class T>
            bool operator==(const polymorphic_allocator<T> &rhs) const {
                return resource() == rhs.resource();
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
            template <class T, class... Args>
            T *new_object(Args &&... args) {
                auto *p = allocate_object<T>();
                construct(p, std::forward<Args>(args)...);
                return p;
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

    template <typename T, int N>
    class array {
      public:
        using value_type = T;
        using iterator = value_type *;
        using const_iterator = const value_type *;
        using size_t = std::size_t;

        array() = default;
        AKR_XPU
        array(std::initializer_list<T> v) {
            size_t i = 0;
            for (const T &val : v)
                values[i++] = val;
        }

        AKR_XPU
        void fill(const T &v) {
            for (int i = 0; i < N; ++i)
                values[i] = v;
        }

        AKR_XPU
        bool operator==(const array<T, N> &a) const {
            for (int i = 0; i < N; ++i)
                if (values[i] != a.values[i])
                    return false;
            return true;
        }
        AKR_XPU
        bool operator!=(const array<T, N> &a) const { return !(*this == a); }

        AKR_XPU
        iterator begin() { return values; }
        AKR_XPU
        iterator end() { return values + N; }
        AKR_XPU
        const_iterator begin() const { return values; }
        AKR_XPU
        const_iterator end() const { return values + N; }

        AKR_XPU
        size_t size() const { return N; }

        AKR_XPU
        T &operator[](size_t i) { return values[i]; }
        AKR_XPU
        const T &operator[](size_t i) const { return values[i]; }

        AKR_XPU
        T *data() { return values; }
        AKR_XPU
        const T *data() const { return values; }

      private:
        T values[N] = {};
    };

    template <typename T, class Allocator = pmr::polymorphic_allocator<T>>
    class vector {
      public:
        using value_type = T;
        using allocator_type = Allocator;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = value_type &;
        using const_reference = const value_type &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const iterator>;

        vector(const Allocator &alloc = {}) : alloc(alloc) {}
        vector(size_t count, const T &value, const Allocator &alloc = {}) : alloc(alloc) {
            reserve(count);
            for (size_t i = 0; i < count; ++i)
                this->alloc.template construct<T>(ptr + i, value);
            nStored = count;
        }
        vector(size_t count, const Allocator &alloc = {}) : vector(count, T{}, alloc) {}
        vector(const vector &other, const Allocator &alloc = {}) : alloc(alloc) {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                this->alloc.template construct<T>(ptr + i, other[i]);
            nStored = other.size();
        }
        template <class InputIt>
        vector(InputIt first, InputIt last, const Allocator &alloc = {}) : alloc(alloc) {
            reserve(last - first);
            size_t i = 0;
            for (InputIt iter = first; iter != last; ++iter, ++i)
                this->alloc.template construct<T>(ptr + i, *iter);
            nStored = nAlloc;
        }
        vector(vector &&other) : alloc(other.alloc) {
            nStored = other.nStored;
            nAlloc = other.nAlloc;
            ptr = other.ptr;

            other.nStored = other.nAlloc = 0;
            other.ptr = nullptr;
        }
        vector(vector &&other, const Allocator &alloc) {
            if (alloc == other.alloc) {
                ptr = other.ptr;
                nAlloc = other.nAlloc;
                nStored = other.nStored;

                other.ptr = nullptr;
                other.nAlloc = other.nStored = 0;
            } else {
                reserve(other.size());
                for (size_t i = 0; i < other.size(); ++i)
                    alloc.template construct<T>(ptr + i, std::move(other[i]));
                nStored = other.size();
            }
        }
        vector(std::initializer_list<T> init, const Allocator &alloc = {}) : vector(init.begin(), init.end(), alloc) {}

        vector &operator=(const vector &other) {
            if (this == &other)
                return *this;

            clear();
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(ptr + i, other[i]);
            nStored = other.size();

            return *this;
        }
        vector &operator=(vector &&other) {
            if (this == &other)
                return *this;

            if (alloc == other.alloc) {
                astd::swap(ptr, other.ptr);
                astd::swap(nAlloc, other.nAlloc);
                astd::swap(nStored, other.nStored);
            } else {
                clear();
                reserve(other.size());
                for (size_t i = 0; i < other.size(); ++i)
                    alloc.template construct<T>(ptr + i, std::move(other[i]));
                nStored = other.size();
            }

            return *this;
        }
        vector &operator=(std::initializer_list<T> &init) {
            reserve(init.size());
            clear();
            iterator iter = begin();
            for (const auto &value : init) {
                *iter = value;
                ++iter;
            }
            return *this;
        }

        void assign(size_type count, const T &value) {
            clear();
            reserve(count);
            for (size_t i = 0; i < count; ++i)
                push_back(value);
        }
        template <class InputIt>
        void assign(InputIt first, InputIt last) {
            AKR_ASSERT(false && "TODO");
        }
        void assign(std::initializer_list<T> &init) { assign(init.begin(), init.end()); }

        ~vector() {
            clear();
            alloc.deallocate_object(ptr, nAlloc);
        }

        AKR_XPU
        iterator begin() { return ptr; }
        AKR_XPU
        iterator end() { return ptr + nStored; }
        AKR_XPU
        const_iterator begin() const { return ptr; }
        AKR_XPU
        const_iterator end() const { return ptr + nStored; }
        AKR_XPU
        const_iterator cbegin() const { return ptr; }
        AKR_XPU
        const_iterator cend() const { return ptr + nStored; }

        AKR_XPU
        reverse_iterator rbegin() { return reverse_iterator(end()); }
        AKR_XPU
        reverse_iterator rend() { return reverse_iterator(begin()); }
        AKR_XPU
        const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
        AKR_XPU
        const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

        allocator_type get_allocator() const { return alloc; }
        AKR_XPU
        size_t size() const { return nStored; }
        AKR_XPU
        bool empty() const { return size() == 0; }
        AKR_XPU
        size_t max_size() const { return (size_t)-1; }
        AKR_XPU
        size_t capacity() const { return nAlloc; }
        void reserve(size_t n) {
            if (nAlloc >= n)
                return;

            T *ra = alloc.template allocate_object<T>(n);
            for (size_t i = 0; i < nStored; ++i) {
                alloc.template construct<T>(ra + i, std::move(begin()[i]));
                alloc.destroy(begin() + i);
            }

            alloc.deallocate_object(ptr, nAlloc);
            nAlloc = n;
            ptr = ra;
        }
        // TODO: shrink_to_fit

        AKR_XPU
        reference operator[](size_type index) {
            // AKR_CHECK(index < size());
            return ptr[index];
        }
        AKR_XPU
        const_reference operator[](size_type index) const {
            // AKR_CHECK(index < size());
            return ptr[index];
        }
        AKR_XPU
        reference front() { return ptr[0]; }
        AKR_XPU
        const_reference front() const { return ptr[0]; }
        AKR_XPU
        reference back() { return ptr[nStored - 1]; }
        AKR_XPU
        const_reference back() const { return ptr[nStored - 1]; }
        AKR_XPU
        pointer data() { return ptr; }
        AKR_XPU
        const_pointer data() const { return ptr; }

        void clear() {
            for (size_t i = 0; i < nStored; ++i)
                alloc.destroy(&ptr[i]);
            nStored = 0;
        }

        iterator insert(const_iterator, const T &value) { AKR_ASSERT(false && "TODO"); }
        iterator insert(const_iterator, T &&value) { AKR_ASSERT(false && "TODO"); }
        iterator insert(const_iterator pos, size_type count, const T &value) { AKR_ASSERT(false && "TODO"); }
        template <class InputIt>
        iterator insert(const_iterator pos, InputIt first, InputIt last) {
            if (pos == end()) {
                size_t firstOffset = size();
                for (auto iter = first; iter != last; ++iter)
                    push_back(*iter);
                return begin() + firstOffset;
            } else
                AKR_ASSERT(false && "TODO");
        }
        iterator insert(const_iterator pos, std::initializer_list<T> init) { AKR_ASSERT(false && "TODO"); }

        template <class... Args>
        iterator emplace(const_iterator pos, Args &&... args) {
            AKR_ASSERT(false && "TODO");
        }
        template <class... Args>
        void emplace_back(Args &&... args) {
            if (nAlloc == nStored)
                reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

            alloc.construct(ptr + nStored, std::forward<Args>(args)...);
            ++nStored;
        }

        iterator erase(const_iterator pos) { AKR_ASSERT(false && "TODO"); }
        iterator erase(const_iterator first, const_iterator last) { AKR_ASSERT(false && "TODO"); }

        void push_back(const T &value) {
            if (nAlloc == nStored)
                reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

            alloc.construct(ptr + nStored, value);
            ++nStored;
        }
        void push_back(T &&value) {
            if (nAlloc == nStored)
                reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

            alloc.construct(ptr + nStored, std::move(value));
            ++nStored;
        }
        void pop_back() {
            DCHECK(!empty());
            alloc.destroy(ptr + nStored - 1);
            --nStored;
        }

        void resize(size_type n) {
            if (n < size()) {
                for (size_t i = n; i < size(); ++i)
                    alloc.destroy(ptr + i);
                if (n == 0) {
                    alloc.deallocate_object(ptr, nAlloc);
                    ptr = nullptr;
                    nAlloc = 0;
                }
            } else if (n > size()) {
                reserve(n);
                for (size_t i = nStored; i < n; ++i)
                    alloc.construct(ptr + i);
            }
            nStored = n;
        }
        void resize(size_type count, const value_type &value) {
            // TODO
            AKR_ASSERT(false && "TODO");
        }

        void swap(vector &other) {
            // TODO
            AKR_ASSERT(false && "TODO");
        }

      private:
        Allocator alloc;
        T *ptr = nullptr;
        size_t nAlloc = 0, nStored = 0;
    };
    namespace pmr {
        template <typename T>
        using vector = astd::vector<T, polymorphic_allocator<T>>;
    }
} // namespace akari::astd