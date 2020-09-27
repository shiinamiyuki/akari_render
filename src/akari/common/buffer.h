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
#include <vector>
#include <memory_resource>
#include <akari/common/fwd.h>
#include <akari/core/mode.h>
#include <akari/common/bufferview.h>
namespace akari {
    astd::pmr::memory_resource *get_managed_memory_resource();
    // template <typename T> using Buffer = std::vector<T, std::pmr::polymorphic_allocator<T>>;
    template <typename T = std::byte>
    struct DeviceAllocator : astd::pmr::polymorphic_allocator<T> {
        DeviceAllocator() : astd::pmr::polymorphic_allocator<T>(get_managed_memory_resource()) {}
    };
    template <typename T>
    struct BufferView;
    template <typename T>
    struct Buffer {
        using Allocator = DeviceAllocator<T>;
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
        // using reverse_iterator = std::reverse_iterator<iterator>;
        // using const_reverse_iterator = std::reverse_iterator<const iterator>;

        Buffer() {}
        Buffer(size_t s) { resize(s); }
        Buffer(const Buffer &other) {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                this->allocator.construct(_data + i, other[i]);
            _size = other.size();
        }
        Buffer &operator=(const Buffer &other) {
            if (this == &other)
                return *this;

            clear();
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                this->allocator.construct(_data + i, other[i]);
            _size = other.size();

            return *this;
        }
        Buffer(Buffer &&rhs) : allocator(std::move(rhs.allocator)) {
            _data = rhs._data;
            _capacity = rhs._capacity;
            _size = rhs._size;
            rhs._data = nullptr;
            rhs._capacity = rhs._size = 0u;
        }
        Buffer &operator=(Buffer &&rhs) {
            _data = rhs._data;
            _capacity = rhs._capacity;
            _size = rhs._size;
            rhs._data = nullptr;
            rhs._capacity = rhs._size = 0u;
            allocator = std::move(rhs.allocator);
            return *this;
        }
        void push_back(const T &v) {
            if (_size == _capacity) {
                reserve(_capacity == 0 ? 4 : 2 * _capacity);
            }
            allocator.construct(_data + _size, v);
            _size++;
        }
        template <typename... Ts>
        void emplace_back(Ts &&... args) {
            if (_size == _capacity) {
                reserve(_capacity == 0 ? 4 : 2 * _capacity);
            }
            allocator.construct(_data + _size, std::forward<Ts>(args)...);
            _size++;
        }
        void reserve(size_t s) {
            if (s > _capacity) {
                size_t new_cap = s;
                auto *new_data = allocator.allocate(new_cap);
                if (_data) {
                    for (size_t i = 0; i < _size; i++) {
                        allocator.construct(new_data + i, std::move(_data[i]));
                        allocator.destroy(_data + i);
                    }
                    allocator.deallocate(_data, _capacity);
                }
                _data = new_data;
                _capacity = new_cap;
            }
        }
        void resize(size_t s) {
            if (s < _size) {
                for (size_t i = s; i < _size; i++) {
                    allocator.destroy(_data + i);
                }
                if (s == 0) {
                    allocator.deallocate(_data, _capacity);
                    _data = 0;
                    _capacity = 0;
                }
            } else if (s > _size) {
                reserve(s);
                for (size_t i = _size; i < s; i++) {
                    allocator.construct(_data + i);
                }
            }
            _size = s;
        }
        void pop_back() {
            _size -= 1;
            allocator.destroy(_data + _size);
        }
        ~Buffer() {
            for (size_t i = 0; i < _size; i++) {
                allocator.destroy(_data + i);
            }
            allocator.deallocate(_data, _capacity);
        }
        AKR_XPU T &operator[](size_t i) { return _data[i]; }
        AKR_XPU const T &operator[](size_t i) const { return _data[i]; }
        AKR_XPU T &at(size_t i) { return _data[i]; }
        AKR_XPU const T &at(size_t i) const { return _data[i]; }
        AKR_XPU T *data() const { return _data; }
        AKR_XPU T *begin() { return _data; }
        AKR_XPU T *end() { return _data + _size; }
        AKR_XPU const T *cbegin() const { return _data; }
        AKR_XPU const T *cend() const { return _data + _size; }
        AKR_XPU size_t size() const { return _size; }
        AKR_XPU size_t capacity() const { return _capacity; }
        void clear() {
            for (auto it = begin(); it != end(); it++) {
                allocator.destroy(it);
            }
            _data = 0;
            _size = 0;
            _capacity = 0;
        }
        AKR_XPU const T &back() const { return *(end() - 1); }
        AKR_XPU T &back() { return *(end() - 1); }

      private:
        Allocator allocator;
        T *_data = nullptr;
        size_t _size = 0;
        size_t _capacity = 0;
    };
} // namespace akari