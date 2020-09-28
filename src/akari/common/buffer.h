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
#include <akari/core/device.h>
#include <akari/common/bufferview.h>
namespace akari {
    astd::pmr::memory_resource *get_managed_memory_resource();
    // template <typename T> using Buffer = std::vector<T, std::pmr::polymorphic_allocator<T>>;
    template <typename T>
    struct BufferView;

    template <typename T>
    struct Buffer {
        // as we need to memcpy directly across device
        static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
        using Allocator = astd::pmr::polymorphic_allocator<T>;
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

        Buffer(MemoryResource *resource) : resource(resource), allocator(Allocator(resource)) {}
        Buffer(const Buffer &) = delete;
        // copy host data and resize
        template <typename A>
        void copy(const std::vector<T, A> &v) {
            copy(v.data(), v.size());
        }
        void copy(const astd::pmr::vector<T> &v) { copy(v.data(), v.size()); }
        void copy(const T *host_ptr, size_t s) {
            resize(s);
            if (auto device_res = dynamic_cast<DeviceMemoryResource *>(resource)) {
                device_res->device()->copy_host_to_device(data(), host_ptr, sizeof(T) * size());
            } else if (auto managed_res = dynamic_cast<ManagedMemoryResource *>(resource)) {
                managed_res->device()->copy_host_to_device(data(), host_ptr, sizeof(T) * size());
            } else {
                std::memcpy(data(), host_ptr, sizeof(T) * size());
            }
        }
        Buffer(Buffer &&rhs) : allocator(std::move(rhs.allocator)) {
            _data = rhs._data;
            _size = rhs._size;
            rhs._data = nullptr;
            rhs._size = 0;
        }
        Buffer &operator=(Buffer &&rhs) {
            _data = rhs._data;
            _size = rhs._size;
            rhs._data = nullptr;
            rhs._size = 0;
            allocator = std::move(rhs.allocator);
            return *this;
        }
        void resize(size_t s) {
            if (_data)
                allocator.deallocate(_data, _size);
            _size = s;
            _data = allocator.allocate(_size);
        }
        ~Buffer() {
            if (_data)
                allocator.deallocate(_data, _size);
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
        void clear() {
            allocator.deallocate(_data, _size);
            _data = nullptr;
            _size = 0;
        }
        AKR_XPU const T &back() const { return *(end() - 1); }
        AKR_XPU T &back() { return *(end() - 1); }
        BufferView<T> view() const { return {data(), size()}; }
        void set_memory_resource(MemoryResource *resource) {
            AKR_ASSERT(_data == nullptr);
            this->resource = resource;
            allocator = Allocator(resource);
        }

      private:
        MemoryResource *resource;
        Allocator allocator;
        T *_data = nullptr;
        size_t _size = 0;
    };
} // namespace akari