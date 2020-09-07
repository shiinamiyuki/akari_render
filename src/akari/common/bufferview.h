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
#include <akari/common/fwd.h>

namespace akari {
    template <typename T>
    struct BufferView {
        BufferView() = default;
        BufferView(T *data, size_t size) : _data(data), _size(size) {}
        template <typename Allocator>
        BufferView(std::vector<T, Allocator> &vec) : BufferView(vec.data(), vec.size()) {}
        T &operator[](uint32_t i) const { return _data[i]; }
        size_t size() const { return _size; }
        T *begin() const { return _data; }
        T *end() const { return _data + _size; }
        const T *cbegin() const { return _data; }
        const T *cend() const { return _data + _size; }

      private:
        T *_data = nullptr;
        size_t _size = 0;
    };
} // namespace akari