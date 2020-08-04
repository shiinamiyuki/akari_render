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

#include <akari/common/fwd.h>
namespace akari {
    template <typename T> struct Buffer {
        struct MutAccessor {
            Buffer<T> *_buf;
            T &operator[](int i) { return _buf._data[i]; }
            size_t size() const { return _buf._data.size(); }
        };
        struct Accessor {
            const Buffer<T> &_buf;
            const T &operator[](int i) const { return _buf._data[i]; }
            size_t size() const { return _buf._data.size(); }
        };
        MutAccessor mut_accessor() { return MutAccessor{*this}; }
        MutAccessor accessor() const { return MutAccessor{*this}; }
        friend struct MutAccessor;
        friend struct Accessor;
        std::vector<T> &data() const { return _data; }

      private:
        std::vector<T> _data;
        mutable int16_t _borrow_cnt = 0;
        mutable bool _borrow_mut = false;
    };
} // namespace akari