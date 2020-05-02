// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#ifndef AKARIRENDER_STREAM_H
#define AKARIRENDER_STREAM_H

#include <akari/Core/Component.h>

namespace akari {
    class Stream;

    template <typename T> struct StreamWriter { static void Write(const T &value, Stream &); };
    class Stream {
      public:
        virtual size_t GetStreamType() const = 0;
        virtual size_t WriteBytes(const uint8_t *buffer, size_t count) = 0;
        virtual void WriteInt32(int32_t) = 0;
        virtual void WriteUInt32(uint32_t) = 0;
        virtual void WriteInt64(int64_t) = 0;
        virtual void WriteUInt64(uint64_t) = 0;
        virtual void WriteFloat(float) = 0;
        virtual void WriteString(const std::string &) = 0;
        virtual void WriteArrayBegin(size_t) = 0;
        virtual void WriteArrayEnd() = 0;
        template <typename T> void Write(const T &value) { StreamWriter<T>::Write(value, *this); }
        template <typename T> void WriteArray(const std::vector<T> &vec) {
            WriteArrayBegin(vec.size());
            for (const auto &i : vec) {
                Write(i);
            }
            WriteArrayEnd();
        }
    };
} // namespace akari
#endif // AKARIRENDER_STREAM_H
