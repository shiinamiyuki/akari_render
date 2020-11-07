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
#include <optional>
#include <istream>
#include <ostream>
#include <fstream>
#include <akari/core/akari.h>
#include <akari/core/memory.h>
#include <unordered_map>

namespace akari {
    // bidrectional byte stream
    class AKR_EXPORT Stream {
      public:
        enum class Mode { Read = 0, Write = 1, ReadWrite = 2 };
        virtual size_t read(char *buf, size_t size) = 0;
        virtual size_t write(const char *buf, size_t size) = 0;
        virtual Mode mode() = 0;
        virtual bool closed() const = 0;
        virtual ~Stream() = default;
    };
    class AKR_EXPORT FStream : public Stream {
        Stream::Mode mode_;
        std::optional<std::fstream> file;

      public:
        FStream(const fs::path &path, Stream::Mode mode);
        size_t read(char *buf, size_t size) override;
        size_t write(const char *buf, size_t size) override;
        Mode mode() override;
        bool closed() const override;
    };
    class AKR_EXPORT ByteStream : public Stream {
        Stream::Mode mode_;
        std::optional<std::fstream> file;
        std::vector<char> out;
        std::string_view in;
        size_t pos = 0;

      public:
        ByteStream(Stream::Mode mode) : mode_(mode) {}
        ByteStream(std::string_view in, Stream::Mode mode) : mode_(mode), in(in) {}
        const std::vector<char> &buf_out() { return out; }
        size_t read(char *buf, size_t size) override;
        size_t write(const char *buf, size_t size) override;
        Mode mode() override;
        bool closed() const override;
    };
} // namespace akari