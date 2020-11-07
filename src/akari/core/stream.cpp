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
#include <cstring>
#include <akari/core/stream.h>

namespace akari {
    FStream::FStream(const fs::path &path, Stream::Mode mode) : mode_(mode) {
        if (mode == Stream::Mode::Read) {
            file = std::fstream(path, std::ios::binary | std::ios::in);
        } else if (mode == Stream::Mode::Write) {
            file = std::fstream(path, std::ios::binary | std::ios::out);
        } else {
            file = std::fstream(path, std::ios::binary);
        }
    }
    size_t FStream::read(char *buf, size_t size) {
        AKR_ASSERT(mode_ == Stream::Mode::Read || mode_ == Stream::Mode::ReadWrite);
        file->read(buf, size);
        return file->gcount();
    }
    size_t FStream::write(const char *buf, size_t size) {
        AKR_ASSERT(mode_ == Stream::Mode::Write || mode_ == Stream::Mode::ReadWrite);
        file->write(buf, size);
        return file->gcount();
    }
    Stream::Mode FStream::mode() { return mode_; }
    bool FStream::closed() const { return file.has_value() && !file->eof(); }

    size_t ByteStream::read(char *buf, size_t size) {
        auto r = std::min(in.size(), pos + size) - pos;
        std::memcpy(buf, in.data() + pos, r);
        pos += r;
        return r;
    }
    size_t ByteStream::write(const char *buf, size_t size) {
        auto sz = out.size();
        out.resize(sz + size);
        std::memcpy(&out[sz], buf, size);
        return size;
    }
    Stream::Mode ByteStream::mode() { return mode_; }
    bool ByteStream::closed() const { return false; }
} // namespace akari