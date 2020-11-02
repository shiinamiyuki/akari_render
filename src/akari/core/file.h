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
#include <fstream>
#include <akari/core/plugin.h>
#include <akari/core/astd.h>

namespace akari {
    class AKR_EXPORT FileStream {
      public:
        virtual size_t read(char *buf, size_t bytes) = 0;
        virtual std::vector<char> read_all();
    };
    class AKR_EXPORT StdFileStream : public FileStream {
        std::ifstream in;

      public:
        StdFileStream(std::ifstream &&in) : in(std::move(in)) {}
        size_t read(char *buf, size_t bytes) override {
            in.read(buf, bytes);
            return in.gcount();
        }
    };
    class AKR_EXPORT ByteFileStream : public FileStream {
        std::vector<char> content;
        size_t pos = 0;

      public:
        ByteFileStream(std::vector<char> content):content(std::move(content)){}
        size_t read(char *buf, size_t bytes) override {
            if (pos + bytes > content.size()) {
                auto read = content.size() - pos;
                std::memcpy(buf, content.data() + pos, read);
                return read;
            } else {
                std::memcpy(buf, content.data() + pos, bytes);
                return bytes;
            }
        }
    };
    class AKR_EXPORT FileResolver {
      public:
        virtual std::unique_ptr<FileStream> resolve(const fs::path &path) = 0;
    };
    class AKR_EXPORT TrivialFileResolver : public FileResolver {
      public:
        std::unique_ptr<FileStream> resolve(const fs::path &path) override;
    };

    class AKR_EXPORT LocalFileResolver {
        std::vector<fs::path> search_paths;

      public:
        void add_search_path(const fs::path &path) { search_paths.push_back(path); }
    };
    class AKR_EXPORT MultiFileResolver : public FileResolver {
        std::list<std::shared_ptr<FileResolver>> resolvers;

      public:
        void add_file_resolver(std::shared_ptr<FileResolver> resolver) { resolvers.emplace_front(std::move(resolver)); }
        std::unique_ptr<FileStream> resolve(const fs::path &path) override;
    };
    AKR_EXPORT std::shared_ptr<MultiFileResolver> file_resolver();
    AKR_EXPORT std::unique_ptr<FileStream> resolve_file(const fs::path &path);
    AKR_EXPORT std::string read_file_to_str(const fs::path &path);
} // namespace akari