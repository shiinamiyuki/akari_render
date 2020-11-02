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

#include <akari/core/file.h>
#include <mutex>
#include <sstream>
namespace akari {
    AKR_EXPORT std::string read_file_to_str(const fs::path &path) {
        auto stream = resolve_file(path);
        auto buf = stream->read_all();
        std::ostringstream os;
        auto it = buf.begin();
        while (it < buf.end()) {
            if (*it == '\r' && (it + 1) < buf.end() && *(it + 1) == '\n') {
                it += 2;
                os << '\n';
            } else {
                os << *it;
                it++;
            }
        }
        return os.str();
    }
    std::vector<char> FileStream::read_all() {
        std::vector<char> buf(4096);
        std::vector<char> all;
        while (true) {
            size_t bytes_read = read(buf.data(), buf.size());
            size_t prev_size = all.size();
            if (bytes_read > 0) {
                all.resize(all.size() + bytes_read);
                std::memcpy(all.data() + prev_size, buf.data(), bytes_read);
            }
            if (bytes_read != buf.size()) {
                break;
            }
        }
        return all;
    }
    std::unique_ptr<FileStream> TrivialFileResolver::resolve(const fs::path &path) {
        if (!fs::exists(path)) {
            return nullptr;
        }
        return std::make_unique<StdFileStream>(std::ifstream(path, std::ios::in | std::ios::binary));
    }
    std::unique_ptr<FileStream> MultiFileResolver::resolve(const fs::path &path) {
        for (auto &resolver : resolvers) {
            auto s = resolver->resolve(path);
            if (s)
                return s;
        }
        return nullptr;
    }
    AKR_EXPORT std::shared_ptr<MultiFileResolver> file_resolver() {
        static std::shared_ptr<MultiFileResolver> resolver;
        static std::once_flag flag;
        std::call_once(flag, [&] {
            resolver.reset(new MultiFileResolver());
            resolver->add_file_resolver(std::make_shared<TrivialFileResolver>());
        });
        return resolver;
    }
    AKR_EXPORT std::unique_ptr<FileStream> resolve_file(const fs::path &path) { return file_resolver()->resolve(path); }
} // namespace akari