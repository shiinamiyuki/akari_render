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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sstream>
#include <thread>
#include <akari/core/util.h>
#if defined(AKR_PLATFORM_LINUX)
#    include <dlfcn.h>
#    include <unistd.h>
#    include <limits.h>
#    include <sys/ioctl.h>
#elif defined(AKR_PLATFORM_WINDOWS)
#    include <windows.h>
#endif
namespace akari {
    AKR_EXPORT int terminal_width() {
        static int cached_width = -1;

        if (cached_width == -1) {
#if defined(AKR_PLATFORM_WINDOWS)
            HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
            if (h != INVALID_HANDLE_VALUE && h != nullptr) {
                CONSOLE_SCREEN_BUFFER_INFO bufferInfo = {0};
                GetConsoleScreenBufferInfo(h, &bufferInfo);
                cached_width = bufferInfo.dwSize.X;
            }
#else
            struct winsize w;
            if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) >= 0)
                cached_width = w.ws_col;
#endif
            if (cached_width == -1)
                cached_width = 80;
        }
        return cached_width;
    }
    AKR_EXPORT std::string info_build() {

        std::ostringstream oss;
        oss << "AkariRender v" << AKR_VERSION_MAJOR << "." << AKR_VERSION_MINOR << "." << AKR_VERSION_PATCH << ", ";
#if defined(AKR_PLATFORM_WINDOWS)
        oss << "Windows, ";
#elif defined(AKR_PLATFORM_LINUX)
        oss << "Linux, ";
#elif defined(__OSX__)
        oss << "Mac OS, ";
#else
        oss << "Unknown, ";
#endif
        oss << (sizeof(size_t) * 8) << "bit, ";
        auto thread_count = std::thread::hardware_concurrency();
        oss << thread_count << " thread" << (thread_count > 1 ? "s" : "");

        return oss.str();
    }

    AKR_EXPORT fs::path extend_filename(const fs::path &path, const std::string &s) {
        auto parent_path = path.parent_path();
        auto filename = path.stem();
        auto ext = path.extension().string();
        return (parent_path / (filename.concat(s))).concat(ext);
    }
} // namespace akari