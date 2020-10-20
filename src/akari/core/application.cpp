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

#include <akari/core/akari.h>
#include <akari/core/application.h>
#include <akari/core/resource.h>
#include <akari/core/parallel.h>
#include <akari/core/logger.h>
#define NOMINMAX
#if AKR_PLATFORM_WINDOWS
#    include <Windows.h>
#endif
namespace akari {
    Application::Application(int argc, const char **argv) {
        auto cg = core_globals();

#if AKR_PLATFORM_WINDOWS
        char self_proc[MAX_PATH] = {0};
        auto res = GetModuleFileNameA(nullptr, self_proc, sizeof(self_proc));
        if (res == 0) {
            fprintf(stderr, "error retreiving program path; code=%d\n", GetLastError());
        }
        cg->program_path = fs::absolute(fs::path(std::string(self_proc)));
#endif
    }
    Application::~Application() {
        thread::finalize();
        ResourceManager::finalize();
    }
} // namespace akari