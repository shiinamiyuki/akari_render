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

#ifndef AKARIRENDER_PROFILER_HPP
#define AKARIRENDER_PROFILER_HPP
#include <chrono>
#include <deque>
#include <string_view>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <akari/core/fwd.h>
#include <akari/core/platform.h>
#include <akari/core/lock.h>

namespace akari {
    class Timer {
        decltype(std::chrono::system_clock::now()) start = std::chrono::system_clock::now();

      public:
        [[nodiscard]] std::chrono::nanoseconds elapsed_nanoseconds() const {
            auto now = std::chrono::system_clock::now();
            std::chrono::nanoseconds elapsed = now - start;
            return elapsed;
        }
        [[nodiscard]] double elapsed_seconds() const {
            auto now = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed = now - start;
            return elapsed.count();
        }
    };

    class AKR_EXPORT Profiler {
        struct Stats {
            using InternalDuration = std::chrono::nanoseconds;
            std::atomic<size_t> nanos;
            template <typename Rep, typename Period>
            void add(const std::chrono::duration<Rep, Period> &duration) {
                auto cst = std::chrono::duration_cast<InternalDuration>(duration);
                (void)nanos.fetch_add(cst.count());
            }
        };
        std::deque<std::pair<std::string_view, Timer>> active_frames;
        std::unordered_map<std::string_view, Stats> stats;
        RwLock lock;

      public:
        struct Frame {
            Profiler &profiler;
            Frame(Profiler &profiler, const char *name) : profiler(profiler) { profiler.enter(name); }
            ~Frame() { profiler.exit(); }
        };

        void enter(const char *name) {
            {
                std::lock_guard<RwLock> _(lock);
                active_frames.emplace_back(std::make_pair(name, Timer()));
                (void)stats[name];
            }
        }
        void exit() {
            {
                std::lock_guard<RwLock> _(lock);
                auto &&[name, timer] = active_frames.back();
                stats[name].add(timer.elapsed_nanoseconds());
                active_frames.pop_back();
            }
        }
        Frame frame(const char *name) { return Frame{*this, name}; }
        void print_stats();
    };

    AKR_EXPORT Profiler *get_profiler();
} // namespace akari
#endif // AKARIRENDER_PROFILER_HPP
