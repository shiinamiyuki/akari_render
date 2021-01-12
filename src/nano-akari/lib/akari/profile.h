// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <akari/util.h>
#include <chrono>
#include <mutex>
namespace akari {
    class Timer {
        using TP = decltype(std::chrono::high_resolution_clock::now());
        TP t0, t1;

      public:
        void start() { t0 = std::chrono::high_resolution_clock::now(); }
        void stop() { t1 = std::chrono::high_resolution_clock::now(); }
        double elapsed_seconds() const {
            std::chrono::duration<double> diff = t1 - t0;
            return diff.count();
        }
        double seconds_since_start() const {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = now - t0;
            return diff.count();
        }
    };
    inline void show_progress(double progress, double elpased, double remaining) {
        printf("[");
        size_t width = 80; // terminal_width() - 30;
        size_t pos = size_t(width * progress);
        for (size_t i = 0; i < width; ++i) {
            if (i < pos)
                printf("=");
            else if (i == pos)
                printf(">");
            else
                printf(" ");
        }
        printf("] %2d %% (%.3fs|%.3fs)\r", int(progress * 100.0), elpased, remaining);
        fflush(stdout);
    }
    inline void show_progress(double progress, size_t width) {
        printf("[");
        size_t pos = size_t(width * progress);
        for (size_t i = 0; i < width; ++i) {
            if (i < pos)
                printf("=");
            else if (i == pos)
                printf(">");
            else
                printf(" ");
        }
        printf("] %2d %%\r", int(progress * 100.0));
        fflush(stdout);
    }
    struct ProgressReporter {
        explicit ProgressReporter(size_t total, double min_report_interval = 0.5)
            : total(total), min_report_interval(min_report_interval) {
            timer.start();
        }
        void update() {
            auto cur = count.fetch_add(1);
            std::unique_lock<std::mutex> lock(m, std::try_to_lock);
            if (!lock.owns_lock())
                return;
            if (timer.seconds_since_start() - last_report_time >= min_report_interval) {
                last_report_time = timer.seconds_since_start();
                auto remaining = last_report_time * double(total) / cur - last_report_time;
                show_progress(double(cur) / total, last_report_time, remaining);
            }
        }

      private:
        double min_report_interval;
        double last_report_time = 0;
        Timer timer;
        std::mutex m;
        std::atomic<size_t> count = 0;
        std::atomic<size_t> total;
    };
} // namespace akari