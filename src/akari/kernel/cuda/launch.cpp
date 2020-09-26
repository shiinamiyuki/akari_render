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

#include <map>
#include "launch.h"
#include <akari/core/logger.h>
namespace akari::gpu {
    namespace cuda_profiler {
        struct KernelStats {
            size_t num_launches = 0;
            std::string name;
            double total_ms = 0;
            double max_ms = 0;
            double min_ms = 0;
            bool active = false;
            KernelStats() {
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));
            }
            void sync() {
                AKR_CHECK(active);
                CUDA_CHECK(cudaEventSynchronize(start));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                total_ms += ms;
                num_launches++;
                if (num_launches == 1) {
                    min_ms = max_ms = ms;
                } else {
                    min_ms = std::min(min_ms, (double)ms);
                    max_ms = std::max(max_ms, (double)ms);
                }
                active = false;
            }
            cudaEvent_t start, stop;
        };
        static std::unordered_map<std::string_view, KernelStats> stats;
    } // namespace cuda_profiler
    std::pair<cudaEvent_t, cudaEvent_t> get_profiler_events(const char *description) {
        using namespace cuda_profiler;
        auto it = stats.find(description);
        if (it == stats.end()) {
            auto [it_, _] = stats.emplace(description, KernelStats());
            it = it_;
            it->second.name = description;
        }
        auto &stat = it->second;
        if (stat.active) {
            stat.sync();
        }
        stat.active = true;
        return std::make_pair(stat.start, stat.stop);
    }
    void print_kernel_stats() {
        using namespace cuda_profiler;
        std::vector<KernelStats *> vstats;
        for (auto &[name, stat] : stats) {
            if (stat.active) {
                stat.sync();
            }
            vstats.push_back(&stat);
        }
        double total_ms = 0;
        int total_launches = 0;
        for (auto &[name, stat] : stats) {
            total_ms += stat.total_ms;
            total_launches += stat.num_launches;
        }
        std::sort(vstats.begin(), vstats.end(), [](auto *s1, auto *s2) { return s1->total_ms > s2->total_ms; });
        info("GPU Kernel Profile:");
        fmt::print("Total GPU Time: {:.3f}ms\n", total_ms);
        fmt::print("Total Launches: {}\n", total_launches);
        for (auto stat : vstats) {
            fmt::print("  {:>30} {:6} launches {:9.2f}ms {:5.1f}% (avg: {:6.3f}ms min: {:6.3f}ms max: {:6.3f}ms)\n",
                       stat->name, stat->num_launches, stat->total_ms, 100.0 * stat->total_ms / total_ms,
                       stat->total_ms / stat->num_launches, stat->min_ms, stat->max_ms);
        }
    }
} // namespace akari::gpu