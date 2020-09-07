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
#include <akari/core/parallel.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace akari {
    size_t num_work_threads() { return std::thread::hardware_concurrency(); }
    struct ParallelForContext {
        std::atomic_uint32_t workIndex;
        size_t count = 0;
        uint32_t chunkSize = 0;
        ParallelForContext() : workIndex(0) {}
        const std::function<void(uint32_t, uint32_t)> *func = nullptr;
        bool done() const { return workIndex >= count; }
        ParallelForContext(const ParallelForContext &rhs)
            : workIndex(rhs.workIndex.load()), count(rhs.count), chunkSize(rhs.chunkSize), func(rhs.func) {}
    };

    struct ParallelForWorkPool {
        std::deque<ParallelForContext> works;
        std::vector<std::thread> threads;
        std::condition_variable hasWork, oneThreadFinished, mainWaiting;
        std::mutex workMutex;
        std::atomic_bool stopped;
        std::uint32_t workId;
        std::uint32_t nThreadFinished;
        ParallelForWorkPool() : workId(0), nThreadFinished(0) {
            stopped = false;
            auto n = num_work_threads();
            for (uint32_t tid = 0; tid < n; tid++) {
                threads.emplace_back([=]() {
                    while (!stopped) {
                        std::unique_lock<std::mutex> lock(workMutex);
                        while (works.empty() && !stopped) {
                            hasWork.wait(lock);
                        }
                        if (stopped)
                            return;
                        auto &loop = works.front();
                        auto id = workId;
                        lock.unlock();
                        // lock held
                        while (!loop.done()) {
                            auto begin = loop.workIndex.fetch_add(loop.chunkSize);
                            for (auto i = begin; i < begin + loop.chunkSize && i < loop.count; i++) {
                                (*loop.func)(i, tid);
                            }
                        }
                        lock.lock();
                        nThreadFinished++;
                        oneThreadFinished.notify_all();

                        while (nThreadFinished != (uint32_t)threads.size() && workId == id) {
                            oneThreadFinished.wait(lock);
                        }

                        if (workId == id) {
                            workId++; // only one thread would reach here
                            works.pop_front();
                            if (works.empty()) {
                                mainWaiting.notify_one();
                            }
                            nThreadFinished = 0;
                        }
                        lock.unlock();
                    }
                });
            }
        }
        void enqueue(const ParallelForContext &context) {
            std::lock_guard<std::mutex> lock(workMutex);
            works.emplace_back(context);
            hasWork.notify_all();
        }
        void wait() {
            std::unique_lock<std::mutex> lock(workMutex);
            while (!works.empty()) {

                mainWaiting.wait(lock);
            }
        }
        ~ParallelForWorkPool() {
            stopped = true;
            hasWork.notify_all();
            for (auto &thr : threads) {
                thr.join();
            }
        }
    };
    static std::once_flag flag;
    static std::unique_ptr<ParallelForWorkPool> pool;
    void parallel_for(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunkSize) {

        std::call_once(flag, [&]() { pool = std::make_unique<ParallelForWorkPool>(); });
        ParallelForContext ctx;
        ctx.func = &func;
        ctx.chunkSize = (uint32_t)chunkSize;
        ctx.count = count;
        ctx.workIndex = 0;
        pool->enqueue(ctx);
        pool->wait();
    }

    void ThreadPoolFinalize() { pool.reset(nullptr); }
} // namespace akari