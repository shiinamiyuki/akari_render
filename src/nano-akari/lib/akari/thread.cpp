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
#include <akari/thread.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace akari::thread {

    struct ParallelForContext {
        std::atomic_uint32_t workIndex;
        size_t count = 0;
        uint32_t chunkSize = 0;
        ParallelForContext() : workIndex(0) {}
        const std::function<void(size_t, uint32_t)> *func = nullptr;
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
    namespace thread_internal {
        static std::once_flag flag;
        static std::unique_ptr<ParallelForWorkPool> pool;
        static size_t n_threads = std::thread::hardware_concurrency();
    } // namespace thread_internal
    size_t num_work_threads() { return thread_internal::n_threads; }
    void parallel_for_impl(size_t count, const std::function<void(size_t, uint32_t)> &func, size_t chunkSize) {
        using namespace thread_internal;
        if (!pool) {
            throw std::runtime_error("thread pool not initialized. call thread::init(num_threads);");
        }
        ParallelForContext ctx;
        ctx.func = &func;
        ctx.chunkSize = (uint32_t)chunkSize;
        ctx.count = count;
        ctx.workIndex = 0;
        pool->enqueue(ctx);
        pool->wait();
    }

    AKR_EXPORT void init(size_t num_threads) {
        using namespace thread_internal;
        if (pool) {
            AKR_PANIC("thread::init(num_threads); called multiple times");
        }
        n_threads = num_threads;
        std::call_once(flag, [&]() { pool = std::make_unique<ParallelForWorkPool>(); });
    }
    void finalize() {
        using namespace thread_internal;
        pool.reset(nullptr);
    }

} // namespace akari::thread