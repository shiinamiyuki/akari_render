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

#include <cuda.h>
#include <akari/common/fwd.h>
#include <akari/common/tuple.h>
#include <akari/common/soa.h>
#include <akari/kernel/soa.h>

namespace akari {
    template <typename T>
    class WorkQueue : SOA<T> {
        using value_type = T;
        using SOA<T>::SOA<T>;
        size_t head = 0;
        AKR_XPU void append(const T &el) {
            auto i = atomicAdd(&head, 1);
            AKR_ASSERT(i < head);
            buffer[i] = el;
        }
        AKR_XPU size_t elements_in_queue() const { return head; }
        AKR_XPU void clear() { head = 0; }
    };

    template <typename... Ts>
    struct MultiWorkQueue : Tuple<WorkQueue<Ts>...> {
        MultiWorkQueue(MemoryArena &arena, size_t max_size) {
            foreach_cpu([](auto &arg) {
                using Queue = std::decay_t<decltype(arg)>;
                using T = typename Queue::value_type;
                arg = Queue(arena.allocN<T>(), max_size);
            });
        }
    };
} // namespace akari