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
#include <akari/common/def.h>
#include <akari/common/fwd.h>
#include <akari/common/tuple.h>
#include <akari/common/soa.h>
#include <akari/kernel/soa.h>
#include <cooperative_groups.h>
namespace akari {
#if 0
    template <typename T>
    struct WorkQueue {
        using value_type = T;
        int _size = 0;
        T *__restrict__ array = nullptr;
        size_t head = 0;
        template <class Allocator>
        WorkQueue(size_t s, Allocator &&allocator) : _size(s) {
            array = allocator.template allocate_object<T>(s);
        }
        AKR_XPU T &operator[](int i) { return array[i]; }
        AKR_XPU const T &operator[](int i) const { return array[i]; }
        AKR_XPU size_t size() const { return _size; }
        AKR_XPU int append(const T &el) {
            auto i = atomicAdd(&head, 1);
            AKR_ASSERT(i < size());
            (*this)[i] = el;
            return i;
        }
        AKR_XPU size_t elements_in_queue() const { return head; }
        AKR_XPU void clear() { head = 0; }
    };
#else
    template <typename T>
    struct WorkQueue : SOA<T> {
        using value_type = T;
        using SOA<T>::SOA;
        int head = 0;
        AKR_XPU int append(const T &el) {
            auto i = atomicAdd(&head, 1);
            AKR_ASSERT(i < size());
            (*this)[i] = el;
            return i;
        }
        AKR_XPU int allocate(int n) {
            auto i = atomicAdd(&head, n);
            AKR_ASSERT(i + n <= size());
            return i;
        }
        AKR_XPU size_t elements_in_queue() const { return head; }
        AKR_XPU void clear() { head = 0; }
    };
#endif
    template <typename... Ts>
    struct MultiWorkQueue : Tuple<WorkQueue<Ts>...> {
        template <class Allocator>
        MultiWorkQueue(size_t max_size, Allocator &allocator) {
            foreach_cpu([&](auto &arg) {
                using Queue = std::decay_t<decltype(arg)>;
                using T = typename Queue::value_type;
                arg = Queue(max_size, allocator);
            });
        }
    };
} // namespace akari