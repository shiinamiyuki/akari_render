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

#include <akari/core/astd.h>
#include <akari/core/memory.h>
namespace akari::astd {
    namespace pmr {
        

        class new_delete_resource_impl : public memory_resource {
          public:
            void *do_allocate(size_t bytes, size_t alignment) {
#ifdef AKR_PLATFORM_WINDOWS
                auto p = _aligned_malloc(bytes, std::max<size_t>(16, alignment));
                AKR_ASSERT(p);
                return p;
#else
                return aligned_alloc(alignment, bytes);
#endif
            }
            void do_deallocate(void *p, size_t bytes, size_t alignment) {
#ifdef AKR_PLATFORM_WINDOWS
                return _aligned_free(p);
#else
                return free(p);
#endif
            }
            bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
        };
        static new_delete_resource_impl _new_delete_resource;
        static memory_resource *_default_resource = &_new_delete_resource;
        AKR_EXPORT memory_resource *new_delete_resource() noexcept { return &_new_delete_resource; }
        AKR_EXPORT memory_resource *set_default_resource(memory_resource *r) noexcept {
            auto tmp = _default_resource;
            _default_resource = r;
            return tmp;
        }
        AKR_EXPORT memory_resource *get_default_resource() noexcept { return _default_resource; }
    } // namespace pmr
} // namespace akari::astd