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

#include <akari/pmr.h>

namespace akari::astd {
    namespace pmr {

        class new_delete_resource_impl : public memory_resource {
          public:
            void *do_allocate(size_t bytes, size_t alignment) {
#ifdef WIN32
                auto p = _aligned_malloc(bytes, std::max<size_t>(16, alignment));
                AKR_ASSERT(p);
                return p;
#else
                return aligned_alloc(alignment, bytes);
#endif
            }
            void do_deallocate(void *p, size_t bytes, size_t alignment) {
#ifdef WIN32
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