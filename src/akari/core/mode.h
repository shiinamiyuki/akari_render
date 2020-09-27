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

#pragma once
#include <akari/common/astd.h>
#include <unordered_map>
namespace akari {
    AKR_EXPORT void set_device_gpu();
    AKR_EXPORT void set_device_cpu();
    AKR_EXPORT void sync_device();
    enum class ComputeDevice { cpu, gpu };
    AKR_EXPORT ComputeDevice get_device();

    class ManagedDeviceMemoryResource : public astd::pmr::memory_resource {
      public:
        virtual void prefech_to_device(void *p, size_t) {}
    };
    AKR_EXPORT astd::pmr::memory_resource *get_managed_memory_resource();
    AKR_EXPORT astd::pmr::memory_resource *get_device_memory_resource();
    // allocate only; no need to free (manually)
    class AKR_EXPORT TrackedDeviceMemoryResource : public astd::pmr::memory_resource {
        astd::pmr::memory_resource *resource;
        std::unordered_map<void *, std::pair<size_t, size_t>> allocated;

      public:
        TrackedDeviceMemoryResource(const TrackedDeviceMemoryResource &) = delete;
        TrackedDeviceMemoryResource(astd::pmr::memory_resource *resource) : resource(resource) {}
        void *do_allocate(size_t bytes, size_t alignment) {
            auto p = resource->allocate(bytes, alignment);
            allocated.emplace(p, std::make_pair(bytes, alignment));
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) {
            AKR_ASSERT(allocated.find(p) != allocated.end());
            resource->deallocate(p, bytes, alignment);
            allocated.erase(p);
        }
        bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
        void prefech_to_device() {
            auto r = dynamic_cast<ManagedDeviceMemoryResource *>(resource);
            AKR_ASSERT(r);
            for (auto p : allocated)
                r->prefech_to_device(p.first, p.second.first);
        }
        ~TrackedDeviceMemoryResource() {
            for (auto p : allocated) {
                auto [bytes, alignment] = p.second;
                resource->deallocate(p.first, bytes, alignment);
            }
            allocated.clear();
        }
    };
} // namespace akari