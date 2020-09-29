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
#include <akari/core/parallel.h>
namespace akari {
    class Device;
    class MemoryResource : public astd::pmr::memory_resource {
      public:
        Device *device() const { return _device; }
        MemoryResource(Device *device) : _device(device) {}

      protected:
        Device *_device;
    };

    class ManagedMemoryResource : public MemoryResource {
      public:
        using MemoryResource::MemoryResource;
    };
    class HostMemoryResource : public MemoryResource {
      public:
        using MemoryResource::MemoryResource;
    };
    class DeviceMemoryResource : public MemoryResource {
      public:
        using MemoryResource::MemoryResource;
    };
    class Device {
      public:
        virtual HostMemoryResource *host_resource() = 0;
        virtual DeviceMemoryResource *device_resource() = 0;
        virtual ManagedMemoryResource *managed_resource() = 0;
        virtual void copy_host_to_device(void *device_ptr, const void *host_ptr, size_t bytes) = 0;
        virtual void copy_device_to_host(void *host_ptr, const void *device_ptr, size_t bytes) = 0;
        virtual void copy_host_to_device_async(void *device_ptr, const void *host_ptr, size_t bytes) = 0;
        virtual void copy_device_to_host_async(void *host_ptr, const void *device_ptr, size_t bytes) = 0;
        virtual void prefetch_managed(void *p, size_t bytes) = 0;
        virtual bool is_gpu() const { return false; }
        virtual void sync() {}
    };
    AKR_EXPORT Device *cpu_device();
    AKR_EXPORT Device *gpu_device();

    AKR_EXPORT void set_device_gpu();
    AKR_EXPORT void set_device_cpu();
    AKR_EXPORT Device *active_device();

    // allocate only; no need to free (manually)
    template <typename Resource = astd::pmr::memory_resource>
    class TrackedMemoryResource : public Resource {
      protected:
        std::unordered_map<void *, std::pair<size_t, size_t>> allocated;
        Resource *resource;

      public:
        TrackedMemoryResource(Resource *resource) : Resource(resource->device()), resource(resource) {}
        TrackedMemoryResource(const TrackedMemoryResource &) = delete;
        void *do_allocate(size_t bytes, size_t alignment) override {
            auto p = resource->allocate(bytes, alignment);
            allocated.emplace(p, std::make_pair(bytes, alignment));
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) override {
            AKR_ASSERT(allocated.find(p) != allocated.end());
            resource->deallocate(p, bytes, alignment);
            allocated.erase(p);
        }
        bool do_is_equal(const astd::pmr::memory_resource &other) const noexcept override{ return &other == this; }
        void prefetch() {
            if constexpr (std::is_base_of_v<ManagedMemoryResource, Resource>) {
                for (auto p : allocated) {
                    auto [bytes, alignment] = p.second;
                    resource->device()->prefetch_managed(p.first, bytes);
                }
            }
        }
        ~TrackedMemoryResource() {
            for (auto p : allocated) {
                auto [bytes, alignment] = p.second;
                resource->deallocate(p.first, bytes, alignment);
            }
            allocated.clear();
        }
    };
    using TrackedManagedMemoryResource = TrackedMemoryResource<ManagedMemoryResource>;
    using TrackedDeviceMemeoryResource = TrackedMemoryResource<DeviceMemoryResource>;

    inline MemoryResource *default_resource() { return active_device()->managed_resource(); }

    template <typename T>
    using TAllocator = astd::pmr::polymorphic_allocator<T>;
} // namespace akari