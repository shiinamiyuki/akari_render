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
#include <akari/common.h>
// enabling C++20 pmr

namespace akari::astd {

    enum class byte : unsigned char {};

    // MSVC has incomplete pmr support ...
    namespace pmr {
        class AKR_EXPORT memory_resource {
            static constexpr size_t max_align = alignof(std::max_align_t);

          public:
            virtual ~memory_resource() = default;
            void *allocate(size_t bytes, size_t alignment = max_align) { return do_allocate(bytes, alignment); }
            void deallocate(void *p, size_t bytes, size_t alignment = max_align) {
                return do_deallocate(p, bytes, alignment);
            }
            bool is_equal(const memory_resource &other) const noexcept { return do_is_equal(other); }

          private:
            virtual void *do_allocate(size_t bytes, size_t alignment) = 0;
            virtual void do_deallocate(void *p, size_t bytes, size_t alignment) = 0;
            virtual bool do_is_equal(const memory_resource &other) const noexcept = 0;
        };

        AKR_EXPORT memory_resource *new_delete_resource() noexcept;
        AKR_EXPORT memory_resource *set_default_resource(memory_resource *r) noexcept;
        AKR_EXPORT memory_resource *get_default_resource() noexcept;

        template <class Tp = astd::byte>
        class polymorphic_allocator {
          public:
            using value_type = Tp;
            polymorphic_allocator() noexcept { memoryResource = get_default_resource(); }
            polymorphic_allocator(memory_resource *r) : memoryResource(r) {}
            polymorphic_allocator(const polymorphic_allocator &other) = default;
            template <class U>
            polymorphic_allocator(const polymorphic_allocator<U> &other) noexcept : memoryResource(other.resource()) {}
            polymorphic_allocator &operator=(const polymorphic_allocator &rhs) = default;
            polymorphic_allocator &operator=(polymorphic_allocator &&rhs) {
                this->memoryResource = rhs.memoryResource;
                rhs.memoryResource = nullptr;
                return *this;
            }
            [[nodiscard]] Tp *allocate(size_t n) {
                return static_cast<Tp *>(resource()->allocate(n * sizeof(Tp), alignof(Tp)));
            }
            template <class T>
            bool operator==(const polymorphic_allocator<T> &rhs) const {
                return resource() == rhs.resource();
            }
            template <class T>
            bool operator!=(const polymorphic_allocator<T> &rhs) const {
                return resource() != rhs.resource();
            }
            void deallocate(Tp *p, size_t n) { resource()->deallocate(p, n); }

            void *allocate_bytes(size_t nbytes, size_t alignment = alignof(max_align_t)) {
                return resource()->allocate(nbytes, alignment);
            }
            void deallocate_bytes(void *p, size_t nbytes, size_t alignment = alignof(std::max_align_t)) {
                return resource()->deallocate(p, nbytes, alignment);
            }
            template <class T>
            T *allocate_object(size_t n = 1) {
                return static_cast<T *>(allocate_bytes(n * sizeof(T), alignof(T)));
            }
            template <class T>
            void deallocate_object(T *p, size_t n = 1) {
                deallocate_bytes(p, n * sizeof(T), alignof(T));
            }
            template <class T, class... Args>
            void construct(T *p, Args &&... args) {
                ::new ((void *)p) T(std::forward<Args>(args)...);
            }
            template <class T, class... Args>
            T *new_object(Args &&... args) {
                auto *p = allocate_object<T>();
                construct(p, std::forward<Args>(args)...);
                return p;
            }
            template <class T>
            void destroy(T *p) {
                p->~T();
            }
            memory_resource *resource() const { return memoryResource; }

          private:
            memory_resource *memoryResource;
        };

        struct pool_options {
            size_t max_blocks_per_chunk = 0;
            size_t largest_required_pool_block = 0;
        };

        class synchronized_pool_resource : public memory_resource {
          public:
            synchronized_pool_resource(const pool_options &opts, memory_resource *upstream);

            synchronized_pool_resource() : synchronized_pool_resource(pool_options(), get_default_resource()) {}
            explicit synchronized_pool_resource(memory_resource *upstream)
                : synchronized_pool_resource(pool_options(), upstream) {}
            explicit synchronized_pool_resource(const pool_options &opts)
                : synchronized_pool_resource(opts, get_default_resource()) {}

            synchronized_pool_resource(const synchronized_pool_resource &) = delete;
            virtual ~synchronized_pool_resource();

            synchronized_pool_resource &operator=(const synchronized_pool_resource &) = delete;

            void release();
            memory_resource *upstream_resource() const;
            pool_options options() const;

          protected:
            void *do_allocate(size_t bytes, size_t alignment) override;
            void do_deallocate(void *p, size_t bytes, size_t alignment) override;

            bool do_is_equal(const memory_resource &other) const noexcept override;
        };

        class unsynchronized_pool_resource : public memory_resource {
          public:
            unsynchronized_pool_resource(const pool_options &opts, memory_resource *upstream);

            unsynchronized_pool_resource() : unsynchronized_pool_resource(pool_options(), get_default_resource()) {}
            explicit unsynchronized_pool_resource(memory_resource *upstream)
                : unsynchronized_pool_resource(pool_options(), upstream) {}
            explicit unsynchronized_pool_resource(const pool_options &opts)
                : unsynchronized_pool_resource(opts, get_default_resource()) {}

            unsynchronized_pool_resource(const unsynchronized_pool_resource &) = delete;
            virtual ~unsynchronized_pool_resource();

            unsynchronized_pool_resource &operator=(const unsynchronized_pool_resource &) = delete;

            void release();
            memory_resource *upstream_resource() const;
            pool_options options() const;

          protected:
            void *do_allocate(size_t bytes, size_t alignment) override;
            void do_deallocate(void *p, size_t bytes, size_t alignment) override;

            bool do_is_equal(const memory_resource &other) const noexcept override;
        };

        class monotonic_buffer_resource : public memory_resource {
            memory_resource *upstream_rsrc;
            template <size_t alignment>
            static constexpr size_t align(size_t x) {
                static_assert((alignment & (alignment - 1)) == 0);
                x = (x + alignment - 1) & (~(alignment - 1));
                return x < 4 ? 4 : x;
            }

            struct Block {
                size_t size;
                astd::byte *data;
                Block(size_t size, astd::byte *data) : size(size), data(data) {}
                Block(size_t size, memory_resource *resource, size_t alignment = 8) : size(size) {
                    data = (astd::byte *)resource->allocate(size, alignment);
                    AKR_ASSERT(data);
                }

                ~Block() = default;
            };
            static constexpr size_t DEFAULT_BLOCK_SIZE = 262144ull;
            std::list<Block> availableBlocks, usedBlocks;
            size_t currentBlockPos = 0;
            Block currentBlock;

          public:
            explicit monotonic_buffer_resource(memory_resource *upstream)
                : monotonic_buffer_resource(DEFAULT_BLOCK_SIZE, upstream) {}
            monotonic_buffer_resource(size_t initial_size, memory_resource *upstream)
                : upstream_rsrc(upstream), currentBlock(initial_size, upstream) {}
            monotonic_buffer_resource(void *buffer, size_t buffer_size, memory_resource *upstream)
                : upstream_rsrc(upstream), currentBlock(buffer_size, (astd::byte *)buffer) {}

            monotonic_buffer_resource() : monotonic_buffer_resource(get_default_resource()) {}
            explicit monotonic_buffer_resource(size_t initial_size)
                : monotonic_buffer_resource(initial_size, get_default_resource()) {}
            monotonic_buffer_resource(void *buffer, size_t buffer_size)
                : monotonic_buffer_resource(buffer, buffer_size, get_default_resource()) {}

            monotonic_buffer_resource(const monotonic_buffer_resource &) = delete;

            virtual ~monotonic_buffer_resource() {
                upstream_rsrc->deallocate(currentBlock.data, currentBlock.size);
                for (auto i : availableBlocks) {
                    upstream_rsrc->deallocate(i.data, i.size);
                }
                for (auto i : usedBlocks) {
                    upstream_rsrc->deallocate(i.data, i.size);
                }
            }

            monotonic_buffer_resource &operator=(const monotonic_buffer_resource &) = delete;

            void release() {
                currentBlockPos = 0;
                availableBlocks.splice(availableBlocks.begin(), usedBlocks);
            }
            memory_resource *upstream_resource() const { return upstream_rsrc; }

          protected:
            void *do_allocate(size_t bytes, size_t alignment) override {
                typename std::list<Block>::iterator iter;
                void *p = nullptr;
                size_t storage = bytes;
                if (intptr_t(currentBlock.data + currentBlockPos) % alignment != 0) {
                    bytes += alignment;
                }
                if (currentBlockPos + bytes > currentBlock.size) {
                    usedBlocks.emplace_front(currentBlock);
                    currentBlockPos = 0;
                    for (iter = availableBlocks.begin(); iter != availableBlocks.end(); iter++) {
                        if (iter->size >= bytes) {
                            currentBlockPos = bytes;
                            currentBlock = *iter;
                            availableBlocks.erase(iter);
                            break;
                        }
                    }
                    if (iter == availableBlocks.end()) {
                        auto sz = std::max<size_t>(bytes, DEFAULT_BLOCK_SIZE);
                        currentBlock = Block(sz, upstream_rsrc, alignment);
                    }
                }
                p = currentBlock.data + currentBlockPos;
                currentBlockPos += bytes;
                auto q = std::align(alignment, storage, p, bytes);
                AKR_ASSERT(q);
                return p;
            }
            void do_deallocate(void *p, size_t bytes, size_t alignment) override {}
            bool do_is_equal(const memory_resource &other) const noexcept override { return this == &other; }
        };
    } // namespace pmr

    template <typename T, int N>
    class array {
      public:
        using value_type = T;
        using iterator = value_type *;
        using const_iterator = const value_type *;
        using size_t = std::size_t;

        array() = default;

        array(std::initializer_list<T> v) {
            size_t i = 0;
            for (const T &val : v)
                values[i++] = val;
        }

        void fill(const T &v) {
            for (int i = 0; i < N; ++i)
                values[i] = v;
        }

        bool operator==(const array<T, N> &a) const {
            for (int i = 0; i < N; ++i)
                if (values[i] != a.values[i])
                    return false;
            return true;
        }

        bool operator!=(const array<T, N> &a) const { return !(*this == a); }

        iterator begin() { return values; }

        iterator end() { return values + N; }

        const_iterator begin() const { return values; }

        const_iterator end() const { return values + N; }

        size_t size() const { return N; }

        T &operator[](size_t i) { return values[i]; }

        const T &operator[](size_t i) const { return values[i]; }

        T *data() { return values; }

        const T *data() const { return values; }

      private:
        T values[N] = {};
    };
    namespace pmr {
        template <typename T>
        using vector = std::vector<T, polymorphic_allocator<T>>;
    }
} // namespace akari::astd