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

#ifndef AKARIRENDER_MEMORYARENA_HPP
#define AKARIRENDER_MEMORYARENA_HPP
#include <cstddef>
#include <cstdint>
#include <list>
#include <akari/common/buffer.h>
namespace akari {
    template <typename Allocator = astd::pmr::polymorphic_allocator<>>
    class MemoryArena {
        template <size_t alignment>
        static constexpr size_t align(size_t x) {
            static_assert((alignment & (alignment - 1)) == 0);
            x = (x + alignment - 1) & (~(alignment - 1));
            return x < 4 ? 4 : x;
        }

        struct Block {
            size_t size;
            std::byte *data;

            Block(size_t size, Allocator &allocator) : size(size) {
                data = allocator.allocate(size);
                AKR_ASSERT(data);
            }

            ~Block() = default;
        };

        std::list<Block> availableBlocks, usedBlocks;
        Allocator allocator;
        size_t currentBlockPos = 0;
        Block currentBlock;

      public:
        static constexpr size_t DEFAULT_BLOCK_SIZE = 262144ull;
        MemoryArena(Allocator allocator) : allocator(allocator), currentBlock(DEFAULT_BLOCK_SIZE, allocator) {}
        std::byte *alloc_bytes(size_t bytes) {
            typename std::list<Block>::iterator iter;
            std::byte *p = nullptr;
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
                    currentBlock = Block(sz, allocator);
                }
            }
            p = currentBlock.data + currentBlockPos;
            currentBlockPos += bytes;
            return p;
        }
        template <typename T, typename... Args>
        T *allocN(size_t count, Args &&... args) {
            auto allocSize = align<alignof(T)>(sizeof(T) * count + alignof(T));
            auto p = alloc_bytes(allocSize);
            auto q = (void *)p;
            AKR_ASSERT(astd::align(alignof(T), sizeof(T), q, allocSize));
            p = (std::byte *)q;
            for (size_t i = 0; i < count; i++) {
                new (p + i * sizeof(T)) T(std::forward<Args>(args)...);
            }

            return reinterpret_cast<T *>(p);
        }

        template <typename T, typename... Args>
        T *alloc(Args &&... args) {
            return allocN<T>(1, std::forward<Args>(args)...);
        }

        void reset() {
            currentBlockPos = 0;
            availableBlocks.splice(availableBlocks.begin(), usedBlocks);
        }

        ~MemoryArena() {
            allocator.deallocate(currentBlock.data, currentBlock.size);
            for (auto i : availableBlocks) {
                allocator.deallocate(i.data, i.size);
            }
            for (auto i : usedBlocks) {
                allocator.deallocate(i.data, i.size);
            }
        }
    };
} // namespace akari
#endif // AKARIRENDER_MEMORYARENA_HPP
