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
    class MemoryArena {
        static constexpr size_t align16(size_t x) { return (x + 15ULL) & (~15ULL); }

        struct Block {
            size_t size;
            uint8_t *data;

            Block(size_t size) : size(size) { data = (uint8_t *)get_device_memory_resource()->allocate(size); }

            ~Block() = default;
        };

        std::list<Block> availableBlocks, usedBlocks;

        size_t currentBlockPos = 0;
        Block currentBlock;

      public:
        static constexpr size_t DEFAULT_BLOCK_SIZE = 262144ull;
        MemoryArena() : currentBlock(DEFAULT_BLOCK_SIZE) {}
        uint8_t *alloc_bytes(size_t bytes) {
            typename std::list<Block>::iterator iter;
            uint8_t *p = nullptr;
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
                    currentBlock = Block(sz);
                }
            }
            p = currentBlock.data + currentBlockPos;
            currentBlockPos += bytes;
            return p;
        }
        template <typename T, typename... Args>
        T *allocN(size_t count, Args &&... args) {
            auto allocSize = sizeof(T) * count;
            auto p = alloc_bytes(allocSize);
            if constexpr (!std::is_trivially_constructible_v<T>) {
                for (size_t i = 0; i < count; i++) {
                    new (p + i * sizeof(T)) T(std::forward<Args>(args)...);
                }
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
            get_device_memory_resource()->deallocate(currentBlock.data, currentBlock.size);
            for (auto i : availableBlocks) {
                get_device_memory_resource()->deallocate(i.data, i.size);
            }
            for (auto i : usedBlocks) {
                get_device_memory_resource()->deallocate(i.data, i.size);
            }
        }
    };
} // namespace akari
#endif // AKARIRENDER_MEMORYARENA_HPP
