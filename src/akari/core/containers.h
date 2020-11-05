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
#include <akari/core/fwd.h>
#include <akari/core/math.h>
#include <akari/core/astd.h>
#include <akari/core/memory.h>

namespace akari {
    inline uint64_t murmur_hash64a(const void *key, int len, uint64_t seed) {
        const uint64_t m = 0xc6a4a7935bd1e995ull;
        const int r = 47;

        uint64_t h = seed ^ (len * m);

        const uint64_t *data = (const uint64_t *)key;
        const uint64_t *end = data + (len / 8);

        while (data != end) {
            uint64_t k = *data++;

            k *= m;
            k ^= k >> r;
            k *= m;

            h ^= k;
            h *= m;
        }

        const unsigned char *data2 = (const unsigned char *)data;

        switch (len & 7) {
        case 7:
            h ^= uint64_t(data2[6]) << 48;
        case 6:
            h ^= uint64_t(data2[5]) << 40;
        case 5:
            h ^= uint64_t(data2[4]) << 32;
        case 4:
            h ^= uint64_t(data2[3]) << 24;
        case 3:
            h ^= uint64_t(data2[2]) << 16;
        case 2:
            h ^= uint64_t(data2[1]) << 8;
        case 1:
            h ^= uint64_t(data2[0]);
            h *= m;
        };

        h ^= h >> r;
        h *= m;
        h ^= h >> r;

        return h;
    }
    template <typename T>
    uint64_t hash_chain(T v, uint64_t seed = 0) {
        T tmp = v;
        return murmur_hash64a(&tmp, sizeof(T), seed);
    }
    template <typename T>
    uint64_t hash(T v) {
        return hash_chain(v, 0);
    }
    template <typename T, typename... Ts>
    uint64_t hash(T v, Ts... u) {
        return hash_chain(v, hash(u...));
    }
    // from pbrt-v4
    template <typename Key, typename Value, typename Hash,
              typename Allocator = astd::pmr::polymorphic_allocator<std::optional<std::pair<Key, Value>>>>
    class HashMap {
      public:
        // HashMap Type Definitions
        using TableEntry = std::optional<std::pair<Key, Value>>;

        class Iterator {
          public:
            
            Iterator &operator++() {
                while (++ptr < end && !ptr->has_value())
                    ;
                return *this;
            }

            
            Iterator operator++(int) {
                Iterator old = *this;
                operator++();
                return old;
            }

            
            bool operator==(const Iterator &iter) const { return ptr == iter.ptr; }
            
            bool operator!=(const Iterator &iter) const { return ptr != iter.ptr; }

            
            std::pair<Key, Value> &operator*() { return ptr->value(); }
            
            const std::pair<Key, Value> &operator*() const { return ptr->value(); }

            
            std::pair<Key, Value> *operator->() { return &ptr->value(); }
            
            const std::pair<Key, Value> *operator->() const { return ptr->value(); }

          private:
            friend class HashMap;
            Iterator(TableEntry *ptr, TableEntry *end) : ptr(ptr), end(end) {}
            TableEntry *ptr;
            TableEntry *end;
        };

        using iterator = Iterator;
        using const_iterator = const iterator;

        // HashMap Public Methods
        
        size_t size() const { return nStored; }
        
        size_t capacity() const { return table.size(); }
        void clear() {
            table.clear();
            nStored = 0;
        }

        HashMap(Allocator alloc) : table(8, alloc) {}

        HashMap(const HashMap &) = delete;
        HashMap &operator=(const HashMap &) = delete;

        void insert(const Key &key, const Value &value) {
            size_t offset = FindOffset(key);
            if (table[offset].has_value() == false) {
                // Grow hash table if it is too full
                if (3 * ++nStored > capacity()) {
                    Grow();
                    offset = FindOffset(key);
                }
            }
            table[offset] = std::make_pair(key, value);
        }
        
        bool contains(const Key &key) const { return table[FindOffset(key)].has_value(); }
        std::optional<Value> lookup(const Key &key) const {
            size_t offset = FindOffset(key);
            if (table[offset].has_value()) {
                return table[offset]->second;
            }
            return std::nullopt;
        }
        
        const Value &operator[](const Key &key) const {
            size_t offset = FindOffset(key);
            AKR_CHECK(table[offset].has_value());
            return table[offset]->second;
        }

        
        iterator begin() {
            Iterator iter(table.data(), table.data() + capacity());
            while (iter.ptr < iter.end && !iter.ptr->has_value())
                ++iter.ptr;
            return iter;
        }
        
        iterator end() { return Iterator(table.data() + capacity(), table.data() + capacity()); }

      private:
        // HashMap Private Methods
        
        size_t FindOffset(const Key &key) const {
            size_t baseOffset = Hash()(key) & (capacity() - 1);
            for (int nProbes = 0;; ++nProbes) {
                // Find offset for _key_ using quadratic probing
                size_t offset = (baseOffset + nProbes / 2 + nProbes * nProbes / 2) & (capacity() - 1);
                if (table[offset].has_value() == false || key == table[offset]->first)
                    return offset;
            }
        }

        void Grow() {
            size_t currentCapacity = capacity();
            std::vector<TableEntry, Allocator> newTable(std::max<size_t>(64, 2 * currentCapacity),
                                                         table.get_allocator());
            size_t newCapacity = newTable.size();
            for (size_t i = 0; i < currentCapacity; ++i) {
                // insert _table[i]_ into _newTable_ if it is set
                if (!table[i].has_value())
                    continue;
                size_t baseOffset = Hash()(table[i]->first) & (newCapacity - 1);
                for (int nProbes = 0;; ++nProbes) {
                    size_t offset = (baseOffset + nProbes / 2 + nProbes * nProbes / 2) & (newCapacity - 1);
                    if (!newTable[offset]) {
                        newTable[offset] = std::move(*table[i]);
                        break;
                    }
                }
            }
            table = std::move(newTable);
        }

        // HashMap Private Members
        std::vector<TableEntry, Allocator> table;
        size_t nStored = 0;
    };
    
} // namespace akari