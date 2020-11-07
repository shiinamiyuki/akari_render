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
#include <optional>
#include <istream>
#include <ostream>
#include <akari/core/akari.h>
#include <akari/core/memory.h>
#include <unordered_map>

// Simple serialization
namespace akari {
    class Serializable;
}
namespace akari {
    class Stream {};
    class InputArchive;
    class OutputArchive;
    class Serializable {
      public:
        virtual void save(OutputArchive &ar) const {}
        virtual void load(InputArchive &ar) {}
    };
    template <class T>
    struct shared_ptr_trait {
        using value_type = void;
        static constexpr bool value = false;
    };
    template <class T>
    struct shared_ptr_trait<std::shared_ptr<T>> {
        using value_type = T;
        static constexpr bool value = true;
    };
    // class AKR_EXPORT InputArchive : public cereal::BinaryInputArchive {
    //     class Impl;
    //     std::shared_ptr<Impl> impl;

    //   public:
    //     InputArchive(std::istream &stream);
    //     std::shared_ptr<Serializable> load();
    //     template <class T, class... Ts>
    //     void operator()(T &first, Ts &... args) {
    //         if constexpr (shared_ptr_trait<T>::value) {
    //             using value_type = typename shared_ptr_trait<T>::type;
    //             static_assert(std::is_base_of_v<Serializable, value_type>);
    //             first = load();
    //         } else {
    //             (*this)(first);
    //         }
    //         if constexpr (sizeof...(args) > 0) {
    //             (*this)(args...);
    //         }
    //     }
    // };
    class AKR_EXPORT OutputArchive {
      public:
        // virtual void save_i1(bool) = 0;
        // virtual void save_i8(int8_t) = 0;
        // virtual void save_i32(int32_t) = 0;
        // virtual void save_i64(int64_t) = 0;
        // virtual void save_u8(uint8_t) = 0;
        // virtual void save_u32(uint32_t) = 0;
        // virtual void save_u64(uint64_t) = 0;
        // virtual void save_f32(float) = 0;
        // virtual void save_f64(double) = 0;
        virtual void save_bytes(char *buf, size_t bytes) = 0;
        virtual void save_ptr(const std::shared_ptr<Serializable> &p) = 0;
        virtual void begin_object() = 0;
        virtual void end_object() = 0;
        template <class T>
        std::enable_if_t<std::is_fundamental_v<T>> do_save(const T &arg) {
            save_bytes(reinterpret_cast<const char *>(&arg), sizeof(T));
        }
        template <class T>
        std::enable_if_t<!std::is_fundamental_v<T> && !shared_ptr_trait<T>::value, T> do_save(const T &arg) {
            save(*this, arg);
        }
        template <class T>
        void do_save(const std::shared_ptr<T> &ptr) {
            save_ptr(ptr);
        }
        template <class T, class... Ts>
        void operator()(const T &first, const Ts &... args) {
            do_save(first);
            if constexpr (sizeof...(args) > 0) {
                (*this)(args...);
            }
        }
    };
    class AKR_EXPORT InputArchive {
      public:
        // virtual bool load_i1(bool &) = 0;
        // virtual bool load_i8(int8_t &) = 0;
        // virtual bool load_i32(int32_t &) = 0;
        // virtual bool load_i64(int64_t &) = 0;
        // virtual bool load_u8(int8_t &) = 0;
        // virtual bool load_u32(uint32_t &) = 0;
        // virtual bool load_u64(uint64_t &) = 0;
        // virtual bool load_f32(float) = 0;
        // virtual bool load_f64(double) = 0;
        template <class T, class... Ts>
        void operator()(T &first, Ts &... args) {
            do_load(first);
            if constexpr (sizeof...(args) > 0) {
                (*this)(args...);
            }
        }
        template <class T>
        std::enable_if_t<std::is_fundamental_v<T>> do_load(T &arg) {
            load_bytes(reinterpret_cast<char *>(&arg), sizeof(T));
        }
        template <class T>
        std::enable_if_t<!std::is_fundamental_v<T> && !shared_ptr_trait<T>::value> do_load(T &arg) {
            load(*this, arg);
        }
        template <class T>
        void do_load(std::shared_ptr<T> &ptr) {
            ptr = load();
        }
        virtual bool load_bytes(const char *buf, size_t bytes) = 0;
        virtual std::shared_ptr<Serializable> load() = 0;
        virtual void begin_object() = 0;
        virtual void end_object() = 0;
    };

    AKR_EXPORT std::unique_ptr<InputArchive> create_input_archive(std::istream &in);
} // namespace akari