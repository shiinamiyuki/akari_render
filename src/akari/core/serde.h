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
#include <akari/core/stream.h>
#include <unordered_map>
#include <akari/core/color.h>
#include <json_fwd.hpp>

// Simple serialization
namespace akari {
    class Serializable;
}
namespace akari {
    class InputArchive;
    class OutputArchive;
    class Serializable {
      public:
        virtual void save(OutputArchive &ar) const {}
        virtual void load(InputArchive &ar) {}
        // this is used as an unique identifier for its type
        virtual std::string_view type_str() const = 0;
        using Constructor = std::function<Serializable *(std::string_view)>;
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
    template <class T>
    struct Serde {
        static void serialize(OutputArchive &ar, const T &arg) { arg.save(ar); }
        static void deserialize(InputArchive &ar, T &arg) { arg.load(ar); }
    };
    template <class T>
    static constexpr bool serde_serializable = !std::is_enum_v<T> && !std::is_fundamental_v<T> &&
                                               !shared_ptr_trait<T>::value && !std::is_same_v<T, std::string>;
    class AKR_EXPORT OutputArchive {
      public:
        virtual void save_i1(bool) = 0;
        virtual void save_i8(int8_t) = 0;
        virtual void save_i32(int32_t) = 0;
        virtual void save_i64(int64_t) = 0;
        virtual void save_u8(uint8_t) = 0;
        virtual void save_u32(uint32_t) = 0;
        virtual void save_u64(uint64_t) = 0;
        virtual void save_f32(float) = 0;
        virtual void save_f64(double) = 0;
        virtual void save_str(const std::string &) = 0;
        virtual void save_bytes(const char *buf, size_t bytes) = 0;
        virtual void save_ptr(const std::shared_ptr<Serializable> &p) = 0;
        virtual void begin_object() = 0;
        virtual void end_object() = 0;
        void do_save(bool v) { return save_i1(v); }
        void do_save(int8_t v) { return save_i8(v); }
        void do_save(int32_t v) { return save_i32(v); }
        void do_save(int64_t v) { return save_i64(v); }
        void do_save(uint8_t v) { return save_u8(v); }
        void do_save(uint32_t v) { return save_u32(v); }
        void do_save(uint64_t v) { return save_u64(v); }
        void do_save(float v) { return save_f32(v); }
        void do_save(double v) { return save_f64(v); }
        void do_save(const std::string &v) { return save_str(v); }
        template <class T>
        std::enable_if_t<std::is_enum_v<T>> do_save(const T &arg) {
            save_bytes(reinterpret_cast<const char *>(&arg), sizeof(T));
        }
        template <class T>
        std::enable_if_t<serde_serializable<T>> do_save(const T &arg) {
            begin_object();
            Serde<T>::serialize(*this, arg);
            end_object();
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
        virtual bool load_i1(bool &) = 0;
        virtual bool load_i8(int8_t &) = 0;
        virtual bool load_i32(int32_t &) = 0;
        virtual bool load_i64(int64_t &) = 0;
        virtual bool load_u8(uint8_t &) = 0;
        virtual bool load_u32(uint32_t &) = 0;
        virtual bool load_u64(uint64_t &) = 0;
        virtual bool load_f32(float &) = 0;
        virtual bool load_f64(double &) = 0;
        virtual bool load_str(std::string &s) = 0;
        template <class T, class... Ts>
        void operator()(T &first, Ts &... args) {
            do_load(first);
            if constexpr (sizeof...(args) > 0) {
                (*this)(args...);
            }
        }
        bool do_load(bool &v) { return load_i1(v); }
        bool do_load(int8_t &v) { return load_i8(v); }
        bool do_load(int32_t &v) { return load_i32(v); }
        bool do_load(int64_t &v) { return load_i64(v); }
        bool do_load(uint8_t &v) { return load_u8(v); }
        bool do_load(uint32_t &v) { return load_u32(v); }
        bool do_load(uint64_t &v) { return load_u64(v); }
        bool do_load(float &v) { return load_f32(v); }
        bool do_load(double &v) { return load_f64(v); }
        bool do_load(std::string &v) { return load_str(v); }
        template <class T>
        std::enable_if_t<std::is_enum_v<T>> do_load(T &arg) {
            load_bytes(reinterpret_cast<char *>(&arg), sizeof(T));
        }
        template <class T>
        std::enable_if_t<serde_serializable<T>> do_load(T &arg) {
            begin_object();
            Serde<T>::deserialize(*this, arg);
            end_object();
        }
        template <class T>
        void do_load(std::shared_ptr<T> &ptr) {
            ptr = dyn_cast<T>(load());
        }
        virtual bool load_bytes(char *buf, size_t bytes) = 0;
        virtual std::shared_ptr<Serializable> load() = 0;
        virtual void begin_object() = 0;
        virtual void end_object() = 0;
    };
    struct SerdeObjectRecord {
        std::string type_str;
        bool is_null = false;
        bool valid = true; // whether data is saved after this record
        size_t uid = 0;
        void load(InputArchive &ar) { ar(type_str, is_null, valid, uid); }
        void save(OutputArchive &ar) const { ar(type_str, is_null, valid, uid); }
    };
    AKR_EXPORT std::unique_ptr<OutputArchive> create_output_archive(nlohmann::json &);
    AKR_EXPORT std::unique_ptr<OutputArchive> create_output_archive(std::unique_ptr<Stream> stream);
    AKR_EXPORT std::unique_ptr<InputArchive> create_input_archive(std::string_view buf,
                                                                  const Serializable::Constructor &);
#define AKR_SER(...)                                                                                                   \
    void load(InputArchive &ar) { ar(__VA_ARGS__); }                                                                   \
    void save(OutputArchive &ar) const { ar(__VA_ARGS__); }
#define AKR_SER_CLASS(Self)                                                                                            \
    std::string_view type_str() const override { return Self; }

    // common structs
    template <typename T1, typename T2>
    struct Serde<std::pair<T1, T2>> {
        using T = std::pair<T1, T2>;
        static void serialize(OutputArchive &ar, const T &arg) { ar(arg.first, arg.second); }
        static void deserialize(InputArchive &ar, T &arg) { ar(arg.first, arg.second); }
    };

    template <typename U, int N>
    struct Serde<Vector<U, N>> {
        using T = Vector<U, N>;
        static void serialize(OutputArchive &ar, const T &arg) {
            for (int i = 0; i < N; i++) {
                ar(arg[i]);
            }
        }
        static void deserialize(InputArchive &ar, T &arg) {
            for (int i = 0; i < N; i++) {
                ar(arg[i]);
            }
        }
    };
    template <typename U, int N>
    struct Serde<Color<U, N>> {
        using T = Color<U, N>;
        static void serialize(OutputArchive &ar, const T &arg) {
            for (int i = 0; i < N; i++) {
                ar(arg[i]);
            }
        }
        static void deserialize(InputArchive &ar, T &arg) {
            for (int i = 0; i < N; i++) {
                ar(arg[i]);
            }
        }
    };

    template <typename U, int N>
    struct Serde<Mat<U, N>> {
        using T = Mat<U, N>;
        static void serialize(OutputArchive &ar, const T &arg) {
            for (int i = 0; i < N; i++) {
                ar(arg[i]);
            }
        }
        static void deserialize(InputArchive &ar, T &arg) {
            for (int i = 0; i < N; i++) {
                ar(arg[i]);
            }
        }
    };
    template <>
    struct Serde<TRSTransform> {
        using T = TRSTransform;
        static void serialize(OutputArchive &ar, const T &arg) { ar(arg.translation, arg.rotation, arg.scale); }
        static void deserialize(InputArchive &ar, T &arg) { ar(arg.translation, arg.rotation, arg.scale); }
    };
    template <>
    struct Serde<fs::path> {
        using T = fs::path;
        static void serialize(OutputArchive &ar, const T &arg) { ar(arg.string()); }
        static void deserialize(InputArchive &ar, T &arg) {
            std::string s;
            ar(s);
            arg = s;
        }
    };
    template <typename U, class A>
    struct Serde<std::vector<U, A>> {
        using T = std::vector<U, A>;
        static void serialize(OutputArchive &ar, const T &arg) {
            size_t count = arg.size();
            ar(count);
            for (auto &i : arg) {
                ar(i);
            }
        }
        static void deserialize(InputArchive &ar, T &arg) {
            size_t count;
            ar(count);
            arg.resize(count);
            for (auto &i : arg) {
                ar(i);
            }
        }
    };
} // namespace akari