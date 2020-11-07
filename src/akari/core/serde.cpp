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
#include <unordered_set>
#include <akari/core/serde.h>
#include <json.hpp>
// #include <cereal/archives/binary.hpp>
// #include <cereal/archives/json.hpp>
// #include <cereal/archives/xml.hpp>

namespace akari {
    static constexpr char OBJECT_BEGIN = 0x78;
    static constexpr char OBJECT_END = 0x56;
    using nlohmann::json;
    class OutputArchiveBase : public OutputArchive {
      public:
        std::unordered_set<std::shared_ptr<Serializable>> visited;
        void save_ptr(const std::shared_ptr<Serializable> &p) override {
            begin_object();
            if (!p) {
                SerdeObjectRecord rec;
                rec.type_str = "";
                rec.is_null = true;
                do_save(rec);
            } else {
                SerdeObjectRecord rec;
                rec.type_str = p->type_str();
                rec.is_null = false;
                rec.uid = uint64_t(p.get());
                if (visited.find(p) == visited.end()) {
                    visited.insert(p);
                    rec.valid = true;
                    do_save(rec);
                    p->save(*this);
                } else {
                    rec.valid = false;
                    do_save(rec);
                }
            }
            end_object();
        }
    };
    class JsonOutputArchive : public OutputArchiveBase {
        json &j;
        std::vector<int> st;
        std::vector<json *> j_st;

      public:
        JsonOutputArchive(json &j_) : j(j_) {
            j_st.push_back(&j);
            j = json::object();
            st.push_back(0);
        }
        template <class T>
        auto save_helper(const T &v) {
            (*j_st.back())[std::to_string(st.back())] = v;
            st.back()++;
        }
        void save_i1(bool v) override { return save_helper(v); }
        void save_i8(int8_t v) override { return save_helper(v); }
        void save_i32(int32_t v) override { return save_helper(v); }
        void save_i64(int64_t v) override { return save_helper(v); }
        void save_u8(uint8_t v) override { return save_helper(v); }
        void save_u32(uint32_t v) override { return save_helper(v); }
        void save_u64(uint64_t v) override { return save_helper(v); }
        void save_f32(float v) override { return save_helper(v); }
        void save_f64(double v) override { return save_helper(v); }
        void save_str(const std::string &s) { save_helper(s); }
        void save_bytes(const char *buf, size_t bytes) override {
            (*j_st.back())[std::to_string(st.back())] = json::array();
            auto &arr = (*j_st.back())[std::to_string(st.back())];
            for (size_t i = 0; i < bytes; i++) {
                arr[i] = (int)buf[i];
            }
        }
        void begin_object() {
            json o = json::object();
            auto id = std::to_string(st.back());
            (*j_st.back())[id] = o;
            auto &next = (*j_st.back())[id];
            st.back()++;
            st.push_back(0);
            j_st.push_back(&next);
        }
        void end_object() {
            j_st.pop_back();
            st.pop_back();
        }
    };
    AKR_EXPORT std::unique_ptr<OutputArchive> create_output_archive(nlohmann::json &j) {
        return std::make_unique<JsonOutputArchive>(j);
    }
    class BinaryOutputArchive : public OutputArchiveBase {
        std::unique_ptr<Stream> stream;

      public:
        BinaryOutputArchive(std::unique_ptr<Stream> stream) : stream(std::move(stream)) {}
        template <class T>
        auto save_helper(const T &v) {
            return save_bytes((const char *)&v, sizeof(T));
        }
        void save_i1(bool v) override { return save_helper(v); }
        void save_i8(int8_t v) override { return save_helper(v); }
        void save_i32(int32_t v) override { return save_helper(v); }
        void save_i64(int64_t v) override { return save_helper(v); }
        void save_u8(uint8_t v) override { return save_helper(v); }
        void save_u32(uint32_t v) override { return save_helper(v); }
        void save_u64(uint64_t v) override { return save_helper(v); }
        void save_f32(float v) override { return save_helper(v); }
        void save_f64(double v) override { return save_helper(v); }
        void save_bytes(const char *buf, size_t bytes) override { stream->write(buf, bytes); }
        void save_str(const std::string &s) {
            do_save(s.length());
            save_bytes(s.data(), s.size());
        }

        void begin_object() { do_save(OBJECT_BEGIN); }
        void end_object() { do_save(OBJECT_END); }
    };
    AKR_EXPORT std::unique_ptr<OutputArchive> create_output_archive(std::unique_ptr<Stream> stream) {
        return std::make_unique<BinaryOutputArchive>(std::move(stream));
    }
    class InputArchiveBase : public InputArchive {
      public:
        InputArchiveBase(const Serializable::Constructor &ctor) : ctor(ctor) {}
        Serializable::Constructor ctor;
        std::unordered_map<uint64_t, std::shared_ptr<Serializable>> visited;
        std::shared_ptr<Serializable> load() override {
            begin_object();
            SerdeObjectRecord rec;
            do_load(rec);

            std::shared_ptr<Serializable> object;
            if (!rec.is_null) {
                bool is_vis = visited.find(rec.uid) != visited.end();
                if (!is_vis) {
                    object.reset(ctor(rec.type_str));
                    visited[rec.uid] = object;
                } else {
                    object = visited.at(rec.uid);
                }
                // valid is written only once
                // this guarantees once initialization
                if (rec.valid) {
                    object->load(*this);
                }
            }
            end_object();
            return object;
        }
    };

    // class JsonInputArchive : public InputArchiveBase {
    //     const json &j;

    //   public:
    //     JsonInputArchive(const json &j,const Serializable::Constructor &ctor) : j(j),ctor(ctor) {}
    // };
    class BinaryInputArchive : public InputArchiveBase {
        std::unique_ptr<Stream> stream;

      public:
        template <class T>
        auto load_helper(T &v) {
            return load_bytes((char *)&v, sizeof(T));
        }
        bool load_i1(bool &v) override { return load_helper(v); }
        bool load_i8(int8_t &v) override { return load_helper(v); }
        bool load_i32(int32_t &v) override { return load_helper(v); }
        bool load_i64(int64_t &v) override { return load_helper(v); }
        bool load_u8(uint8_t &v) override { return load_helper(v); }
        bool load_u32(uint32_t &v) override { return load_helper(v); }
        bool load_u64(uint64_t &v) override { return load_helper(v); }
        bool load_f32(float &v) override { return load_helper(v); }
        bool load_f64(double &v) override { return load_helper(v); }
        bool load_str(std::string &s) override {
            size_t len;
            if (!load_u64(len))
                return false;
            s.resize(len);
            std::vector<char> buf(len);
            if (!load_bytes(buf.data(), len))
                return false;
            s.copy(buf.data(), buf.size());
            return true;
        }
        BinaryInputArchive(std::unique_ptr<Stream> stream, const Serializable::Constructor &ctor)
            : InputArchiveBase(ctor), stream(std::move(stream)) {}

        bool load_bytes(char *buf, size_t bytes) override { return bytes == stream->read(buf, bytes); }
        void begin_object() {
            char c = 0;
            stream->read(&c, 1);
            if (c != OBJECT_BEGIN) {
                AKR_PANIC("expected object begin tag");
            }
        }
        void end_object() {
            char c = 0;
            stream->read(&c, 1);
            if (c != OBJECT_END) {
                AKR_PANIC("expected object end tag");
            }
        }
    };
    AKR_EXPORT std::unique_ptr<InputArchive> create_input_archive(std::string_view buf,
                                                                  const Serializable::Constructor &ctor) {
        auto stream = std::make_unique<ByteStream>(buf, Stream::Mode::Read);
        return std::make_unique<BinaryInputArchive>(std::move(stream), ctor);
    }
} // namespace akari