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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once
#include <akari/core/platform.h>
#include <akari/core/akari.h>
#include <akari/core/logger.h>
#include <optional>
namespace akari::sdl {
    template <typename T>
    using P = std::shared_ptr<T>;
    class AKR_EXPORT ValueBase : public std::enable_shared_from_this<ValueBase> {
      public:
        virtual bool is_object() const { return false; }
        virtual bool is_array() const { return false; }
        virtual bool is_number() const { return false; }
        virtual bool is_string() const { return false; }
        virtual bool is_null() const { return false; }
        virtual bool is_boolean() const { return false; }
    };
    struct ParserContext;
    class Vector;
    class Object;
    class Null;
    class Boolean;
    class Number;
    class String;

    struct ParseError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct AKR_EXPORT Value {
        using array = std::vector<Value>;
        using dict = std::unordered_map<std::string, Value>;
        Value(const P<ValueBase> &v) : data(v) {}
        Value();
        Value(const array &a);
        explicit Value(bool i);
        explicit Value(int i);
        explicit Value(double i);
        explicit Value(float i);
        explicit Value(std::string s);
        bool is_object() const { return data->is_object(); }
        bool is_array() const { return data->is_array(); }
        bool is_number() const { return data->is_number(); }
        bool is_string() const { return data->is_string(); }
        bool is_null() const { return data->is_null(); }
        bool is_boolean() const { return data->is_boolean(); }
        array::const_iterator begin() const;
        array::const_iterator end() const;
        size_t size() const;
        Value at(int i) const;
        P<Object> object() const;
        template <typename T>
        inline std::optional<T> get() const;

      private:
        P<ValueBase> data;
    };

    class AKR_EXPORT Null : public ValueBase {
      public:
        bool is_null() const { return true; }
    };
    class AKR_EXPORT Number : public ValueBase {
      public:
        double number;
        Number(double n = 0.0) : number(n) {}
        bool is_number() const { return true; }
    };
    class AKR_EXPORT Boolean : public ValueBase {
      public:
        bool v;
        Boolean(bool v = false) : v(v) {}
        bool is_boolean() const { return true; }
    };
    class AKR_EXPORT String : public ValueBase {
        friend struct ParseContext;

      public:
        std::string s;
        String(std::string s = "") : s(s) {}
        bool is_string() const { return true; }
    };
    struct ParserContext;
    class Parser;
    class AKR_EXPORT Object : public ValueBase {
        friend struct ParserContext;

      public:
        virtual ~Object() = default;
        bool is_object() const override { return true; }
        virtual void object_field(Parser &parser, ParserContext &ctx, const std::string &field, const Value &value) = 0;
    };
    class AKR_EXPORT Vector : public ValueBase {
        std::vector<Value> arr;
        friend struct ParseContext;

      public:
        Vector(std::vector<Value> arr = {}) : arr(arr) {}
        const std::vector<Value> &array() const { return arr; }
        bool is_array() const { return true; }
        Value at(int i) {
            if (i < (int)arr.size() && i >= 0) {
                return arr[i];
            }
            return Value();
        }
    };
    template <typename T>
    inline std::optional<T> Value::get() const {
        if (is_null())
            return std::nullopt;
        if constexpr (std::is_same_v<T, bool>) {
            if (is_boolean()) {
                return dyn_cast<Boolean>(data)->v;
            }
            return std::nullopt;
        } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
            if (!is_number()) {
                return std::nullopt;
            }
            auto number = dyn_cast<Number>(data);
            return static_cast<T>(number->number);
        } else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) {
            if (!is_string()) {
                return std::nullopt;
            }
            auto s = dyn_cast<String>(data);
            return T(s->s);
        } else {
            static_assert("T is not supported");
        }
    }
    struct SourceLoc {
        int line = 1, col = 1;
        P<std::string> filename;
    };
    using Accessor = std::vector<std::string>;
    struct Module {
        std::string name;
        std::unordered_map<std::string, P<Module>> submodules;
        std::unordered_map<std::string, Value> exports;
        std::unordered_map<std::string, Value> locals;
    };
    struct AKR_EXPORT ParserContext {
        std::vector<P<Module>> mod_stack;
        P<Module> main;

        SourceLoc loc;
        const std::string &source;
        size_t pos = 0;
        ParserContext(const std::string &source, const std::string &filename) : source(source) {
            loc.filename = std::make_shared<std::string>(filename);
            main = std::make_shared<Module>();
            mod_stack.push_back(main);
        }
        char peek() {
            if (pos < source.length()) {
                return source[pos];
            }
            return 0;
        }
        bool startswith(const std::string &s) {
            if (pos + s.length() > source.length()) {
                return false;
            }
            for (size_t i = 0; i < s.length(); i++) {
                if (source[pos + i] != s[i])
                    return false;
            }
            return true;
        }
        void putback() {
            AKR_ASSERT(pos > 0);
            pos--;
        }
        char extract() {
            if (pos >= source.length()) {
                report_error("unexpected EOF", loc);
            }
            char c = peek();
            advance();
            return c;
        }
        void expect(char c) {
            auto e = extract();
            if (e != c) {
                report_error(fmt::format("'{}' expected but found '{}'", c, e), loc);
            }
        }
        void expect(const std::string &s) {
            for (auto c : s) {
                auto e = extract();
                if (e != c) {
                    report_error(fmt::format("'{}' expected", s), loc);
                }
            }
        }
        P<Module> cur_mod() { return mod_stack.back(); }
        [[noreturn]] void report_error(const std::string &message, SourceLoc loc_) {
            auto path = fs::path(*loc_.filename);
            path = fs::absolute(path);
            error("parse error: {}:{}:{} {} ", path.string(), loc_.line, loc_.col, message);
            throw ParseError("Parse Error");
        }
        void advance() {
            // putchar(source[pos]);
            if (source[pos] == '\n') {
                loc.line++;
                loc.col = 1;
            } else {
                loc.col++;
            }
            pos++;
        }
    };

    class AKR_EXPORT Parser {
      protected:
        virtual P<Object> do_parse_object_creation(ParserContext &ctx, const std::string &type) = 0;
        // virtual void do_parse_import(ParserContext &ctx, const std::string &path) = 0;

      public:
        std::string parse_identifier(ParserContext &ctx);
        Accessor parse_accessor(ParserContext &ctx);
        void skip(ParserContext &ctx);
        void expect_space(ParserContext &ctx);
        void expect_newline(ParserContext &ctx);
        Value parse(ParserContext &ctx);
        Value parse_array(ParserContext &ctx);
        Value parse_object(ParserContext &ctx);
        int parse_int(ParserContext &ctx);
        double parse_float(ParserContext &ctx);
        std::string parse_string(ParserContext &ctx);
        P<Object> parse_object_creation(ParserContext &ctx, const std::string &type);
        void parse_object_field(ParserContext &ctx, P<Object> object, const std::string &field, const Value &value);
        void parse_import(ParserContext &ctx);
        void parse_let(ParserContext &ctx);
        void parse_export(ParserContext &ctx);
        P<Module> parse_file(const fs::path &, const std::string &module_name = "");
        P<Module> parse_string(const std::string &src, const fs::path &filename = "",
                               const std::string &module_name = "");
        void skip_comment(ParserContext &ctx);
        bool is_comment(ParserContext &ctx);
    };
} // namespace akari::sdl