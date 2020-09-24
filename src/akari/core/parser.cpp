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
#include <fstream>
#include <unordered_set>
#include <akari/core/parser.h>
#include <cctype>
namespace akari::sdl {
    Value::Value() : data(std::make_shared<Null>()) {}
    Value::Value(const array &a) : data(std::make_shared<Array>(a)) {}
    Value::Value(bool i) : data(std::make_shared<Boolean>(i)) {}
    Value::Value(int i) : data(std::make_shared<Number>(i)) {}
    Value::Value(double i) : data(std::make_shared<Number>(i)) {}
    Value::Value(float i) : data(std::make_shared<Number>(i)) {}
    Value::Value(std::string s) : data(std::make_shared<String>(std::move(s))) {}
    Value::array::const_iterator Value::begin() const {
        if (!is_array()) {
            return array::iterator();
        }
        return dyn_cast<Array>(data)->array().begin();
    }
    Value::array::const_iterator Value::end() const {
        if (!is_array()) {
            return array::iterator();
        }
        return dyn_cast<Array>(data)->array().end();
    }

    Value Value::at(int i) const {
        if (!is_array()) {
            return Value();
        }
        return dyn_cast<Array>(data)->at(i);
    }
    size_t Value::size() const {
        if (!is_array()) {
            return 0;
        }
        return dyn_cast<Array>(data)->array().size();
    }
    P<Object> Value::object() const {
        if (!is_object()) {
            return nullptr;
        }
        AKR_ASSERT(data);
        return dyn_cast<Object>(data);
    }
    P<Object> Parser::parse_object_creation(ParserContext &ctx, const std::string &type) {
        return do_parse_object_creation(ctx, type);
    }
    bool Parser::is_comment(ParserContext &ctx) {
        if (ctx.peek() == '/') {
            ctx.advance();
            ctx.expect('/');
            return true;
        }
        return false;
    }
    void Parser::skip_comment(ParserContext &ctx) {
        // printf("skip_comment\n");
        while (is_comment(ctx)) {
            while (ctx.peek() && ctx.peek() != '\n') {
                ctx.advance();
            }
            ctx.advance();
        }
        // printf("end skip_comment\n");
    }
    void Parser::skip(ParserContext &ctx) {
        // printf("skip\n");
        skip_comment(ctx);
        {
            while (isspace(ctx.peek())) {
                ctx.advance();
            }
            if (is_comment(ctx)) {
                while (ctx.peek() && ctx.peek() != '\n') {
                    ctx.advance();
                }
                skip(ctx);
            }
        }
        // printf("end skip\n");
    }
    char is_iden_header(char c) { return isalpha(c) || c == '_'; }
    Accessor Parser::parse_accessor(ParserContext &ctx) {
        ctx.expect('$');
        Accessor acc;

        acc.push_back(parse_identifier(ctx));
        while (ctx.peek() == '.') {
            ctx.advance();
            acc.push_back(parse_identifier(ctx));
        }
        return acc;
    }
    void Parser::expect_space(ParserContext &ctx) {
        if (!isspace(ctx.peek())) {
            ctx.report_error("space expected", ctx.loc);
        } else {
            ctx.advance();
        }
    }
    void Parser::expect_newline(ParserContext &ctx) {
        if (is_comment(ctx)) {
            while (ctx.peek() && ctx.peek() != '\n') {
                ctx.advance();
            }
            skip(ctx);
            return;
        }
        while (ctx.peek() == ' ' || ctx.peek() == '\t') {
            ctx.advance();
        }
        if (ctx.peek() == '\n') {
            ctx.advance();
            skip(ctx);
        } else {
            ctx.report_error("newline expected", ctx.loc);
        }
    }
    std::string Parser::parse_identifier(ParserContext &ctx) {
        std::string s;
        if (!is_iden_header(ctx.peek())) {
            ctx.report_error(fmt::format("illegal character in identifier: '{}'", ctx.peek()), ctx.loc);
        }
        while (isalnum(ctx.peek()) || ctx.peek() == '_') {
            s += ctx.extract();
        }
        return s;
    }
    Module Parser::parse_string(const std::string &src, const fs::path &filename, const std::string &module_name) {
        ParserContext ctx(src, filename.string());
        ctx.main.name = module_name;
        while (ctx.peek()) {
            skip(ctx);
            if (ctx.startswith("import")) {
                parse_import(ctx);
            } else if (ctx.startswith("let")) {
                parse_let(ctx);
            } else if (ctx.startswith("export")) {
                parse_export(ctx);
            } else {
                ctx.report_error(fmt::format("stray token {}", ctx.peek()), ctx.loc);
            }
        }
        return ctx.main;
    }
    Module Parser::parse_file(const fs::path &_path, const std::string &module_name) {
        auto path = fs::absolute(_path);
        auto parent_path = path.parent_path();
        CurrentPathGuard _;
        fs::current_path(parent_path);
        std::ifstream in(path);
        std::string src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return parse_string(src, path, module_name);
    }
    void Parser::parse_import(ParserContext &ctx) {
        ctx.expect("import");
        expect_space(ctx);
        auto path = fs::path(parse_string(ctx));
        if (!fs::exists(path)) {
            ctx.report_error(fmt::format("module \"{}\" not found", path.string()), ctx.loc);
        }
        expect_space(ctx);
        ctx.expect("as");
        expect_space(ctx);
        skip(ctx);
        auto alias = parse_identifier(ctx);
        if (ctx.cur_mod()->submodules.find(alias) != ctx.cur_mod()->submodules.end()) {
            ctx.report_error(fmt::format("{} is already defined", alias), ctx.loc);
        }
        expect_newline(ctx);
        ctx.cur_mod()->submodules[alias] = parse_file(path, alias);
    }
    void Parser::parse_export(ParserContext &ctx) {
        ctx.expect("export");
        expect_space(ctx);
        auto var = parse_identifier(ctx);
        if (ctx.cur_mod()->exports.find(var) != ctx.cur_mod()->exports.end()) {
            ctx.report_error(fmt::format("{} is already defined", var), ctx.loc);
        }
        skip(ctx);
        ctx.expect('=');
        skip(ctx);
        auto val = parse(ctx);
        ctx.cur_mod()->exports[var] = val;
        expect_newline(ctx);
    }
    void Parser::parse_let(ParserContext &ctx) {
        ctx.expect("let");
        expect_space(ctx);
        auto var = parse_identifier(ctx);
        if (ctx.cur_mod()->locals.find(var) != ctx.cur_mod()->locals.end()) {
            ctx.report_error(fmt::format("{} is already defined", var), ctx.loc);
        }
        skip(ctx);
        ctx.expect('=');
        skip(ctx);
        auto val = parse(ctx);
        ctx.cur_mod()->locals[var] = val;
        expect_newline(ctx);
    }
    Value Parser::parse(ParserContext &ctx) {
        auto c = ctx.peek();
        if (c == '$') {
            auto acc = parse_accessor(ctx);
            auto mod = ctx.cur_mod();
            auto it = acc.begin();
            for (; it + 1 != acc.end(); it++) {
                auto m = mod->submodules.find(*it);
                if (m == mod->submodules.end()) {
                    ctx.report_error(fmt::format("module {} has no submodule named {}", mod->name, *it), ctx.loc);
                } else {
                    mod = &m->second;
                }
            }
            auto v = mod->exports.find(*it);
            if (v == mod->exports.end()) {
                ctx.report_error(fmt::format("module {} has no exported variable named {}", mod->name, *it), ctx.loc);
            } else {
                return v->second;
            }
        } else if (c == '[') {
            return parse_array(ctx);
        } else if (is_iden_header(c)) {
            if (ctx.startswith("true")) {
                ctx.pos += 4;
                return Value(true);
            } else if (ctx.startswith("false")) {
                ctx.pos += 5;
                return Value(false);
            }
            return parse_object(ctx);
        } else if (c == '"') {
            return Value(parse_string(ctx));
        } else if (c == '-' || isdigit(c)) {
            return Value(parse_float(ctx));
        } else {
            ctx.report_error(fmt::format("stray token {}", c), ctx.loc);
        }
    }
    Value Parser::parse_object(ParserContext &ctx) {
        auto type = parse_identifier(ctx);
        auto object = parse_object_creation(ctx, type);
        AKR_ASSERT_THROW(object);
        skip(ctx);
        ctx.expect('{');
        skip(ctx);
        std::unordered_set<std::string> fields;
        while (ctx.peek() && ctx.peek() != '}') {
            auto fieldname = parse_identifier(ctx);
            if(fields.find(fieldname) != fields.end()){
                ctx.report_error(fmt::format("field {} redefined", fieldname), ctx.loc);
            }
            skip(ctx);
            ctx.expect(':');
            skip(ctx);
            auto val = parse(ctx);
            skip(ctx);
            try {
                object->object_field(*this, ctx, fieldname, val);
            } catch (ParseError &e) {
                throw e;
            } catch (std::exception &e) {
                ctx.report_error(fmt::format("exception caught: {}", e.what()), ctx.loc);
            }
            if (ctx.peek() != '}') {
                ctx.expect(',');
            }
            skip(ctx);
        }
        ctx.expect('}');
        return Value(object);
    }
    Value Parser::parse_array(ParserContext &ctx) {
        ctx.expect('[');
        skip(ctx);
        Value::array a;
        while (ctx.peek() && ctx.peek() != ']') {
            a.push_back(parse(ctx));
            skip(ctx);
            if (ctx.peek() != ']') {
                ctx.expect(',');
            }
            skip(ctx);
        }
        ctx.expect(']');
        return Value(std::move(a));
    }
    std::string Parser::parse_string(ParserContext &ctx) {
        ctx.expect('"');
        std::string s;
        while (ctx.peek() && ctx.peek() != '"') {
            if (ctx.peek() == '\\') {
                ctx.advance();
                auto c = ctx.extract();
                if (c == '\\') {
                    s += c;
                } else if (c == 'n') {
                    s += '\n';
                } else if (c == '"') {
                    s += '"';
                } else {
                    ctx.report_error("illegal espace sequence", ctx.loc);
                }
            } else {
                s += ctx.extract();
            }
        }
        ctx.expect('"');
        return s;
    }
    double Parser::parse_float(ParserContext &ctx) {
        auto peek = ctx.peek();
        if (peek == '-') {
            ctx.advance();
            return -parse_float(ctx);
        }
        if (!isdigit(ctx.peek())) {
            ctx.report_error("digit expected", ctx.loc);
        }
        int n = 0;
        while (isdigit(ctx.peek())) {
            n = n * 10 + ctx.peek() - '0';
            ctx.advance();
        }
        int frac = 0;
        double p = 1.0;
        if (ctx.peek() == '.') {
            ctx.advance();
            while (isdigit(ctx.peek())) {
                frac = frac * 10 + ctx.peek() - '0';
                p *= 10;
                ctx.advance();
            }
        }
        return n + frac / p;
    }
    int Parser::parse_int(ParserContext &ctx) {
        auto peek = ctx.peek();
        if (peek == '-') {
            ctx.advance();
            return -parse_int(ctx);
        } else if (peek == '0') {
            ctx.advance();
            ctx.expect('x');
            int n = 0;
            while (isdigit(ctx.peek()) || (ctx.peek() >= 'a' && ctx.peek() <= 'f') ||
                   (ctx.peek() >= 'A' && ctx.peek() <= 'F')) {
                if (isdigit(ctx.peek())) {
                    n = n * 16 + ctx.peek() - '0';
                } else if (ctx.peek() >= 'a' && ctx.peek() <= 'f') {
                    n = n * 16 + ctx.peek() - 'a';
                } else {
                    n = n * 16 + ctx.peek() - 'A';
                }
                ctx.advance();
            }
            return n;
        } else {
            AKR_ASSERT(isdigit(ctx.peek()));
            int n = 0;
            while (isdigit(ctx.peek())) {
                n = n * 10 + ctx.peek() - '0';
                ctx.advance();
            }
            return n;
        }
    }
} // namespace akari::sdl