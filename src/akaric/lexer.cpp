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

#include <akaric/lexer.h>
#include <cctype>
#include <set>
namespace akari::asl {
    static std::set<std::string> keywords = {"in",     "out",  "inout",  "uniform", "const", "if",       "else",
                                             "while",  "for",  "do",     "return",  "break", "continue", "struct",
                                             "buffer", "auto", "switch", "case",    "true",  "false"};
    static std::vector<std::set<std::string>> operators = {
        {"&&=", "||=", ">>=", "<<=", "..."},
        {"&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=", "|=", "&=", "^=", ">=", "<=", "!=", "==", "->", ">>",
         "<<"},
        {"+", "-", "*", "/", "%",  "&", "|", "^", "(", ")", "[", "]", ";",
         "{", "}", ",", "=", "\\", "<", ">", ".", ":", "?", "~", "!"}};
    static std::set<char> op_char = {'+', '-', '*', '/', '%', '^', '>', '<',  '!', '=', '(', ')', ';',
                                     '[', ']', '{', '}', '.', ':', ',', '\\', '&', '|', '~', '?'};
    class Lexer::Impl {
        std::string src;
        size_t pos = 0;
        TokenStream ts;
        SourceLocation loc;

      public:
        char at(size_t i) { return i < src.length() ? src[i] : 0; }
        char cur() { return at(pos); }
        char peek() { return at(pos + 1); }
        char peek2() { return at(pos + 2); }
        void advance() {
            if (cur() == '\n') {
                loc.line++;
                loc.col = 1;
            } else {
                if (cur() == '\t')
                    loc.col += 4;
                else
                    loc.col++;
            }
            pos++;
        }
        Token parse_symbol() {
            char p2 = peek2();
            char p = peek();
            std::string s1, s2, s3;
            s1 = cur();
            s2 = s1 + p;
            s3 = s2 + p2;
            // std::cout << s1 << " " << s2 << " " << s3 << " " << std::endl;
            // system("pause");
            if (operators[0].find(s3) != operators[0].end()) {
                auto t = Token{s3, symbol, loc};
                advance();
                advance();
                advance();
                return t;
            } else if (operators[1].find(s2) != operators[1].end()) {
                auto t = Token{s2, symbol, loc};
                advance();
                advance();
                return t;
            } else if (operators[2].find(s1) != operators[2].end()) {
                auto t = Token{s1, symbol, loc};
                advance();
                return t;
            } else {
                throw std::runtime_error(std::string("unable to parse ") + cur());
            }
        }
        static bool is_space(char c) { return c == ' ' || c == '\r' || c == '\t' || c == '\n'; }

        void skip_space() {
            skip_comment();
            while (is_space(cur())) {
                advance();
                skip_comment();
            }
        }
        int is_comment() {
            if (cur() == '#') // skip preprocessor instructions
                return 1;
            if (cur() == '/' && peek() == '/') {
                return 2;
            }
            if (cur() == '/' && peek() == '*') {
                return 3;
            }
            return 0;
        }
        void skip_comment() {
            int i = is_comment();
            if (!i)
                return;
            if (i == 1 || i == 2) {
                while (cur() != '\n')
                    advance();
            } else if (i == 3) {
                while (!(cur() == '*' && peek() == '/'))
                    advance();
                advance();
                advance();
            }
        }
        const TokenStream &operator()(const std::string &filename, std::string _s) {
            loc.filename = filename;
            src = std::move(_s);
            while (cur()) {
                skip_space();
                if (!cur()) {
                    break;
                }
                if (isalpha(cur()) || cur() == '_') {
                    std::string s;
                    while (isalnum(cur()) || cur() == '_') {
                        s += cur();
                        advance();
                    }
                    if (keywords.find(s) != keywords.end()) {
                        ts.emplace_back(Token{s, keyword, loc});
                    } else
                        ts.emplace_back(Token{s, identifier, loc});
                } else if (isdigit(cur())) {
                    std::string s;
                    TokenType type = int_literal;
                    while (isdigit(cur())) {
                        s += cur();
                        advance();
                    }
                    if (cur() == '.') {
                        type = float_literal;
                        s += cur();
                        advance();
                        while (isdigit(cur())) {
                            s += cur();
                            advance();
                        }
                    }
                    ts.emplace_back(Token{s, type, loc});
                } else if (op_char.count(cur())) {
                    ts.emplace_back(parse_symbol());
                } else {
                    fprintf(stderr, "stray token %s:%d:=%d\n", loc.filename.c_str(), loc.line, loc.col);
                    throw std::runtime_error("Lexer error");
                }
            }
            return ts;
        }
    };
    Lexer::Lexer() : impl(std::make_shared<Impl>()) {}
    const TokenStream &Lexer::operator()(const std::string &filename, const std::string &s) {
        return (*impl)(filename, s);
    }
} // namespace akari::asl