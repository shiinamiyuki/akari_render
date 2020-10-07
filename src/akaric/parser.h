
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
#include <akaric/ast.h>
namespace akari::asl {

    class Parser {
        class Impl;

      public:
        struct ParseRecord {
            std::string filename, src;
            TokenStream ts;
            ast::TopLevel tree;
            std::unordered_set<std::string> typenames;
            bool parse_typenames = false;
            bool parse_body = false;
        };

      private:
        std::unordered_map<std::string, ParseRecord> parsed_modules;
        std::vector<std::string> module_search_path;
        const std::unordered_set<std::string> &resolve_typenames(const std::string &full_path);
        fs::path resolve_module(const fs::path &current_file, const std::string &module);
        void init_parse_record(const std::string &full_path);
        ast::TopLevel parse(const std::string &full_path);
        

      public:
        Parser();
        ast::TopLevel operator()(const std::string &filename);
    };

    struct OperatorPrecedence {
        std::unordered_map<std::string, int> opPrec;
        std::unordered_map<std::string, int> opAssoc;
        int ternaryPrec;
        OperatorPrecedence() {
            int prec = 0;
            opPrec["?"] = ternaryPrec = prec;
            prec++;
            opPrec["||"] = prec;
            prec++;
            opPrec["&&"] = prec;
            prec++;
            opPrec["|"] = prec;
            prec++;
            opPrec["^"] = prec;
            opPrec["&"] = prec;
            prec++;
            opPrec["=="] = prec;
            opPrec["!="] = prec;
            prec++;
            opPrec[">="] = prec;
            opPrec["<="] = prec;
            opPrec[">"] = prec;
            opPrec["<"] = prec;

            prec++;
            opPrec[">>"] = prec;
            opPrec["<<"] = prec;
            prec++;
            opPrec["+"] = prec;
            opPrec["-"] = prec;
            prec++;
            opPrec["*"] = prec;
            opPrec["/"] = prec;
            opPrec["%"] = prec;
            prec++;
            opPrec["."] = prec;
            opAssoc = {{"+", 1},  {"-", 1}, {"*", 1}, {"/", 1},  {"!=", 1}, {"==", 1}, {">", 1}, {">=", 1},
                       {"<=", 1}, {"<", 1}, {"%", 1}, {"&&", 1}, {"&", 1},  {"||", 1}, {"|", 1}};
        }
    };
} // namespace akari::asl