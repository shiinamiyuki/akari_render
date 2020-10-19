
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
    struct TranslationUnit {
        std::string filename;
        ast::TopLevel tree;
    };
    class Parser {
        class Impl;

      public:

      private:
        std::unordered_set<std::string> type_parameters;
        ast::TopLevel parse(const std::string &full_path);
        std::vector<TranslationUnit> headers;
        std::unordered_map<std::string, std::unordered_set<std::string>> header_types;
        void handle_include(const std::string & path, Impl*);
      public:
        Parser();
        std::vector<TranslationUnit> operator()(const std::vector<std::string> &filenames);
        void add_type_parameter(const std::string &type) { type_parameters.insert(type); }
    };

    struct OperatorPrecedence {
        std::unordered_map<std::string, int> opPrec;
        std::unordered_map<std::string, int> opAssoc;
        int ternaryPrec;
        OperatorPrecedence() {
            int prec = 0;
            // opPrec["?"] = ternaryPrec = prec;
            // prec++;
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