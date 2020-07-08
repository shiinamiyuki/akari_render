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

#include <iostream>
#include <json.hpp>
#include <akari/asl/asl.h>
#include <akari/asl/parser.h>
#include <akari/asl/backend.h>

namespace akari::asl {
    Expected<std::shared_ptr<Program>> compile(const std::vector<TranslationUnit> &units,
                                               CompileOptions opt) {
        try {
            ParsedProgram prog;
            using namespace nlohmann;
            json j = json::array();
            Parser parser;
            for (auto &unit : units) {
                prog.modules.emplace_back(parser(unit.filename, unit.source));
                json _;
                prog.modules.back()->dump_json(_);
                // std::cout << _.dump(1) <
                j.emplace_back(_);
            }
            std::cout << j.dump(1) << std::endl;
            auto backend = create_llvm_backend();
            return backend->compile(prog, opt);
        } catch (std::runtime_error &e) {
            return Error(e.what());
        }
    }
} // namespace akari::asl