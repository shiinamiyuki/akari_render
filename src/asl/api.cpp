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
    const char * asl_stdlib = R"(
float tan(float x){
    return sin(x)/cos(x);
}


vec2 sqrtf2(vec2 v){
    return vec2(sqrt(v.x),sqrt(v.y));
}
vec3 sqrtf3(vec3 v){
    return vec3(sqrt(v.x),sqrt(v.y),sqrt(v.z));
}
vec4 sqrtf4(vec4 v){
    return vec4(sqrt(v.x),sqrt(v.y),sqrt(v.z), sqrt(v.w));
}
vec2 sinf2(vec2 v){
    return vec2(sin(v.x),sin(v.y));
}
vec3 sinf3(vec3 v){
    return vec3(sin(v.x),sin(v.y),sin(v.z));
}
vec4 sinf4(vec4 v){
    return vec4(sin(v.x),sin(v.y),sin(v.z), sin(v.w));
}
vec2 cosf2(vec2 v){
    return vec2(cos(v.x),cos(v.y));
}
vec3 cosf3(vec3 v){
    return vec3(cos(v.x),cos(v.y),cos(v.z));
}
vec4 cosf4(vec4 v){
    return vec4(cos(v.x),cos(v.y),cos(v.z), cos(v.w));
}
vec2 tanf2(vec2 v){
    return vec2(tan(v.x),tan(v.y));
}
vec3 tanf3(vec3 v){
    return vec3(tan(v.x),tan(v.y),tan(v.z));
}
vec4 tanf4(vec4 v){
    return vec4(tan(v.x),tan(v.y),tan(v.z), tan(v.w));
}

float dotf2(vec2 u, vec2 v){
    return v.x * u.x + v.y * u.y;
}
float dotf3(vec2 u, vec2 v){
    return v.x * u.x + v.y * u.y + v.z * u.z;
}
float lengthf3(vec3 v){
    return sqrt(dotf3(v,v));
}
vec3 normalizef3(vec3 v){
    return v / lengthf3(v);
}
vec3 cross(vec3 x, vec3 y){
    return vec3(x.y * y.z - y.y * x.z,
				x.z * y.x - y.z * x.x,
				x.x * y.y - y.x * x.y);
}
float distancef3(vec3 p, vec3 q){
    return lengthf3(p - q);
}

    )";
    Expected<std::shared_ptr<Program>> compile(const std::vector<TranslationUnit> &units,
                                               CompileOptions opt) {
        try {
            ParsedProgram prog;
            using namespace nlohmann;
            json j = json::array();
            Parser parser;
            prog.modules.emplace_back(parser("stdlib.asl", asl_stdlib));
            {
                json _;
                prog.modules.back()->dump_json(_);
            }
            for (auto &unit : units) {
                prog.modules.emplace_back(parser(unit.filename, unit.source));
                json _;
                prog.modules.back()->dump_json(_);
                // std::cout << _.dump(1) <
                j.emplace_back(_);
            }
            // std::cout << j.dump(1) << std::endl;
            auto backend = create_llvm_backend();
            return backend->compile(prog, opt);
        } catch (std::runtime_error &e) {
            return Error(e.what());
        }
    }
} // namespace akari::asl