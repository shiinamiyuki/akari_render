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


vec2 sqrt(vec2 v){
    return vec2(sqrt(v.x),sqrt(v.y));
}
vec3 sqrt(vec3 v){
    return vec3(sqrt(v.x),sqrt(v.y),sqrt(v.z));
}
vec4 sqrt(vec4 v){
    return vec4(sqrt(v.x),sqrt(v.y),sqrt(v.z), sqrt(v.w));
}
vec2 sin(vec2 v){
    return vec2(sin(v.x),sin(v.y));
}
vec3 sin(vec3 v){
    return vec3(sin(v.x),sin(v.y),sin(v.z));
}
vec4 sin(vec4 v){
    return vec4(sin(v.x),sin(v.y),sin(v.z), sin(v.w));
}
vec2 cos(vec2 v){
    return vec2(cos(v.x),cos(v.y));
}
vec3 cos(vec3 v){
    return vec3(cos(v.x),cos(v.y),cos(v.z));
}
vec4 cos(vec4 v){
    return vec4(cos(v.x),cos(v.y),cos(v.z), cos(v.w));
}
vec2 tan(vec2 v){
    return vec2(tan(v.x),tan(v.y));
}
vec3 tan(vec3 v){
    return vec3(tan(v.x),tan(v.y),tan(v.z));
}
vec4 tan(vec4 v){
    return vec4(tan(v.x),tan(v.y),tan(v.z), tan(v.w));
}

float dot(vec2 u, vec2 v){
    return v.x * u.x + v.y * u.y;
}
float dot(vec3 u, vec3 v){
    return v.x * u.x + v.y * u.y + v.z * u.z;
}
float length(vec3 v){
    return sqrt(dot(v,v));
}
vec3 normalize(vec3 v){
    return v / length(v);
}
vec3 cross(vec3 x, vec3 y){
    return vec3(x.y * y.z - y.y * x.z,
				x.z * y.x - y.z * x.x,
				x.x * y.y - y.x * x.y);
}
float distance(vec3 p, vec3 q){
    return length(p - q);
}
float min(float x, float y){
    if(x < y)
        return x;
    return y;
}
float max(float x, float y){
    if(x > y)
        return x;
    return y;
}
vec2 min(vec2 x, vec2 y){
    return vec2(min(x.x,y.x), min(x.y,y.y));
}
vec3 min(vec3 x, vec3 y){
    return vec3(min(x.x,y.x), min(x.y,y.y),min(x.z,y.z));
}
vec4 min(vec4 x, vec4 y){
    return vec4(min(x.x,y.x), min(x.y,y.y),min(x.z,y.z),min(x.w,y.w));
}

int min(int x, int y){
    if(x < y)
        return x;
    return y;
}
int max(int x, int y){
    if(x > y)
        return x;
    return y;
}
ivec2 min(ivec2 x, ivec2 y){
    return ivec2(min(x.x,y.x), min(x.y,y.y));
}
ivec3 min(ivec3 x, ivec3 y){
    return ivec3(min(x.x,y.x), min(x.y,y.y),min(x.z,y.z));
}
ivec4 min(ivec4 x, ivec4 y){
    return ivec4(min(x.x,y.x), min(x.y,y.y),min(x.z,y.z),min(x.w,y.w));
}
vec3 mix(vec3 x, vec3 y, float a){
    return x * (1.0 - a) + y * a;
}
vec2 mix(vec2 x, vec2 y, float a){
    return x * (1.0 - a) + y * a;
}
vec3 mix(vec3 x, vec3 y, vec3 a){
    return x * (1.0 - a) + y * a;
}
vec2 mix(vec2 x, vec2 y, vec2 a){
    return x * (1.0 - a) + y * a;
}

vec3 pow(vec3 b, vec3 e){
    return vec3(pow(b.x, e.x), pow(b.y,e.y), pow(b.z,e.z));
}
vec3 exp(vec3 b){
    return vec3(exp(b.x), exp(b.y), exp(b.z));
}
vec3 log(vec3 x){
    return vec3(log(x.x), log(x.y), log(x.z));
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
                j.emplace_back(_);
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