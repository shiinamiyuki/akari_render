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

#include <akari/core/logger.h>
#include <akari/asl/parser.h>
#include <iostream>
using namespace akari::asl;
int main() {
    try{
    Parser parser(R"(
        struct point {float x; float y;}
        int sgn(float x){
            if(x> 0.0){
                return 1;
            }
            if(x < 0.0){
                return -1;
            }
            return 0;
        }
        vec3 main(vec3 v){
            return v;
        }


    )");
    auto ast = parser();
    nlohmann::json j;
    ast->dump_json(j);
    std::cout << j.dump(1) << std::endl;
    }catch(std::runtime_error & e){
        std::cerr << e.what() << std::endl;
    }
}
