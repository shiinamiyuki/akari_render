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
#include <akari/compute/lang.hpp>
#include <akari/compute/transform.hpp>
#include <akari/compute/debug.h>
using namespace akari::compute;

int main() {
    using namespace akari::compute;
    using namespace lang;
    Function top([]()->Var{
        Var x(1);
        Var y = x + 1;
        y += 2;
        Var z = 3 * y + x + x;
        Function add([](Var a, Var b)->Var{
            return a + b;
        });
        return add(z, z);
    });
    
    // Function add([](Var x, Var y) -> Var { return x + y; });
    // Var f = add;
    // std::cout << f.get_expr()->dump() << std::endl;
    auto expr = top.get_expr();
    auto anf = transform::convert_to_anf()->transform(expr);
    std::cout << debug::to_text(anf) << std::endl;
}
