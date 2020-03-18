// MIT License
//
// Copyright (c) 2019 椎名深雪
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
#include <Akari/Core/Component.h>
#include <Akari/Core/Film.h>
#include <Akari/Core/Logger.h>
#include <Akari/Core/Math.h>
#include <Akari/Core/Parallel.h>
#include <Akari/Core/Plugin.h>
#include <Akari/Render/Geometry.hpp>
#include <Akari/Render/Reflection.hpp>
#include <atomic>
#include <iostream>
int main() {
    using namespace Akari;
    auto manager = Akari::GetPluginManager();
    auto foo = manager->LoadPlugin("Foo");
        Debug("{}",foo->GetTypeInfo()->name());
    //    std::atomic_uint32_t sum(0);
    //    Akari::ParallelFor(10000, [&](int i){
    //        sum += i;
    //    },256);
    //    std::cout << sum << std::endl;
    //    Akari::ParallelFor(10000, [&](int i){
    //      sum += i;
    //    },256);
    //    std::cout << sum << std::endl;
    //    Akari::ParallelFor(10000, [&](int i){
    //      sum += i;
    //    },256);
    //    std::cout << sum << std::endl;
    Film film(ivec2(1000, 1000));
    film.WriteImage("out.png");
}