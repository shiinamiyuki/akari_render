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
#include <memory>
#include <type_traits>
namespace akari::shading {
    class Base : public std::enable_shared_from_this<Base> {
      public:
        template <typename T>
        bool isa() const {
            return dynamic_cast<T *>(this) != nullptr;
        }
        template <typename T, typename = std::enable_if_t<std::is_const_v<T>>>
        std::shared_ptr<T> cast() const {
            return std::dynamic_pointer_cast<T>(shared_from_this());
        }
        template <typename T, typename = std::enable_if_t<!std::is_const_v<T>>>
        std::shared_ptr<T> cast() {
            return std::dynamic_pointer_cast<T>(shared_from_this());
        }
    };
    class Type : public Base {};
    class PrimitiveType : public Type {};
} // namespace akari::shading