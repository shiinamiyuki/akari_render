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

#ifndef AKARIRENDER_ANY_HPP
#define AKARIRENDER_ANY_HPP

#include <cstring>
#include <memory>
#include <typeinfo>
namespace Akari {
    // An any class that holds **any** thing
    // Why not std::any?
    struct Any {
      private:
        struct Container {
            virtual void *get()  = 0;
            [[nodiscard]] virtual const char *name() const = 0;
        };
        std::shared_ptr<Container> container;
        template <typename T> void from(const T &value) {
            struct Con : Container {
                T v;
                Con(const T &v) : v(v) {}
                void *get() override { return &v; }

                [[nodiscard]] const char *name() const override { return typeid(T).name(); }
            };
            container = std::make_shared<Con>(value);
        }

      public:
        template <typename T> Any(const T &value) { from(value); }
        template <typename T> Any &operator=(const T &value) { from(value);return *this; }

        template <typename U> U *cast() {
            if (strcmp(typeid(U).name(), container->name()) == 0) {
                return reinterpret_cast<U *>(container.get());
            }
            return nullptr;
        }
        template <typename U> const U *cast() const{
            if (strcmp(typeid(U).name(), container->name()) == 0) {
                return reinterpret_cast<U *>(container.get());
            }
            return nullptr;
        }
        [[nodiscard]] const char * type_name()const{
            return container->name();
        }
    };
} // namespace Akari

#endif // AKARIRENDER_ANY_HPP
