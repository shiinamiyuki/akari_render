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

#ifndef AKARIRENDER_SERIALIZE_HPP
#define AKARIRENDER_SERIALIZE_HPP
#include <Akari/Core/Platform.h>
#include <miyuki.serialize/serialize.hpp>
namespace Akari {
    using miyuki::serialize::Serializable;
    using miyuki::serialize::InputArchive;
    using miyuki::serialize::OutputArchive;
    using miyuki::serialize::Context;
    using TypeInfo = Serializable::Type;
#define AKR_SER(...) MYK_SER(__VA_ARGS__)
    class AKR_EXPORT ReviveContext : public miyuki::serialize::Context{
      public:
      private:
        Serializable::Type *getType(const std::string &s) override;
    };
}
#endif // AKARIRENDER_SERIALIZE_HPP
