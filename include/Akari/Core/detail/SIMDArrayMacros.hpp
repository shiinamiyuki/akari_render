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

#ifndef AKARIRENDER_SIMDARRAYMACROS_HPP
#define AKARIRENDER_SIMDARRAYMACROS_HPP


namespace Akari {
#define _AKR_VECTORIZE_CALL(Ty) \
template<size_t N>struct Akari::unique_instance_context<Ty*,N>: Akari::unique_instance_context_base<Ty*, N>{\
    using namespace Akari;\

#define _AKR_VECTROZIE_METHOD(method) \
    template<typename...Args>inline auto method(Args&...args)const{\
        using RetT = std::invoke_t<decltype(Ty::method), Args...>;\
        RetT ret;\
        for(int i =0 ;i< this->n_unique; i++){\
            auto tmp = this->_unique_instances[i]->method(std::forward<Args>(args)..., this->active[i]);\
            ret = select(this->active[i], tmp, ret);\
        } \
        return ret;\
    }
#define _AKR_VECTORIZE_CALL_DONE() }
}

#endif // AKARIRENDER_SIMDARRAYMACROS_HPP
