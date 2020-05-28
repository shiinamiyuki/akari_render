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

#ifndef AKARIRENDER_PLATFORM_H
#define AKARIRENDER_PLATFORM_H
#include <cstdint>
namespace akari {
#ifdef _MSC_VER
    #define AKR_EXPORT __declspec(dllexport)
    #pragma warning(disable : 4275)
    #pragma warning(disable : 4267)
    #pragma warning(disable : 4251) // 'field' : class 'A' needs to have dll-interface to be used by clients of class 'B'
    #pragma warning(disable : 4800) // 'type' : forcing value to bool 'true' or 'false' (performance warning)
    #pragma warning(disable : 4996) // Secure SCL warnings
    #define AKR_FORCEINLINE __forceinline
#else
    #if defined _WIN32 || defined __CYGWIN__
        #ifdef __GNUC__
          #define AKR_EXPORT __attribute__ ((dllexport))
          
        #else
          #define AKR_EXPORT __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
        #endif
    #else

        #define AKR_EXPORT __attribute__ ((visibility ("default")))

    #endif
    #define AKR_FORCEINLINE   inline __attribute__((always_inline))
#endif
} // namespace akari
#endif // AKARIRENDER_PLATFORM_H
