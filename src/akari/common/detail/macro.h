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

#define AKR_ALIAS(Type, ...) using A##Type = Type<__VA_ARGS__>;

// this macro is directly copied from https://github.com/mitsuba-renderer/enoki/blob/master/include/enoki/array_macro.h
#define AKR_EVAL_0(...) __VA_ARGS__
#define AKR_EVAL_1(...) AKR_EVAL_0(AKR_EVAL_0(AKR_EVAL_0(__VA_ARGS__)))
#define AKR_EVAL_2(...) AKR_EVAL_1(AKR_EVAL_1(AKR_EVAL_1(__VA_ARGS__)))
#define AKR_EVAL_3(...) AKR_EVAL_2(AKR_EVAL_2(AKR_EVAL_2(__VA_ARGS__)))
#define AKR_EVAL_4(...) AKR_EVAL_3(AKR_EVAL_3(AKR_EVAL_3(__VA_ARGS__)))
#define AKR_EVAL(...)   AKR_EVAL_4(AKR_EVAL_4(AKR_EVAL_4(__VA_ARGS__)))
#define AKR_MAP_END(...)
#define AKR_MAP_OUT
#define AKR_MAP_COMMA                   ,
#define AKR_MAP_GET_END()               0, AKR_MAP_END
#define AKR_MAP_NEXT_0(test, next, ...) next AKR_MAP_OUT
#define AKR_MAP_NEXT_1(test, next)      AKR_MAP_NEXT_0(test, next, 0)
#define AKR_MAP_NEXT(test, next)        AKR_MAP_NEXT_1(AKR_MAP_GET_END test, next)
#define AKR_EXTRACT_0(next, ...)        next

#if defined(_MSC_VER) // MSVC is not as eager to expand macros, hence this workaround
#    define AKR_MAP_EXPR_NEXT_1(test, next) AKR_EVAL_0(AKR_MAP_NEXT_0(test, AKR_MAP_COMMA next, 0))
#    define AKR_MAP_STMT_NEXT_1(test, next) AKR_EVAL_0(AKR_MAP_NEXT_0(test, next, 0))
#else
#    define AKR_MAP_EXPR_NEXT_1(test, next) AKR_MAP_NEXT_0(test, AKR_MAP_COMMA next, 0)
#    define AKR_MAP_STMT_NEXT_1(test, next) AKR_MAP_NEXT_0(test, next, 0)
#endif

#define AKR_MAP_EXPR_NEXT(test, next) AKR_MAP_EXPR_NEXT_1(AKR_MAP_GET_END test, next)
#define AKR_MAP_STMT_NEXT(test, next) AKR_MAP_STMT_NEXT_1(AKR_MAP_GET_END test, next)

#define AKR_IMPORT_RENDER_TYPES_0(x, peek, ...)                                                                        \
    using A##x = x<Float, Spectrum>;                                                                                   \
    AKR_MAP_STMT_NEXT(peek, AKR_IMPORT_RENDER_TYPES_1)(peek, __VA_ARGS__)
#define AKR_IMPORT_RENDER_TYPES_1(x, peek, ...)                                                                        \
    using A##x = x<Float, Spectrum>;                                                                                   \
    AKR_MAP_STMT_NEXT(peek, AKR_IMPORT_RENDER_TYPES_0)(peek, __VA_ARGS__)
#define AKR_IMPORT_RENDER_TYPES_2(peek, ...)                                                                           \
    AKR_EVAL(AKR_MAP_STMT_NEXT(peek, AKR_IMPORT_RENDER_TYPES_0)(peek, __VA_ARGS__))

#define AKR_IMPORT_RENDER_TYPES(...) AKR_EVAL_0(AKR_IMPORT_RENDER_TYPES_2(__VA_ARGS__, (), 0))
