// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <akari/diagnostic.h>

#ifdef _MSC_VER
#    define AKR_EXPORT __declspec(dllexport)
#    pragma warning(disable : 4275)
#    pragma warning(disable : 4267)
#    pragma warning(                                                                                                   \
        disable : 4251) // 'field' : class 'A' needs to have dll-interface to be used by clients of class 'B'
#    pragma warning(disable : 4800) // 'type' : forcing value to bool 'true' or 'false' (performance warning)
#    pragma warning(disable : 4996) // Secure SCL warnings
#    pragma warning(disable : 5030)
#    pragma warning(disable : 4324)
#    pragma warning(disable : 4201)
#    define AKR_FORCEINLINE __forceinline
#else
#    if defined _WIN32 || defined __CYGWIN__
#        ifdef __GNUC__
#            define AKR_EXPORT __attribute__((dllexport))

#        else
#            define AKR_EXPORT __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
#        endif
#    else

#        define AKR_EXPORT __attribute__((visibility("default")))

#    endif
#    define AKR_FORCEINLINE inline __attribute__((always_inline))
#endif

#ifdef _MSC_VER
#define __restrict__ __restrict 
#endif

#ifdef __GNUC__
#    if __GNUC__ >= 8 || defined(__clang__)

#        include <filesystem>
namespace akari {
    namespace fs = std::filesystem;
}
#    else
#        include <experimental/filesystem>
namespace akari {
    namespace fs = std::experimental::filesystem;
}
#    endif
#else
#    include <filesystem>
namespace akari {
    namespace fs = std::filesystem;
}
#endif
#define AKR_CPU
#define AKR_GPU
#if defined(AKR_GPU_BACKEND_CUDA)
#    define AKR_ENABLE_GPU
#endif
#if defined(__CUDA_ARCH__)
#    define AKR_GPU_CODE
#endif

#ifdef AKR_GPU_BACKEND_CUDA
#    include <cuda_runtime.h>
#    undef AKR_CPU
#    undef AKR_GPU
#    define AKR_CPU __host__
#    define AKR_GPU __device__
#endif

#define AKR_XPU AKR_GPU AKR_CPU

namespace akari {
    [[noreturn]] AKR_XPU inline void panic(const char *file, int line, const char *msg) {
#ifndef AKR_GPU_CODE
        fprintf(stderr, "PANIC at %s:%d: %s\n", file, line, msg);
        std::abort();
#else
#    ifdef AKE_BACKEND_CUDA
         printf"PANIC at %s:%d: %s\n", file, line, msg);
         asm("trap;");
#    endif
#endif
    }
#define AKR_PANIC(msg) akari::panic(__FILE__, __LINE__, msg)
#ifdef AKR_GPU_CODE
#    define AKR_CHECK(expr)
#else
#    define AKR_CHECK(expr)                                                                                            \
        do {                                                                                                           \
            if (!(expr)) {                                                                                             \
                fprintf(stderr, #expr " not satisfied at %s:%d\n", __FILE__, __LINE__);                                \
            }                                                                                                          \
        } while (0)
#endif
#define AKR_ASSERT(expr)                                                                                               \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            AKR_PANIC(#expr " not satisfied");                                                                         \
        }                                                                                                              \
    } while (0)
#define AKR_ASSERT_THROW(expr)                                                                                         \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            throw std::runtime_error(#expr " not satisfied");                                                          \
        }                                                                                                              \
    } while (0)

   
} // namespace akari