#pragma once

#define AKR_ASSERT(expr)                                  \
    if (!(expr)) {                                        \
        fprintf(stderr, "Assertion failed: %s\n", #expr); \
        abort();                                          \
    }

#define AKR_PANIC(msg)                       \
    ([&] () {                   \
        fprintf(stderr, "Panic: %s\n", msg); \
        abort();                             \
    })()