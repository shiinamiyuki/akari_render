#pragma once

#define AKR_ASSERT(expr)                                  \
    if (!(expr)) {                                        \
        fprintf(stderr, "Assertion failed: %s\n", #expr); \
        abort();                                          \
    }