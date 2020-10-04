namespace akari::asl {
    [[noreturn]] inline void panic(const char *file, int line, const char *msg) {
        printf("PANIC at %s:%d: %s\n", file, line, msg);
        std::abort();
    }

}
#define AKR_PANIC(msg) panic(__FILE__, __LINE__, msg)

#define AKR_CHECK(expr)                                                                                                \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, #expr " not satisfied at %s:%d\n", __FILE__, __LINE__);                                    \
        }                                                                                                              \
    } while (0)

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