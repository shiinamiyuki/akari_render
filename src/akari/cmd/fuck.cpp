
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
#define AKR_STRUCT_REFL_MEMBER(member)                                                                                 \
    {                                                                                                                  \
        auto get_member = [](auto &object) { return object.member };                                                   \
        auto get_member_cst = [](const auto &object) { return object.member };                                         \
        vis(get_member, get_member_cst);                                                                               \
    }
#if 0

#    define AKR_STRUCT_REFL_0(dummy, x, peek, ...)                                                                     \
        AKR_STRUCT_REFL_MEMBER(x)                                                                                      \
        AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_1)(dummy, peek, __VA_ARGS__)
#    define AKR_STRUCT_REFL_1(dummy, x, peek, ...)                                                                     \
        AKR_STRUCT_REFL_MEMBER(x)                                                                                      \
        AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_0)(dummy, peek, __VA_ARGS__)
#    define AAKR_STRUCT_REFL_2(dummy, peek, ...)                                                                       \
        AKR_EVAL(AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_0)(dummy, peek, __VA_ARGS__))
#endif


#if 0

#define AKR_USING_TYPES_0(base, x, peek, ...)                                                                          \
    AKR_EVAL_0(AKR_STRUCT_REFL_MEMBER(x))                                                                              \
    AKR_MAP_STMT_NEXT(peek, AKR_USING_TYPES_1)(base, peek, __VA_ARGS__)
#define AKR_USING_TYPES_1(base, x, peek, ...)                                                                          \
    AKR_EVAL_0(AKR_STRUCT_REFL_MEMBER(x))                                                                              \
    AKR_MAP_STMT_NEXT(peek, AKR_USING_TYPES_0)(base, peek, __VA_ARGS__)
#define AKR_USING_TYPES_2(base, peek, ...) AKR_EVAL(AKR_MAP_STMT_NEXT(peek, AKR_USING_TYPES_0)(base, peek, __VA_ARGS__))
#    define AKR_STRUCT_REFL(...) AKR_EVAL_0(AKR_USING_TYPES_2(__VA_ARGS__, (), 0))
AKR_STRUCT_REFL(fuck, a, b)

#else
#define AKR_STRUCT_REFL_0(x, peek, ...)                                                                          \
    AKR_EVAL_0(AKR_STRUCT_REFL_MEMBER(x))                                                                              \
    AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_1)(peek, __VA_ARGS__)
#define AKR_STRUCT_REFL_1(x, peek, ...)                                                                          \
    AKR_EVAL_0(AKR_STRUCT_REFL_MEMBER(x))                                                                              \
    AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_0)(peek, __VA_ARGS__)
#define AKR_STRUCT_REFL_2(peek, ...) AKR_EVAL(AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_0)(peek, __VA_ARGS__))


#    define AKR_STRUCT_REFL(...)                                                                                       \
        template <class Visitor>                                                                                       \
        static void _for_each_field(Visitor &&vis) {                                                                   \
            AKR_EVAL_0(AKR_STRUCT_REFL_2(__VA_ARGS__, (), 0))                                                          \
        }                                                                                                              \
        template <class Visitor>                                                                                       \
        static void foreach_field(Visitor &&vis) {                                                                     \
            _for_each_field([&](auto &&get, auto &&get_c) { vis(get); });                                              \
        }                                                                                                              \
        template <class Visitor>                                                                                       \
        static void foreach_field_const(Visitor &&vis) {                                                               \
            _for_each_field([&](auto &&get, auto &&get_c) { vis(get_c); });                                            \
        }
#endif
struct Fuck {
    int a, b;
    AKR_STRUCT_REFL(a, b)
};