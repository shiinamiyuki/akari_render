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
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <akari/common.h>
#include <akari/pmr.h>
#if defined(AKR_GPU_BACKEND_CUDA) || defined(__CUDACC__) || defined(__NVCC__)
#    include <cuda.h>
#    define GLM_FORCE_CUDA
// #else
// #    define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// #    define GLM_FORCE_PURE
// #    define GLM_FORCE_INTRINSICS
// #    define GLM_FORCE_SSE2
// #    define GLM_FORCE_SSE3
// #    define GLM_FORCE_AVX
// #    define GLM_FORCE_AVX2
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <cereal/cereal.hpp>
#include <akari/common.h>
#include <akari/macro.h>

namespace akari {
    template <class T = astd::byte>
    using Allocator = astd::pmr::polymorphic_allocator<T>;
    template <typename T, typename... Ts>
    std::shared_ptr<T> make_pmr_shared(Allocator<> alloc, Ts &&... args) {
        return std::shared_ptr<T>(alloc.new_object<T>(std::forward<Ts>(args)...), [=](T *p) mutable {
            alloc.destroy(p);
            alloc.deallocate_object(const_cast<std::remove_const_t<T> *>(p), 1);
        });
    }
    template <typename T, int N>
    struct Color;
    using Float = float;
    using Spectrum = Color<Float, 3>;
#define USE_GLM_TVEC(prefix, i) using glm::prefix##vec##i;
#define USE_GLM_VEC_PREFIX(prefix)                                                                                     \
    USE_GLM_TVEC(prefix, 1) USE_GLM_TVEC(prefix, 2) USE_GLM_TVEC(prefix, 3) USE_GLM_TVEC(prefix, 4)
    USE_GLM_VEC_PREFIX(i)
    USE_GLM_VEC_PREFIX(u)
    USE_GLM_VEC_PREFIX(b)
    USE_GLM_VEC_PREFIX(d)
#define USE_GLM_TMAT(prefix, i)    using glm::prefix##mat##i;
#define USE_GLM_MAT_PREFIX(prefix) USE_GLM_TMAT(prefix, 2) USE_GLM_TMAT(prefix, 3) USE_GLM_TMAT(prefix, 4)
    USE_GLM_MAT_PREFIX(d)

    using glm::vec1;
    using glm::vec2;
    using glm::vec3;
    using glm::vec4;

    using glm::mat2;
    using glm::mat3;
    using glm::mat4;

    template <typename T, int N>
    using Vector = glm::vec<N, T, glm::defaultp>;
    template <typename T, int N>
    using Mat = glm::mat<N, N, T, glm::defaultp>;
    using Vec1 = Vector<Float, 1>;
    using Vec2 = Vector<Float, 2>;
    using Vec3 = Vector<Float, 3>;
    using Vec4 = Vector<Float, 4>;

    using Mat2 = Mat<Float, 2>;
    using Mat3 = Mat<Float, 3>;
    using Mat4 = Mat<Float, 4>;
#ifdef AKR_GPU_BACKEND_CUDA
#    define Inf       (std::numeric_limits<Float>::infinity())
#    define Pi        (Float(3.1415926535897932384f))
#    define PiOver2   (Pi / Float(2.0f))
#    define PiOver4   (Pi / Float(4.0f))
#    define InvPi     (Float(1.0f) / Pi)
#    define Inv2Pi    (Float(1.0) / (2.0 * Pi))
#    define Inv4Pi    (Float(1.0) / (4.0 * Pi))
#    define Eps       (Float(0.001f))
#    define ShadowEps (Float(0.0001f))
#else
    static constexpr Float Inf = std::numeric_limits<Float>::infinity();
    static constexpr Float Pi = Float(3.1415926535897932384f);
    static constexpr Float PiOver2 = Pi / Float(2.0f);
    static constexpr Float PiOver4 = Pi / Float(4.0f);
    static constexpr Float InvPi = Float(1.0f) / Pi;
    static constexpr Float Inv2Pi = Float(1.0) / (2.0 * Pi);
    static constexpr Float Inv4Pi = Float(1.0) / (4.0 * Pi);
    static constexpr Float Eps = Float(0.001f);
    static constexpr Float ShadowEps = Float(0.0001f);
#endif
    template <typename T, typename Float>
    T lerp(T a, T b, Float t) {
        return a * ((Float(1.0)) - t) + b * t;
    }
    template <typename V, typename V2>
    inline V lerp3(const V &v0, const V &v1, const V &v2, const V2 &uv) {
        return (1.0f - uv[0] - uv[1]) * v0 + uv[0] * v1 + uv[1] * v2;
    }
    template <typename V, typename V2>
    inline V dlerp3du(const V &v0, const V &v1, const V &v2, V2 u) {
        return -v0 + v1;
    }
    template <typename V, typename V2>
    inline V dlerp3dv(const V &v0, const V &v1, const V &v2, V2 v) {
        return -v0 + v2;
    }
    template <typename T, int N, class F>
    T reduce(const Vector<T, N> &vec, F &&f) {
        T acc = vec[0];
        for (int i = 1; i < N; i++) {
            acc = f(acc, vec[i]);
        }
        return acc;
    }
    template <int... args, typename T, int N>
    auto shuffle(const Vector<T, N> &a) {
        constexpr int pack[] = {args...};
        static_assert(((args < N) && ...));
        Vector<T, sizeof...(args)> s;
        for (size_t i = 0; i < sizeof...(args); i++) {
            s[i] = a[pack[i]];
        }
        return s;
    }
    template <typename T, int N>
    T hsum(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return acc + cur; });
    }
    template <typename T, int N>
    T hprod(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return acc * cur; });
    }
    using std::min;
    template <typename T, int N>
    T hmin(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return min(acc, cur); });
    }
    using std::max;
    template <typename T, int N>
    T hmax(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return max(acc, cur); });
    }
    template <typename T, int N, typename R, class F>
    R foldl(const Vector<T, N> &vec, R init, F &&f) {
        auto acc = f(init, vec[0]);
        for (int i = 1; i < N; i++) {
            acc = f(acc, vec[i]);
        }
        return acc;
    }
    template <typename T, int N>
    Vector<T, N> select(const Vector<bool, N> &c, const Vector<T, N> &a, const Vector<T, N> &b) {
        return glm::mix(b, a, c);
    }

    template <typename T>
    struct vec_trait {
        using value_type = void;
        static constexpr bool is_vector = false;
    };

    template <typename T, int N>
    struct vec_trait<Vector<T, N>> {
        using value_type = T;
        static constexpr int size = N;
        static constexpr bool is_vector = true;
    };

    template <typename T, typename V = typename vec_trait<T>::value_type, int N = vec_trait<T>::size>
    T load(const V *arr) {
        T v;
        for (int i = 0; i < N; i++)
            v[i] = arr[i];
        return v;
    }

    template <typename Scalar, int N>
    struct Color : Vector<Scalar, N> {
        using Base = Vector<Scalar, N>;
        using Base::Base;
        using value_t = Scalar;
        static constexpr size_t size = N;
        AKR_XPU Color(const Base &v) : Base(v) {}
#define AKR_COLOR_OP(op)                                                                                               \
    AKR_XPU Color operator op(const Color &rhs) const { return Color(Base(*this) op Base(rhs)); }                      \
    AKR_XPU Color operator op(Scalar rhs) const { return Color(Base(*this) op Base(rhs)); }                            \
    AKR_XPU friend Color operator op(Scalar lhs, const Color &rhs) { return Color(Base(lhs) op Base(rhs)); }           \
    AKR_XPU Color &operator op##=(const Color &rhs) {                                                                  \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }                                                                                                                  \
    AKR_XPU Color &operator op##=(Scalar rhs) {                                                                        \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }
        AKR_COLOR_OP(+) AKR_COLOR_OP(-) AKR_COLOR_OP(*) AKR_COLOR_OP(/)
#undef AKR_COLOR_OP
    };
    template <typename Scalar, int N>
    AKR_XPU Color<Scalar, N> clamp_zero(const Color<Scalar, N> &in) {
        Color<Scalar, N> c;
        for (int i = 0; i < N; i++) {
            auto x = in[i];
            if (isnan(x)) {
                x = 0;
            } else {
                x = max(Scalar(0.0f), x);
            }
            c[i] = x;
        }
        return c;
    }
    template <typename Scalar, int N>
    AKR_XPU Color<Scalar, N> min(const Color<Scalar, N> &in, const Color<Scalar, N> &v) {
        Color<Scalar, N> c;
        for (int i = 0; i < N; i++) {
            c[i] = std::min(in[i], v[i]);
        }
        return c;
    }
    template <typename Scalar, int N>
    AKR_XPU bool is_black(const Color<Scalar, N> &color) {
        return !foldl(color, false, [](bool acc, Scalar f) { return acc || (f > 0.0f); });
    }

    template <typename Scalar>
    AKR_XPU Color<Scalar, 3> linear_to_srgb(const Color<Scalar, 3> &L) {
        using Color3f = Color<Scalar, 3>;
        return select(glm::lessThan(L, Color3f(0.0031308)), L * 12.92,
                      Float(1.055) * glm::pow(L, Vec3(1.0f / 2.4f)) - Float(0.055));
    }
    template <typename Scalar>
    AKR_XPU Color<Scalar, 3> srgb_to_linear(const Color<Scalar, 3> &S) {
        using Color3f = Color<Scalar, 3>;
        return select(glm::lessThan(S, Color3f(0.04045)), S / 12.92, glm::pow((S + 0.055) / 1.055, Vec3(2.4)));
    }

    using Color3f = Color<Float, 3>;

    AKR_XPU inline Float luminance(const Color3f &rgb) { return dot(rgb, Vec3(0.2126, 0.7152, 0.0722)); }
    AKR_XPU inline Float average(const Color3f &rgb) { return hsum(rgb) / 3.0f; }
    template <typename T, int N>
    struct vec_trait<Color<T, N>> {
        using value_type = T;
        static constexpr int size = N;
        static constexpr bool is_vector = true;
    };
} // namespace akari
namespace cereal {
    template <class Archive, typename T>
    void save(Archive &ar, const std::optional<T> &optional) {
        if (!optional) {
            ar(CEREAL_NVP_("nullopt", true));
        } else {
            ar(CEREAL_NVP_("nullopt", false), CEREAL_NVP_("data", *optional));
        }
    }

    template <class Archive, typename T>
    void load(Archive &ar, std::optional<T> &optional) {
        bool nullopt;
        ar(CEREAL_NVP_("nullopt", nullopt));
        if (nullopt) {
            optional = std::nullopt;
        } else {
            T value;
            ar(CEREAL_NVP_("data", value));
            optional = std::move(value);
        }
    }

    template <typename Archive, int N, typename T, glm::qualifier Q>
    void serialize(Archive &ar, glm::vec<N, T, Q> &v) {
        for (int i = 0; i < N; i++) {
            ar(v[i]);
        }
    }

    template <typename Archive, int C, int R, typename T, glm::qualifier Q>
    void serialize(Archive &ar, glm::mat<C, R, T, Q> &v) {
        for (int i = 0; i < C; i++) {
            ar(v[i]);
        }
    }
} // namespace cereal
namespace akari {
    /*
    A Row major 2x2 matrix
    */
    template <typename T>
    struct Matrix2 {
        Matrix2(Mat<T, 2> m) : m(m) {}
        Matrix2() { m = glm::identity<Mat<T, 2>>(); }
        Matrix2(T x00, T x01, T x10, T x11) { m = Mat<T, 2>(x00, x10, x01, x11); }
        Matrix2 inverse() const { return Matrix2(glm::inverse(m)); }
        Vector<T, 2> operator*(const Vector<T, 2> &v) const { return m * v; }
        Matrix2<T> operator*(const Matrix2<T> &n) const { return Matrix2(m * n.m); }
        T &operator()(int i, int j) { return m[j][i]; }
        const T &operator()(int i, int j) const { return m[j][i]; }
        Float determinant() const { return glm::determinant(m); }

      private:
        Mat<T, 2> m;
    };
    using Matrix2f = Matrix2<float>;
    using Matrix2d = Matrix2<double>;

    struct Ray {
        // Float time = 0.0f;
        vec3 o;
        vec3 d;
        Float tmin = -1, tmax = -1;
        Ray() = default;
        Ray(const vec3 &o, const vec3 &d, Float tmin = Eps, Float tmax = std::numeric_limits<Float>::infinity())
            : o(o), d(d), tmin(tmin), tmax(tmax) {}
        static Ray spawn_to(const vec3 &p0, const vec3 &p1) {
            vec3 dir = p1 - p0;
            return Ray(p0, dir, Eps, Float(1.0f) - ShadowEps);
        }
        vec3 operator()(Float t) const { return o + t * d; }
        AKR_SER(o, d, tmin, tmax)
    };

    struct Frame {
        Frame() = default;
        static inline void compute_local_frame(const vec3 &v1, vec3 *v2, vec3 *v3) {
            if (std::abs(v1.x) > std::abs(v1.y))
                *v2 = vec3(-v1.z, (0), v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
            else
                *v2 = vec3((0), v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
            *v3 = normalize(cross(vec3(v1), *v2));
        }
        explicit Frame(const vec3 &n) : n(n) { compute_local_frame(n, &s, &t); }
        explicit Frame(const vec3 &n, const vec3 &dpdu) : n(n) {
            s = glm::normalize(-n * glm::dot(n, dpdu) + dpdu);
            t = glm::cross(n, s);
        }
        [[nodiscard]] vec3 world_to_local(const vec3 &v) const { return vec3(dot(s, v), dot(n, v), dot(t, v)); }

        [[nodiscard]] vec3 local_to_world(const vec3 &v) const { return vec3(v.x * s + v.y * vec3(n) + v.z * t); }

        vec3 n;
        vec3 s, t;
    };

    struct Transform {
        Mat4 m, minv;
        Mat3 m3, m3inv;
        AKR_SER(m, minv, m3, m3inv)
        Transform() : Transform(glm::mat4(1.0)) {}
        Transform(const Mat4 &m) : Transform(m, glm::inverse(m)) {}
        Transform(const Mat4 &m, const Mat4 &minv) : m(m), minv(minv) {
            m3 = glm::mat3(m);
            m3inv = glm::inverse(m3);
        }
        Transform inverse() const { return Transform(minv, m); }
        vec3 apply_point(const vec3 &p) const {
            Vec4 v(p.x, p.y, p.z, 1.0);
            v = m * v;
            vec3 q(v.x, v.y, v.z);
            if (v.w != 1.0) {
                q /= v.w;
            }
            return q;
        }
        Transform operator*(const Transform &t) const { return Transform(m * t.m); }
        vec3 apply_vector(const vec3 &v) const { return m3 * v; }
        vec3 apply_normal(const vec3 &n) const { return transpose(m3inv) * n; }

        static Transform translate(const vec3 &v) {
            mat4 m = glm::translate(glm::mat4(1.0), v);
            return Transform(m);
        }
        static Transform scale(const vec3 &s) {
            mat4 m = glm::scale(glm::mat4(1.0), s);
            return Transform(m);
        }
        static Transform rotate_x(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(1, 0, 0));
            return Transform(m, glm::transpose(m));
        }

        static Transform rotate_y(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(0, 1, 0));
            return Transform(m, glm::transpose(m));
        }

        static Transform rotate_z(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(0, 0, 1));
            return Transform(m, glm::transpose(m));
        }
    };

    template <typename T, int N>
    struct BoundingBox {
        using V = Vector<T, N>;
        V pmin, pmax;
        BoundingBox() { reset(); }
        BoundingBox(const V &pmin, const V &pmax) : pmin(pmin), pmax(pmax) {}
        void reset() {
            pmin = V(std::numeric_limits<T>::infinity());
            pmax = V(-std::numeric_limits<T>::infinity());
        }
        V extents() const { return pmax - pmin; }
        bool contains(const V &p) const { return all(p >= pmin && p <= pmax); }
        V size() const { return extents(); }
        V offset(const V &p) { return (p - pmin) / extents(); }
        BoundingBox expand(const V &p) const { return BoundingBox(min(pmin, p), max(pmax, p)); }
        BoundingBox merge(const BoundingBox &b1) const { return merge(*this, b1); }
        static BoundingBox merge(const BoundingBox &b1, const BoundingBox &b2) {
            return BoundingBox(min(b1.pmin, b2.pmin), max(b1.pmax, b2.pmax));
        }
        BoundingBox intersect(const BoundingBox &rhs) const {
            return BoundingBox(max(pmin, rhs.pmin), min(pmax, rhs.pmax));
        }
        V clip(const V &p) const { return min(max(p, pmin), pmax); }
        bool empty() const { return any(glm::greaterThan(pmin, pmax)) || hsum(extents()) == 0; }
        V centroid() const { return extents() * 0.5f + pmin; }
        Float surface_area() const {
            if (empty())
                return Float(0.0);
            if constexpr (N == 3) {
                auto ext = extents();
                return hsum(akari::shuffle<1, 2, 0>(ext) * ext) * Float(2);
            } else {
                auto ext = extents();
                Float result = Float(0);
                for (size_t i = 0; i < N; ++i) {
                    Float term = Float(1);
                    for (size_t j = 0; j < N; ++j) {
                        if (i == j)
                            continue;
                        term *= ext[j];
                    }
                    result += term;
                }
                return result * Float(2);
            }
        }
    };

    using Bounds2f = BoundingBox<Float, 2>;
    using Bounds3f = BoundingBox<Float, 3>;

    using Bounds2i = BoundingBox<int, 2>;
    using Bounds3i = BoundingBox<int, 3>;

    struct TRSTransform {
        Vec3 translation;
        Vec3 rotation;
        Vec3 scale = Vec3(1.0, 1.0, 1.0);
        TRSTransform() = default;
        TRSTransform(Vec3 t, Vec3 r, Vec3 s) : translation(t), rotation(r), scale(s) {}
        Transform operator()() const {
            Transform T;
            T = Transform::scale(scale);
            T = Transform::rotate_z(rotation.z) * T;
            T = Transform::rotate_x(rotation.y) * T;
            T = Transform::rotate_y(rotation.x) * T;
            T = Transform::translate(translation) * T;
            return T;
        }
        AKR_SER(translation, rotation, scale)
    };

} // namespace akari

namespace akari {
    using namespace glm;
    namespace astd {
        template <class To, class From>
        typename std::enable_if_t<
            sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>, To>
        // constexpr support needs compiler magic
        bit_cast(const From &src) noexcept {
            static_assert(std::is_trivially_constructible_v<To>,
                          "This implementation additionally requires destination type to be trivially constructible");

            To dst;
            std::memcpy(&dst, &src, sizeof(To));
            return dst;
        }

        inline constexpr size_t max(size_t a, size_t b) { return a < b ? b : a; }
        template <typename T1, typename T2>
        struct alignas(astd::max(alignof(T1), alignof(T2))) pair {
            T1 first;
            T2 second;
        };
        // template <typename T1, typename T2>
        // pair(T1 &&, T2 &&) -> pair<T1, T2>;

        // template <typename T1, typename T2>
        // AKR_XPU pair<T1, T2> make_pair(T1 a, T2 b) {
        //     return pair<T1, T2>{a, b};
        // }

        struct nullopt_t {};
        inline constexpr nullopt_t nullopt{};
        template <typename T>
        class optional {
          public:
            using value_type = T;
            AKR_XPU optional(nullopt_t) : optional() {}
            optional() = default;
            AKR_XPU
            optional(const T &v) : set(true) { new (ptr()) T(v); }
            AKR_XPU
            optional(T &&v) : set(true) { new (ptr()) T(std::move(v)); }
            AKR_XPU
            optional(const optional &v) : set(v.has_value()) {
                if (v.has_value())
                    new (ptr()) T(v.value());
            }
            AKR_XPU
            optional(optional &&v) : set(v.has_value()) {
                if (v.has_value()) {
                    new (ptr()) T(std::move(v.value()));
                    v.reset();
                }
            }

            AKR_XPU
            optional &operator=(const T &v) {
                reset();
                new (ptr()) T(v);
                set = true;
                return *this;
            }
            AKR_XPU
            optional &operator=(T &&v) {
                reset();
                new (ptr()) T(std::move(v));
                set = true;
                return *this;
            }
            AKR_XPU
            optional &operator=(const optional &v) {
                reset();
                if (v.has_value()) {
                    new (ptr()) T(v.value());
                    set = true;
                }
                return *this;
            }
            template <typename... Ts>
            AKR_CPU void emplace(Ts &&... args) {
                reset();
                new (ptr()) T(std::forward<Ts>(args)...);
                set = true;
            }
            AKR_XPU
            optional &operator=(optional &&v) {
                reset();
                if (v.has_value()) {
                    new (ptr()) T(std::move(v.value()));
                    set = true;
                    v.reset();
                }
                return *this;
            }

            AKR_XPU
            ~optional() { reset(); }

            AKR_XPU
            explicit operator bool() const { return set; }

            AKR_XPU
            T value_or(const T &alt) const { return set ? value() : alt; }

            AKR_XPU
            T *operator->() { return &value(); }
            AKR_XPU
            const T *operator->() const { return &value(); }
            AKR_XPU
            T &operator*() { return value(); }
            AKR_XPU
            const T &operator*() const { return value(); }
            AKR_XPU
            T &value() {
                AKR_CHECK(set);
                return *ptr();
            }
            AKR_XPU
            const T &value() const {
                AKR_CHECK(set);
                return *ptr();
            }

            AKR_XPU
            void reset() {
                if (set) {
                    value().~T();
                    set = false;
                }
            }

            AKR_XPU
            bool has_value() const { return set; }

          private:
            // #ifdef __NVCC__
            // Work-around NVCC bug
            AKR_XPU
            T *ptr() { return reinterpret_cast<T *>(&optionalValue); }
            AKR_XPU
            const T *ptr() const { return reinterpret_cast<const T *>(&optionalValue); }
            // #else
            //         AKR_XPU
            //         T *ptr() { return std::launder(reinterpret_cast<T *>(&optionalValue)); }
            //         AKR_XPU
            //         const T *ptr() const { return std::launder(reinterpret_cast<const T *>(&optionalValue)); }
            // #endif

            std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
            bool set = false;
        };

        // template <int I, typename T1, typename T2>
        // AKR_XPU const T1 &get(const pair<T1, T2> &p) {
        //     static_assert(I >= 0 && I < 2);
        //     if constexpr (I == 0) {
        //         return p.first;
        //     } else {
        //         return p.second;
        //     }
        // }
        template <typename T>
        AKR_XPU inline void swap(T &a, T &b) {
            T tmp = std::move(a);
            a = std::move(b);
            b = std::move(tmp);
        }
    } // namespace astd

    template <typename... T>
    struct TypeIndex {
        template <typename U, typename Tp, typename... Rest>
        struct GetIndex_ {
            static const int value =
                std::is_same<Tp, U>::value
                    ? 0
                    : ((GetIndex_<U, Rest...>::value == -1) ? -1 : 1 + GetIndex_<U, Rest...>::value);
        };
        template <typename U, typename Tp>
        struct GetIndex_<U, Tp> {
            static const int value = std::is_same<Tp, U>::value ? 0 : -1;
        };
        template <int I, typename Tp, typename... Rest>
        struct GetType_ {
            using type = typename std::conditional<I == 0, Tp, typename GetType_<I - 1, Rest...>::type>::type;
        };

        template <int I, typename Tp>
        struct GetType_<I, Tp> {
            using type = typename std::conditional<I == 0, Tp, void>::type;
        };

        template <typename U>
        struct GetIndex {
            static const int value = GetIndex_<U, T...>::value;
        };

        template <int N>
        struct GetType {
            using type = typename GetType_<N, T...>::type;
        };
    };
    template <class T, class... Rest>
    struct FirstOf {
        using type = T;
    };

    template <typename U, typename... T>
    struct SizeOf {
        static constexpr int value = std::max<int>(sizeof(U), SizeOf<T...>::value);
    };
    template <typename T>
    struct SizeOf<T> {
        static constexpr int value = sizeof(T);
    };

    template <typename... T>
    struct Variant {
      private:
        static constexpr int nTypes = sizeof...(T);
        static constexpr std::size_t alignment_value = std::max({alignof(T)...});
        typename std::aligned_storage<SizeOf<T...>::value, alignment_value>::type data;
        int index = -1;

      public:
        using Index = TypeIndex<T...>;
        static constexpr size_t num_types = nTypes;

        template <typename U>
        AKR_XPU Variant(const U &u) {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            new (&data) U(u);
            index = Index::template GetIndex<U>::value;
        }

        AKR_XPU Variant(const Variant &v) : index(v.index) {
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                new (&data) U(item);
            });
        }
        AKR_XPU int typeindex() const { return index; }
        template <typename U>
        AKR_XPU constexpr static int indexof() {
            return Index::template GetIndex<U>::value;
        }
        AKR_XPU Variant &operator=(const Variant &v) noexcept {
            if (this == &v)
                return *this;
            if (index != -1)
                _drop();
            index = v.index;
            auto that = this;
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                *that->template get<U>() = item;
            });
            return *this;
        }

        AKR_XPU Variant(Variant &&v) noexcept : index(v.index) {
            index = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
        }

        AKR_XPU Variant &operator=(Variant &&v) noexcept {
            if (index != -1)
                _drop();
            index = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
            return *this;
        }

        template <typename U>
        AKR_XPU Variant &operator=(const U &u) {
            if (index != -1) {
                _drop();
            }
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            new (&data) U(u);
            index = Index::template GetIndex<U>::value;
            return *this;
        }
        AKR_XPU bool null() const { return index == -1; }
        template <typename U>
        AKR_XPU bool isa() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value == index;
        }
        template <typename U>
        AKR_XPU U *get() {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<U *>(&data);
        }

        template <typename U>
        AKR_XPU const U *get() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<const U *>(&data);
        }

#define _GEN_CASE_N(N)                                                                                                 \
    case N:                                                                                                            \
        if constexpr (N < nTypes) {                                                                                    \
            using ty = typename Index::template GetType<N>::type;                                                      \
            if constexpr (!std::is_same_v<ty, std::monostate>) {                                                       \
                if constexpr (std::is_const_v<std::remove_pointer_t<decltype(this)>>)                                  \
                    return visitor(*reinterpret_cast<const ty *>(&data));                                              \
                else                                                                                                   \
                    return visitor(*reinterpret_cast<ty *>(&data));                                                    \
            }                                                                                                          \
        };                                                                                                             \
        break;
#define _GEN_CASES_2()                                                                                                 \
    _GEN_CASE_N(0)                                                                                                     \
    _GEN_CASE_N(1)
#define _GEN_CASES_4()                                                                                                 \
    _GEN_CASES_2()                                                                                                     \
    _GEN_CASE_N(2)                                                                                                     \
    _GEN_CASE_N(3)
#define _GEN_CASES_8()                                                                                                 \
    _GEN_CASES_4()                                                                                                     \
    _GEN_CASE_N(4)                                                                                                     \
    _GEN_CASE_N(5)                                                                                                     \
    _GEN_CASE_N(6)                                                                                                     \
    _GEN_CASE_N(7)
#define _GEN_CASES_16()                                                                                                \
    _GEN_CASES_8()                                                                                                     \
    _GEN_CASE_N(8)                                                                                                     \
    _GEN_CASE_N(9)                                                                                                     \
    _GEN_CASE_N(10)                                                                                                    \
    _GEN_CASE_N(11)                                                                                                    \
    _GEN_CASE_N(12)                                                                                                    \
    _GEN_CASE_N(13)                                                                                                    \
    _GEN_CASE_N(14)                                                                                                    \
    _GEN_CASE_N(15)
#define _GEN_DISPATCH_BODY()                                                                                           \
    using Ret = std::invoke_result_t<Visitor, typename FirstOf<T...>::type &>;                                         \
    static_assert(nTypes <= 16, "too many types");                                                                     \
    if constexpr (nTypes <= 2) {                                                                                       \
        switch (index) { _GEN_CASES_2(); }                                                                             \
    } else if constexpr (nTypes <= 4) {                                                                                \
        switch (index) { _GEN_CASES_4(); }                                                                             \
    } else if constexpr (nTypes <= 8) {                                                                                \
        switch (index) { _GEN_CASES_8(); }                                                                             \
    } else if constexpr (nTypes <= 16) {                                                                               \
        switch (index) { _GEN_CASES_16(); }                                                                            \
    }                                                                                                                  \
    if constexpr (std::is_same_v<void, Ret>) {                                                                         \
        return;                                                                                                        \
    } else {                                                                                                           \
        AKR_PANIC("No matching case");                                                                                 \
    }
        template <class Visitor>
        AKR_XPU auto dispatch(Visitor &&visitor) {
            _GEN_DISPATCH_BODY()
        }

        template <class Visitor>
        AKR_XPU auto dispatch(Visitor &&visitor) const {_GEN_DISPATCH_BODY()}

        AKR_XPU ~Variant() {
            if (index != -1)
                _drop();
        }

      private:
        AKR_XPU void _drop() {
            auto *that = this; // prevent gcc ICE
            dispatch([=](auto &&self) {
                using U = std::decay_t<decltype(self)>;
                that->template get<U>()->~U();
            });
        }
#undef _GEN_CASE_N
#define AKR_VAR_DISPATCH(method, ...)                                                                                  \
    return this->dispatch([&, this](auto &&self) {                                                                     \
        (void)this;                                                                                                    \
        return self.method(__VA_ARGS__);                                                                               \
    });
#define AKR_VAR_PTR_DISPATCH(method, ...)                                                                              \
    return this->dispatch([&, this](auto &&self) {                                                                     \
        (void)this;                                                                                                    \
        return self->method(__VA_ARGS__);                                                                              \
    });
    };
    template <class... Ts>
    struct overloaded : Ts... {
        using Ts::operator()...;
    };
    // explicit deduction guide (not needed as of C++20)
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    template <typename T>
    struct BufferView {
        BufferView() = default;
        template <typename Allocator>
        BufferView(const std::vector<T, Allocator> &vec) : _data(vec.data()), _size(vec.size()) {}
        template <typename Allocator, typename = std::enable_if_t<std::is_const_v<T>>>
        BufferView(const std::vector<std::remove_const_t<T>, Allocator> &vec) : _data(vec.data()), _size(vec.size()) {}
        template <typename Allocator, typename = std::enable_if_t<std::is_const_v<T>>>
        BufferView(const T *data, size_t size) : _data(data), _size(size) {}
        BufferView(T *data, size_t size) : _data(data), _size(size) {}
        T &operator[](uint32_t i) const { return _data[i]; }
        size_t size() const { return _size; }
        T *begin() const { return _data; }
        T *end() const { return _data + _size; }
        const T *cbegin() const { return _data; }
        const T *cend() const { return _data + _size; }
        T *const &data() const { return _data; }
        bool empty() const { return size() == 0; }

      private:
        T *_data = nullptr;
        size_t _size = 0;
    };
} // namespace akari

namespace cereal {
    template <class Archive, typename... Ts>
    void save(Archive &ar, const std::variant<Ts...> &variant) {
        using Index = akari::TypeIndex<Ts...>;

        std::visit(
            [&](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                size_t index = Index::template GetIndex<T>::value;
                ar(CEREAL_NVP_("index", index));
                ar(CEREAL_NVP_("value", arg));
            },
            variant);
    }
    template <class Archive, typename Variant, class Index, typename T, typename... Ts>
    void load_helper(size_t index, Archive &ar, Variant &variant) {
        if (Index::template GetIndex<T>::value == index) {
            T v;
            ar(CEREAL_NVP_("value", v));
            variant = v;
        } else if constexpr (sizeof...(Ts) > 0) {
            load_helper<Archive, Variant, Index, Ts...>(index, ar, variant);
        } else {
            throw std::runtime_error("cannot load variant");
        }
    }
    template <class Archive, typename... Ts>
    void load(Archive &ar, std::variant<Ts...> &variant) {
        size_t index;
        ar(CEREAL_NVP_("index", index));
        using Index = akari::TypeIndex<Ts...>;
        load_helper<Archive, std::variant<Ts...>, Index, Ts...>(index, ar, variant);
    }
} // namespace cereal
