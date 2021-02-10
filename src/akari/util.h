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
#include <cstring>
#include <akari/common.h>
#include <akari/pmr.h>
#include <akari/util_xpu.h>
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
#include <akari/mathutil.h>
#include <cereal/cereal.hpp>
#include <akari/common.h>
#include <akari/macro.h>
namespace akari {
    class NonCopyable {
      public:
        NonCopyable()                    = default;
        NonCopyable(const NonCopyable &) = delete;
        NonCopyable &operator=(const NonCopyable &) = delete;
        NonCopyable(NonCopyable &&)                 = default;
    };
    template <class T = astd::byte>
    using Allocator = astd::pmr::polymorphic_allocator<T>;
    template <typename T, typename... Ts>
    std::shared_ptr<T> make_pmr_shared(Allocator<> alloc, Ts &&...args) {
        return std::shared_ptr<T>(alloc.new_object<T>(std::forward<Ts>(args)...), [=](T *p) mutable {
            alloc.destroy(p);
            alloc.deallocate_object(const_cast<std::remove_const_t<T> *>(p), 1);
        });
    }
    template <class Archive, class T>
    void safe_apply(Archive &ar, const char *name, T &val) {
        try {
            ar(CEREAL_NVP_(name, val));
        } catch (cereal::Exception &e) {
            std::string_view msg(e.what());
            if (msg.find("provided NVP") == std::string_view::npos) {
                // not a name not found error
                throw e;
            }
        }
    }
} // namespace akari
namespace cereal {
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
    struct TRSTransform {
        Vec3 translation;
        Vec3 rotation;
        Vec3 scale     = Vec3(1.0, 1.0, 1.0);
        TRSTransform() = default;
        AKR_XPU TRSTransform(Vec3 t, Vec3 r, Vec3 s) : translation(t), rotation(r), scale(s) {}
        AKR_XPU Transform operator()() const {
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

    template <typename T>
    struct ObjectCache {
        template <class F>
        T &get_cached_or(F &&f) {
            if (!_storage) {
                _storage = f();
            }
            return _storage.value();
        }
        void invalidate() { _storage.reset(); }

      private:
        astd::optional<T> _storage;
    };

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

        template <class Fn>
        struct scope_exit {
            explicit scope_exit(Fn &&fn) noexcept : fn(fn) {}
            ~scope_exit() { fn(); }
            scope_exit(const scope_exit &) = delete;

          private:
            Fn fn;
        };
        template <class Fn>
        scope_exit(Fn &&) -> scope_exit<Fn>;
    } // namespace astd
} // namespace akari