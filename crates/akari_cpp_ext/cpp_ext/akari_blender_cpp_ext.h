#pragma once
#include <cstdint>
#include <type_traits>
#include <concepts>
struct Mesh;
struct MLoopTri;
template<class F, class R, class... Args>
concept Fn = std::is_invocable_v<F, Args...> && std::is_nothrow_invocable_v<F, Args...> && std::same_as<std::invoke_result_t<F, Args...>, R>;
struct RayonScope {
private:
    size_t num_threads{};
    const void *context{};
    void (*_spawn)(const void *context, void (*func)(const void *, void *), void *data) = nullptr;
public:
    template<class F>
        requires Fn<F, void, const RayonScope &>
    void spawn(F &&f) const noexcept {
        _spawn(
            context, [](const void *ctx, void *data) noexcept {
                auto &s = *static_cast<const RayonScope *>(ctx);
                (*static_cast<F *>(data))(s);
            },
            &f);
    }
    template<class F>
    void parallel_for(const size_t count, F &&f) const noexcept {
        spawn([=, &f](const RayonScope &s) noexcept {
            const size_t num_threads = this->num_threads;
            const size_t chunk_size = (count + num_threads - 1) / num_threads;
            for (size_t i = 0; i < num_threads; i++) {
                const size_t start = i * chunk_size;
                const size_t end = std::min(start + chunk_size, count);
                s.spawn([=, &f](const RayonScope &_) noexcept {
                    for (size_t j = start; j < end; j++) {
                        f(j);
                    }
                });
            }
        });
    }
};
namespace blender_util {
extern "C" {
void get_mesh_triangle_indices(const RayonScope &ctx, const Mesh *mesh,
                               const MLoopTri *tri, size_t count,
                               uint32_t *out);
bool get_mesh_tangents(const RayonScope &ctx, const Mesh *mesh,
                       const MLoopTri *tri, size_t count, float *out);
bool get_mesh_split_normals(const RayonScope &ctx, const Mesh *mesh,
                            const MLoopTri *tri, size_t count, float *out);
void get_mesh_material_indices(const RayonScope &ctx, const Mesh *mesh,
                               const MLoopTri *tri, size_t count,
                               uint32_t *out);
}
}// namespace blender_util

namespace spectral {
extern "C" {
int rgb2spec_opt(int argc, const char *const *argv);
}
}// namespace spectral