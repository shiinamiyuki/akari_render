#pragma once
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <cstddef>
// prevent bindgen exploding
#ifndef _MSC_VER
#include <algorithm>
#endif

#define AKR_ASSERT(x) ([&]() { if (!(x)) { fprintf(stderr, "assertion failed %s\n", #x); abort(); } })()
struct Mesh;
struct MLoopTri;
using ParallelForFn = void (*)(const void *, size_t);
struct ParallelForContext {
    template<class F>
        requires std::is_nothrow_invocable_v<F, size_t> && std::same_as<std::invoke_result_t<F, size_t>, void>
    void parallel_for(size_t count, size_t chunk_size, F &&f) const noexcept {
        auto chunks = (count + chunk_size - 1) / chunk_size;
        auto wrapper = [&](size_t chunk_id) noexcept {
            const size_t start = chunk_id * chunk_size;
            const size_t end = std::min(start + chunk_size, count);
            for (size_t i = start; i < end; i++) {
                f(i);
            }
        };
        _parallel_for(chunks, [](const void *f, auto i) { (*static_cast<const decltype(wrapper) *>(f))(i); }, &wrapper);
    }
    const void (*_parallel_for)(size_t, ParallelForFn, const void *);
};
namespace blender_util {
extern "C" {
void get_mesh_triangle_indices(const ParallelForContext &ctx, const Mesh *mesh,
                               const MLoopTri *tri, size_t count,
                               uint32_t *out);
bool get_mesh_tangents(const ParallelForContext &ctx, const Mesh *mesh,
                       const MLoopTri *tri, size_t count, float *out);
bool get_mesh_split_normals(const ParallelForContext &ctx, const Mesh *mesh,
                            const MLoopTri *tri, size_t count, float *out);
void get_mesh_material_indices(const ParallelForContext &ctx, const Mesh *mesh,
                               const MLoopTri *tri, size_t count,
                               uint32_t *out);
}
}// namespace blender_util
#define AKR_PANIC(msg) ([&]() { fprintf(stderr, "panic: %s\n", msg); abort(); })()
namespace spectral {
extern "C" {
int rgb2spec_opt(int argc, const char *const *argv);
}
}// namespace spectral