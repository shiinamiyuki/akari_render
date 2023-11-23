#pragma once
#include <stdint.h>
struct Mesh;
struct MLoopTri;
struct TheadPoolContext {
    size_t num_threads;
    const void *context;
    void (*_spawn)(const void *context, void (*func)(void *), void *data);
    template<class F>
    void spawn(F &&f) const noexcept {
        _spawn(
            context, [](void *data) { (*static_cast<F *>(data))(); }, &f);
    }
    template<class F>
    void parallel_for(size_t count, F &&f) const noexcept {
        size_t num_threads = this->num_threads;
        size_t chunk_size = (count + num_threads - 1) / num_threads;
        for (size_t i = 0; i < num_threads; i++) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, count);
            spawn([=, &f] {
                for (size_t j = start; j < end; j++) {
                    f(j);
                }
            });
        }
    }
};
extern "C" {
void get_mesh_triangle_indices(const TheadPoolContext &ctx, const Mesh *mesh,
                               const MLoopTri *tri, size_t count,
                               uint32_t *out);
bool get_mesh_tangents(const TheadPoolContext &ctx, const Mesh *mesh,
                       const MLoopTri *tri, size_t count, float *out);
bool get_mesh_split_normals(const TheadPoolContext &ctx, const Mesh *mesh,
                            const MLoopTri *tri, size_t count, float *out);
void get_mesh_material_indices(const TheadPoolContext &ctx, const Mesh *mesh,
                               const MLoopTri *tri, size_t count,
                               uint32_t *out);
}