#include "DNA_mesh_types.h"
#include "BKE_mesh_types.hh"
#include "akari_cpp_ext.h"
/// this code is strictly for Blender 4.0

namespace copied_from_blender_4_0 {
int CustomData_get_active_layer_index(const CustomData *data, const eCustomDataType type) {
    const int layer_index = data->typemap[type];
    return (layer_index != -1) ? layer_index + data->layers[layer_index].active : -1;
}

const void *CustomData_get_layer(const CustomData *data, const eCustomDataType type) {
    const int layer_index = CustomData_get_active_layer_index(data, type);
    if (layer_index == -1) {
        return nullptr;
    }

    return data->layers[layer_index].data;
}

int CustomData_get_named_layer_index(const CustomData *data,
                                     const eCustomDataType type,
                                     const char *name) {
    for (int i = 0; i < data->totlayer; i++) {
        if (data->layers[i].type == type) {
            if (STREQ(data->layers[i].name, name)) {
                return i;
            }
        }
    }

    return -1;
}

const void *CustomData_get_layer_named(const CustomData *data,
                                       const eCustomDataType type,
                                       const char *name) {
    const int layer_index = CustomData_get_named_layer_index(data, type, name);
    if (layer_index == -1) {
        return nullptr;
    }
    return data->layers[layer_index].data;
}
const int *BKE_mesh_corner_verts(const Mesh *mesh) {
    return (const int *)CustomData_get_layer_named(&mesh->loop_data, CD_PROP_INT32, ".corner_vert");
}

const int *BKE_mesh_material_indices(const Mesh *mesh) {
    return (const int *)CustomData_get_layer_named(
        &mesh->face_data, CD_PROP_INT32, "material_index");
}

}// namespace copied_from_blender_4_0

namespace blender_util {
extern "C" void get_mesh_triangle_indices(const ParallelForContext &ctx, const Mesh *mesh, const MLoopTri *tri, size_t count, uint32_t *out) {
    const int *corner_verts = copied_from_blender_4_0::BKE_mesh_corner_verts(mesh);
    AKR_ASSERT(corner_verts);
    ctx.parallel_for(count, 1024, [&](size_t i) noexcept {
        out[i * 3 + 0] = corner_verts[tri[i].tri[0]];
        out[i * 3 + 1] = corner_verts[tri[i].tri[1]];
        out[i * 3 + 2] = corner_verts[tri[i].tri[2]]; });
}

extern "C" bool get_mesh_tangents(const ParallelForContext &ctx, const Mesh *mesh, const MLoopTri *tri, size_t count, float *out) {
    const auto layer = static_cast<const std::array<float, 4> *>(copied_from_blender_4_0::CustomData_get_layer(&mesh->loop_data, CD_MLOOPTANGENT));
    if (!layer) {
        return false;
    } else {
        ctx.parallel_for(count, 1024, [&](size_t i) noexcept {
            std::memcpy(out + i * 9 + 0, layer + tri[i].tri[0], sizeof(float) * 3);
            std::memcpy(out + i * 9 + 3, layer + tri[i].tri[1], sizeof(float) * 3);
            std::memcpy(out + i * 9 + 6, layer + tri[i].tri[2], sizeof(float) * 3);
        });
        return true;
    }
}

extern "C" bool get_mesh_split_normals(const ParallelForContext &ctx, const Mesh *mesh, const MLoopTri *tri, size_t count, float *out) {
    const auto layer = static_cast<const std::array<float, 3> *>(copied_from_blender_4_0::CustomData_get_layer(&mesh->loop_data, CD_NORMAL));

    if (!layer) {
        return false;
    } else {
        ctx.parallel_for(count, 1024, [&](size_t i) noexcept {
            std::memcpy(out + i * 9 + 0, layer + tri[i].tri[0], sizeof(float) * 3);
            std::memcpy(out + i * 9 + 3, layer + tri[i].tri[1], sizeof(float) * 3);
            std::memcpy(out + i * 9 + 6, layer + tri[i].tri[2], sizeof(float) * 3);
        });
        return true;
    }
}
extern "C" void get_mesh_material_indices(const ParallelForContext &ctx, const Mesh *mesh, const MLoopTri *tri, size_t count, uint32_t *out) {
    const auto &faces = mesh->runtime->looptri_faces_cache.data();
    const int *material_indices = copied_from_blender_4_0::BKE_mesh_material_indices(mesh);
    AKR_ASSERT(material_indices);
    ctx.parallel_for(count, 1024, [&](size_t i) noexcept {
        const auto face = faces[i];
        out[i] = material_indices[face];
    });
}
}// namespace blender_util