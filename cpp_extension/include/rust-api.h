#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

namespace akari {
namespace rt {

static const uint32_t EMBREE_HIT_ACCEPT = 1;

static const uint32_t EMBREE_HIT_NONE = 0;

static const uint32_t EMBREE_HIT_TERMINATE = 2;

struct EmbreeGeometry {
    void *_0;
};

struct EmbreeMeshBuildArgs {
    const float *vertices;
    size_t num_vertices;
    const uint32_t *indices;
    size_t num_indices;
};

struct EmbreeTlas {
    void *_0;
};

struct EmbreeInstanceBuildArgs {
    EmbreeGeometry geometry;
    float transform[12];
};

struct EmbreeTlasBuildArgs {
    const EmbreeInstanceBuildArgs *instances;
    size_t num_instances;
};

struct EmbreeRay {
    float origin[3];
    float dir[3];
    float time;
    float t_near;
    float t_far;
};

struct EmbreeHit {
    float t;
    float u;
    float v;
    uint32_t prim_id;
    uint32_t geom_id;
};

struct EmbreeTriangleHitCallback {
    void *data;
    uint32_t (*on_hit)(void *, const EmbreeRay *, const EmbreeHit *);
};

struct EmbreeRayQuery {
    EmbreeRay ray;
    bool terminate_on_first_hit;
    EmbreeHit hit;
    EmbreeTriangleHitCallback on_triangle_hit;
};

struct EmbreeApi {
    void *data;
    void (*dtor)(EmbreeApi *);
    EmbreeGeometry (*build_mesh)(const EmbreeApi *, const EmbreeMeshBuildArgs *);
    void (*destroy_mesh)(const EmbreeApi *, EmbreeGeometry);
    EmbreeTlas (*build_tlas)(const EmbreeApi *, const EmbreeTlasBuildArgs *);
    void (*destroy_tlas)(const EmbreeApi *, EmbreeTlas);
    bool (*traverse)(const EmbreeTlas *, EmbreeRayQuery *);
};

extern "C" {

extern EmbreeApi embree_api_create();

}// extern "C"
}// namespace rt
namespace image {
enum PixelFormat : uint8_t {
    R8,
    RGBA8,
    RF32,
    RGBF32,
    RGBAF32,
};
struct Image {
    uint8_t *data = nullptr;
    size_t width = 0;
    size_t height = 0;
    PixelFormat format{};
};
struct ImageApi {
    Image *(*read)(const char *path, PixelFormat format);
    bool (*write)(const char *path, const Image *image);
    void (*destroy_image)(Image *);
};
extern "C" {
extern ImageApi image_api_create();
}
}// namespace image
}// namespace akari
