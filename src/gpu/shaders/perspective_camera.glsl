#version 460
#extension GL_EXT_debug_printf : enable
#include "camera_common.glsl"
#define SCENE_REST_ONLY
#define SCENE_SET_REST 2
#include "scene.glsl"
layout(local_size_x = 256) in;
layout(set=0, binding = 0)
uniform PerspectiveCamera {
    Transform c2w;
    Transform r2c;
    uvec2 resolution;
};

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint pixel = cur_scanline * pixels_per_scanline + tid;
    if(pixel >= pixels_per_scanline * total_scanlines)
        return;
    if(tid >= scanline_per_pass * pixels_per_scanline)
        return;
    init_ld(pixel);
    start_ld_next_sample();
    uint px = pixel % pixels_per_scanline;
    uint py = pixel / pixels_per_scanline;
    vec2 fpixel = vec2(px, py) + ld_float2();
    vec2 p = transform_point(r2c, vec3(fpixel.x,fpixel.y, 0.0)).xy;
    Ray ray;
    ray.tmin = 0.001;
    ray.tmax = 1e9;
    ray.o = vec3(0);
    ray.d = normalize(vec3(p.x,p.y,0.0) - vec3(0,0,1));
    ray.o = transform_point(c2w, ray.o);
    ray.d = transform_vec(c2w, ray.d);
    RayQueueItem item;
    item.sid = tid;
    item.ray = ray;
    store_RayQueue0(tid, item);
    finalize_ld();
}