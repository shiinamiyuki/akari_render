#version 460
#extension GL_EXT_debug_printf : enable
layout(local_size_x = 256) in;
#define PATH_STATES_SET 0

#include "common.glsl"
#include "path_soa.glsl"

layout(set = 1, binding=0)
buffer _Film {
    vec4 pixels[];
}Film;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if(tid >= scanline_per_pass * pixels_per_scanline)
        return;
    uint state = load_PathStates_state(tid);
    if(state == DISCARDED)
        return;
    vec3 l = load_PathStates_l(tid);
    l.x = isnan(l.x) ? 0.0 : l.x;
    l.y = isnan(l.y) ? 0.0 : l.y;
    l.z = isnan(l.z) ? 0.0 : l.z;
    l = clamp(l, 0.0, 100.0);
    uint pixel = load_PathStates_pixel(tid);
    if(pixel < pixels_per_scanline * total_scanlines) {
        Film.pixels[pixel] += vec4(l, 1.0);
    }
}
