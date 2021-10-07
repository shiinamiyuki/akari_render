#version 460
layout(local_size_x = 256) in;
#define PATH_STATES_SET 0
#define QUEUE_COUNTER_SET 1
#include "common.glsl"
#include "queue.glsl"
#include "path_soa.glsl"


void main() {
    if(gl_GlobalInvocationID.x == 0) {
        queue_counters[RAY_QUEUE0] = (min(cur_scanline + scanline_per_pass, total_scanlines) - cur_scanline) * pixels_per_scanline;
    }
    uint tid = gl_GlobalInvocationID.x;
    if(tid >= scanline_per_pass * pixels_per_scanline)
        return;
    uint pixel = cur_scanline * pixels_per_scanline + tid;
    
    PathState state;
    state.beta = vec3(1);
    state.l = vec3(0);
    state.bounce = 0;
    state.pixel = pixel;
    if(pixel >= pixels_per_scanline * total_scanlines){
        state.state = DISCARDED;
    }else{
        state.state = ALIVE;
    }
    store_PathStates(tid, state);    
}