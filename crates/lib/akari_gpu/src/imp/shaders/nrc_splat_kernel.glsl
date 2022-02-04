#version 460
#extension GL_EXT_debug_printf: enable
layout(local_size_x = 256) in;
#define PATH_STATES_SET 0
#define NRC_SET 1
#define QUEUE_COUNTER_SET 2
#include "common.glsl"
#include "path_soa.glsl"
#include "queue.glsl"
#include "nrc.glsl"

void main(){
    uint tid = gl_GlobalInvocationID.x;
    // if(tid==0)
    // printf("%d %d %d",gl_NumWorkGroups.x * 256, scanline_per_pass * pixels_per_scanline, queue_counters[NRC_INFER_QUEUE]);
    if(tid >= scanline_per_pass * pixels_per_scanline)
        return;
    if(tid >= queue_counters[NRC_INFER_QUEUE])
        return;
    NRCInferState infer_state = NRCInferStates.states[tid];
    uint sid = uint(infer_state.idx);
    vec3 nrc_out = vec3(
        NRCInferOutputs.data[3 * tid + 0],
        NRCInferOutputs.data[3 * tid + 1],
        NRCInferOutputs.data[3 * tid + 2]);
    // printf("%d", sid);
    // printf("%f %f %f", nrc_out.x, nrc_out.y, nrc_out.z);
    vec3 beta = load_PathStates_beta(sid);
    vec3 l = load_PathStates_l(sid);
    l += infer_state.albedo.xyz * nrc_out * beta;
    store_PathStates_l(sid, l);
}