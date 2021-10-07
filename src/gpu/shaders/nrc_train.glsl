#version 460
#extension GL_EXT_debug_printf: enable
layout(local_size_x = 256) in;
#define PATH_STATES_SET 0
#define NRC_SET 1
#define QUEUE_COUNTER_SET 2
#define NRC_EXTRA_PC
#include "common.glsl"
#include "path_soa.glsl"
#include "queue.glsl"
#include "nrc.glsl"

void main(){
    uint tid = gl_GlobalInvocationID.x;
    // printf("fuck");
    if(tid >= scanline_per_pass * pixels_per_scanline)
        return;
    if(NRCTrainStates.states[tid].is_training!=0){
        int bounces = load_PathStates_bounce(tid);
        uint idx = atomicAdd(queue_counters[NRC_TRAIN_QUEUE], min(bounces, NRC_MAX_TRAIN_DEPTH));
        if(idx >= training_samples)
            return;
        vec3 l[NRC_MAX_TRAIN_DEPTH];
        vec3 beta[NRC_MAX_TRAIN_DEPTH];
        for(int i=0;i<NRC_MAX_TRAIN_DEPTH;i++){
            beta[i] = vec3(1);
            l[i] = vec3(0);
        }
        for(int bounce = 0; bounce <  min(bounces, NRC_MAX_TRAIN_DEPTH); bounce++){
            for(int i=0; i<= bounce;i++){
                l[i] += beta[i] * NRCTrainStates.states[tid].vertices[bounce].li;
            }
            for(int i=0; i<= bounce;i++){
                beta[i] *= NRCTrainStates.states[tid].vertices[bounce].beta;
            }
        }
        for(int bounce = 0; bounce <  min(bounces, NRC_MAX_TRAIN_DEPTH); bounce++){
            vec3 x = NRCTrainStates.states[tid].vertices[bounce].x;
            vec4 ns_dir = NRCTrainStates.states[tid].vertices[bounce].ns_dir;
            vec3 albedo = NRCTrainStates.states[tid].vertices[bounce].albedo.rgb;
            vec3 irradiance = max(vec3(0), safe_div(l[bounce], albedo));
            // if(bounce>0)
            //     printf("%f %f %f", l[bounce].x, l[bounce].y, l[bounce].z);
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 0] = x.x;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 1] = x.y;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 2] = x.z;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 3] = ns_dir.x;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 4] = ns_dir.y;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 5] = ns_dir.z;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 6] = ns_dir.w;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 7] = NRCTrainStates.states[tid].vertices[bounce].roughness;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 8] = albedo.x;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 9] = albedo.y;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 10] = albedo.z;
            NRCTrainInputs.data[NRC_INPUT_ROWS * (idx + bounce) + 11] = NRCTrainStates.states[tid].vertices[bounce].metallic;
            NRCTrainTargets.data[3 * (idx + bounce) + 0] = irradiance.x;
            NRCTrainTargets.data[3 * (idx + bounce) + 1] = irradiance.y;
            NRCTrainTargets.data[3 * (idx + bounce) + 2] = irradiance.z;

        }
    }
}