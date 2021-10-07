#version 460
#extension GL_EXT_debug_printf: enable
layout(local_size_x = 256) in;
#define NRC_SET 0
#define QUEUE_COUNTER_SET 1
#define NRC_EXTRA_PC
#include "common.glsl"
#include "queue.glsl"
#include "nrc.glsl"


void main(){
    uint tid = gl_GlobalInvocationID.x;
    uint count = queue_counters[NRC_TRAIN_QUEUE];
    if(count>=training_samples)
        return;
    uint remaining = training_samples - count;
    uint nthreads = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    _seed = cur_sample + tid * 197;
    for(uint i = tid; i < remaining; i+= nthreads){
        uint idx = rng_uint() % count;
        for(int j=0;j< NRC_INPUT_ROWS;j++){
            NRCTrainInputs.data[(count + i) * NRC_INPUT_ROWS + j] =  NRCTrainInputs.data[idx * NRC_INPUT_ROWS + j]; 
        }
        for(int j=0;j<3;j++){
            NRCTrainTargets.data[(count + i) * NRC_INPUT_ROWS + j] =  NRCTrainTargets.data[idx * NRC_INPUT_ROWS + j]; 
        }
    }
}