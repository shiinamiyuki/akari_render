#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_debug_printf : enable

#define SHADOW_QUEUE_SET 1
#define PATH_STATES_SET 2
#define QUEUE_COUNTER_SET 3
#define NRC_SET 4
layout(set = 0, binding = 0) uniform accelerationStructureNV topLevelAS;
#include "common.glsl"
#include "path_soa.glsl"
#include "shadow_queue.glsl"
#include "queue.glsl"
#ifdef ENABLE_NRC
#include "nrc.glsl"
#endif
layout(location = 0) rayPayloadNV bool occluded;

void main() 
{
    uint tid = gl_LaunchIDNV.x;
    if(tid >= queue_counters[SHADOW_RAY_QUEUE])
        return;
    
    ShadowQueueItem item = load_ShadowQueue(tid);
    Ray ray = item.ray;

    uint cullMask = 0xff;
    uint rayFlags = gl_RayFlagsOpaqueNV | gl_RayFlagsTerminateOnFirstHitNV;
    float tmin = ray.tmin;
    float tmax = min(ray.tmax, 10000.0);
    occluded = true;
    traceNV(topLevelAS, rayFlags, cullMask, 0 /*sbtRecordOffset*/, 0/*sbtRecordStride*/, 0 /*missIndex*/, ray.o, tmin, ray.d, tmax, 0 /*payload*/);
    
    if(!occluded){
        vec3 l = load_PathStates_l(item.sid);
        l += item.ld;
        store_PathStates_l(item.sid, l);
    #ifdef ENABLE_NRC
        if(NRCTrainStates.states[item.sid].is_training != 0){
            int bounces = load_PathStates_bounce(item.sid);
            
            if(bounces <= NRC_MAX_TRAIN_DEPTH){
                vec3 beta = vec3(1);
                for(int i= 0;i < bounces - 1;i++){
                    beta *= NRCTrainStates.states[item.sid].vertices[i].beta;
                }
                // if(bounces==3)
                    // printf("%f %f %f", beta.x, beta.y, beta.z);
                vec3 ld = safe_div(item.ld, beta);
                NRCTrainStates.states[item.sid].vertices[bounces - 1].li += ld;
            }
        }

    #endif
    }

}