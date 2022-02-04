
#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_debug_printf : enable
#define RAY_QUEUE_SET0 1
#define PATH_STATES_SET 2
#define SCENE_SET_MV 4
#define SCENE_SET_MI 5
#define SCENE_SET_NS 6
#define SCENE_SET_TC 7
#define SCENE_SET_TEX 8
#define SCENE_SET_REST 9
#define MATERIAL_EVAL_SET 10
#define QUEUE_COUNTER_SET 11
#include "scene.glsl"
#include "ray_queue0_soa.glsl"
#include "path_soa.glsl"
#include "material_eval_info.glsl"
#include "queue.glsl"
layout(set = 0, binding = 0) uniform accelerationStructureNV topLevelAS;
struct HitInfo {
    float t;
    vec2 bc;
    int prim_id;
    int inst_id;
};
layout(location = 0) rayPayloadNV HitInfo hit_info;

layout(set = 3, binding=0)
buffer _Film {
    vec4 pixels[];
}Film;

void main() 
{
    uint tid = gl_LaunchIDNV.x;
    if(tid >= queue_counters[RAY_QUEUE0])
        return;
    //printf("%d %d",scanline_per_pass, pixels_per_scanline);
    Ray ray = load_RayQueue0_ray(tid);
    uint sid = load_RayQueue0_sid(tid);
    uint cullMask = 0xff;
    uint rayFlags = gl_RayFlagsOpaqueNV;
    float tmin = ray.tmin;
    float tmax = min(ray.tmax, 10000.0);
    hit_info.t = -1.0;
    traceNV(topLevelAS, rayFlags, cullMask, 0 /*sbtRecordOffset*/, 0/*sbtRecordStride*/, 0 /*missIndex*/, ray.o, tmin, ray.d, tmax, 0 /*payload*/);

    // vec3 beta = load_PathStates_beta(sid);
    // vec3 l = load_PathStates_l(sid);
    if(hit_info.t >= 0.0){
        Triangle trig = get_triangle(hit_info.inst_id, hit_info.prim_id);
        MaterialEvalInfo info;
        info.ng = trig.ng;
        info.ns = lerp3(hit_info.bc, trig.ns[0], trig.ns[1], trig.ns[2]);
        info.texcoords = lerp3(hit_info.bc, trig.tc[0], trig.tc[1], trig.tc[2]);;//hit_info.bc;// TODO
        info.p = lerp3(hit_info.bc, trig.v[0], trig.v[1], trig.v[2]);
        info.wo = -ray.d;
        info.bsdf = instances[hit_info.inst_id].bsdf_id;
        
        store_MaterialEvalInfos(sid, info);
        // Film.pixels[pixel] += vec4(vec3(hit_info.bc, 1.0), 1.0);
    }else{
        // Film.pixels[pixel] += vec4(0,0,0,1);
        // l += beta * vec3(1);
        store_PathStates_state(sid, TERMINATED);
        // store_PathStates_l(sid, l);
    }
}