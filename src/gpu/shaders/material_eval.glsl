#version 460
#extension GL_EXT_debug_printf: enable
layout(local_size_x = 256) in;
#define RAY_QUEUE_SET0 0
#define PATH_STATES_SET 1
#define MATERIAL_EVAL_SET 2
#define QUEUE_COUNTER_SET 3
#define SCENE_SET_MV 4
#define SCENE_SET_MI 5
#define SCENE_SET_NS 6
#define SCENE_SET_TC 7
#define SCENE_SET_TEX 8
#define SCENE_SET_REST 9
#define LIGHT_SET 10
#define MESH_AREA_DIST_TABLE_SET 11
#define MESH_AREA_DIST_PDF_SET 12
#define LIGHT_DIST_SET 13
#define SHADOW_QUEUE_SET 14
#define NRC_SET 15
#ifdef ENABLE_NRC
#define NRC_EXTRA_PC
#endif
#include "common.glsl"
#include "ray_queue0_soa.glsl"
#include "path_soa.glsl"
#include "material_eval_info.glsl"
#include "queue.glsl"
#include "scene.glsl"
#include "texture_eval.glsl"
#include "light.glsl"
#include "shadow_queue.glsl"
#include "material_eval_common.glsl"
#ifdef ENABLE_NRC
#include "nrc.glsl"
#endif

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint pixel = cur_scanline * pixels_per_scanline + tid;
    
    if(tid >= scanline_per_pass * pixels_per_scanline)
        return;
    if(load_PathStates_state(tid) != ALIVE)
        return;
    init_ld(pixel);
    int bounces = load_PathStates_bounce(tid);
#ifdef ENABLE_NRC
    if(nrc_training != 0){
        if(bounces == 0) {
            uint freq = (scanline_per_pass * pixels_per_scanline) / training_samples;
            bool train = (pixel + cur_sample) % freq == 0;
            NRCTrainStates.states[tid].is_training = train ? 1 : 0;
            if(!train){
                store_PathStates_state(tid, TERMINATED);
                return;
            }
        }
    }
#endif
    MaterialEvalInfo info = load_MaterialEvalInfos(tid);
    
    vec3 beta = load_PathStates_beta(tid);

    vec3 wo = info.wo;
    
  
    BSDFClosure closure;
    {
        BSDF bsdf = BSDFs.bsdfs[info.bsdf];
        closure.frame = frame_from1(info.ns);
        closure.color = evaluate_texture(bsdf.color, info.texcoords);
        vec3 emission = evaluate_texture(bsdf.emission, info.texcoords);
        if(bounces == 0){
            vec3 l = load_PathStates_l(tid);
            l += beta * emission;
            store_PathStates_l(tid, l);
        }
    }
#ifdef ENABLE_NRC
    NRCVertexInfo vertex_info;
    vertex_info.x = info.p;
    vertex_info.ns_dir = vec4(dir_to_uv(info.ns), dir_to_uv(wo));
    vertex_info.albedo = vec4(closure.color, 1.0);
    vertex_info.roughness = 1.0;
    vertex_info.metallic = 0.0;
    vertex_info.li = vec3(0);
    if(nrc_training == 0 && bounces >= 1){
        uint idx = atomicAdd(queue_counters[NRC_INFER_QUEUE], 1);
        if(idx < inference_samples) {
            vec3 x = vertex_info.x;
            vec4 ns_dir = vertex_info.ns_dir;
            vec3 albedo = vertex_info.albedo.rgb;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 0] = x.x;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 1] = x.y;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 2] = x.z;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 3] = ns_dir.x;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 4] = ns_dir.y;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 5] = ns_dir.z;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 6] = ns_dir.w;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 7] = vertex_info.roughness;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 8] = albedo.x;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 9] = albedo.y;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 10] = albedo.z;
            NRCInferInputs.data[NRC_INPUT_ROWS * idx + 11] = vertex_info.metallic;
            NRCInferState infer_state;
            infer_state.albedo = vec4(albedo, 1.0);
            infer_state.idx = tid;
            NRCInferStates.states[idx] = infer_state;
            store_PathStates_state(tid, TERMINATED);
            finalize_ld();
            return;
        }
    }
#endif
    // printf("%d", PointLights.lights.length());
    
    if(Lights.lights.length() > 0){
        float light_select_pdf;
        Light light = sample_light(ld_float2(), light_select_pdf);
        vec3 pos;
        vec3 emission;
        vec3 ng;
        float light_pdf;
        vec2 u0 = ld_float2();
        vec2 u1 = ld_float2();
        bool error = false;
        if(light.type == LIGHT_TYPE_POINT){
            int light_idx = light.index;
            pos = PointLights.lights[light_idx].pos;
            emission = evaluate_texture(PointLights.lights[light_idx].emission, vec2(0));
            vec3 wi = pos - info.p;
            float dist2 = dot(wi, wi);
            light_pdf = light_select_pdf * dist2;
            ng = normalize(-vec3(pos - info.p));
        }else if(light.type == LIGHT_TYPE_MESH){
            int light_idx = light.index;
            
            int instance_id = AreaLights.lights[light_idx].instance_id;
            // printf("light.index= %d instance_id= %d", light.index, instance_id);
            // error = true;
            SurfaceSample surface_sample = sample_mesh_triangle(u0, u1, instance_id, AreaLights.lights[light_idx].area_dist_id);
            pos = surface_sample.p;
            ng = surface_sample.ng;
            vec3 wi = pos - info.p;
            float dist2 = dot(wi, wi);
            wi /= sqrt(dist2);
            light_pdf = light_select_pdf * surface_sample.pdf / abs(dot(wi, ng));
            emission = evaluate_texture(AreaLights.lights[light_idx].emission, surface_sample.texcoords) / dist2;
        }else{
            error = true;
        }
        if(!error){
            vec3 wi = normalize(pos - info.p);
            vec3 ld = emission * beta * evaluate_bsdf(closure, wo, wi) / light_pdf * abs(dot(wi, info.ns));
            uint shadow_ray_queue_id = atomicAdd(queue_counters[SHADOW_RAY_QUEUE], 1);
            Ray shadow_ray = spawn_to_n(info.p, pos, info.ng);
            ShadowQueueItem item;
            item.ray = shadow_ray;
            item.sid = tid;
            item.ld = ld;
            store_ShadowQueue(shadow_ray_queue_id, item);
        }
    }

    vec3 wi;
    float pdf;
    vec3 f;

    if(!sample_bsdf(closure, wo, ld_float2(), wi, f, pdf)){
        store_PathStates_state(tid, TERMINATED);
        finalize_ld();
        return;
    }

    vec3 beta_mul = evaluate_bsdf(closure, wo, wi) / pdf * abs(dot(wi, info.ng));
#ifdef ENABLE_NRC
   
    vertex_info.beta = beta_mul;
    if(nrc_training != 0 && NRCTrainStates.states[tid].is_training!=0 && bounces < NRC_MAX_TRAIN_DEPTH){
        NRCTrainStates.states[tid].vertices[bounces] = vertex_info;
    }
#endif
    beta *= beta_mul;

    Ray ray = spawn_n(info.p, wi, info.ng);

    uint ray_queue_id = atomicAdd(queue_counters[RAY_QUEUE0], 1);
    RayQueueItem ray_item;
    ray_item.ray = ray;
    ray_item.sid = tid;
    store_RayQueue0(ray_queue_id, ray_item);
    store_PathStates_beta(tid, beta);
   
    store_PathStates_bounce(tid, bounces + 1);
    // store_PathStates_l(tid, info.ns * 0.5 + 0.5);
    finalize_ld();
}