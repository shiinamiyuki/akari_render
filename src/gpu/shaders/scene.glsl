#ifndef SCENE_GLSL
#define SCENE_GLSL
#include "common.glsl"

#ifndef SCENE_REST_ONLY
layout(set = SCENE_SET_MV,binding = 0)
buffer _MeshVertices{
    float vertices[];
}MeshVertices[];

layout(set = SCENE_SET_MI,binding = 0)
buffer _MeshIndices{
    int indices[];
}MeshIndices[];

layout(set = SCENE_SET_NS,binding = 0)
buffer _MeshNormals{
    float normals[];
}MeshNormals[];

layout(set = SCENE_SET_TC,binding = 0)
buffer _MeshTexcoords{
    float texcoords[];
}MeshTexcoords[];

layout(set = SCENE_SET_TEX,binding = 0)
uniform texture2D image_textures[];


#endif
#define HAS_NORMAL (1)
#define HAS_TC (2)
struct MeshInstance{
    int geom_id;
    int bsdf_id;
    int flags;
};
layout(set = SCENE_SET_REST,binding = 0)
buffer MeshInstances{
    MeshInstance instances[];
};


layout(set = SCENE_SET_REST,binding = 1)
buffer _BSDFs{
    BSDF bsdfs[];
}BSDFs;


layout(set = SCENE_SET_REST,binding = 2)
uniform sampler tex_sampler;

layout(set = SCENE_SET_REST,binding = 3)
buffer Seeds {
    uint seeds[];
};

layout(set = SCENE_SET_REST,binding = 4)
buffer _Textures {
    Texture textures[];
}Textures;

#define SOBOL_BITS 32
#define SOBOL_MAX_DIMENSION 21201
layout(set = SCENE_SET_REST,binding = 5)
buffer Sobol {
    uint sobolmat[SOBOL_BITS * SOBOL_MAX_DIMENSION];
};
struct SobolSamplerState {
    uint rotation;
    uint index;
    uint dimension;
};
uint _sobol_pixel=0;
SobolSamplerState _cur_sobol_state;
layout(set = SCENE_SET_REST,binding = 6)
buffer _SobolSamplerStates {
    SobolSamplerState states[];
}SobolSamplerStates;

#ifndef SCENE_REST_ONLY
ivec3 get_mesh_index(int geom_id, int prim_id) {
    int i = prim_id;
    return ivec3(MeshIndices[geom_id].indices[3 * i + 0],
                MeshIndices[geom_id].indices[3 * i + 1],
                MeshIndices[geom_id].indices[3 * i + 2]);
}


vec3 get_mesh_vertex(int geom_id, int i) {
    return vec3(MeshVertices[geom_id].vertices[3 * i + 0],
                MeshVertices[geom_id].vertices[3 * i + 1],
                MeshVertices[geom_id].vertices[3 * i + 2]);
}
vec3 get_mesh_normal(int geom_id, int i) {
    return vec3(MeshNormals[geom_id].normals[3 * i + 0],
                MeshNormals[geom_id].normals[3 * i + 1],
                MeshNormals[geom_id].normals[3 * i + 2]);
    
}
vec2 get_mesh_texcoords(int geom_id, int i) {
    return vec2(MeshTexcoords[geom_id].texcoords[2 * i + 0],
                MeshTexcoords[geom_id].texcoords[2 * i + 1]);
}
Triangle get_triangle(int instance_id, int i){
    int geom_id = instances[instance_id].geom_id;
    int flags = instances[instance_id].flags;
    ivec3 index = get_mesh_index(geom_id, i);
    Triangle triangle;
    vec3 v0 = get_mesh_vertex(geom_id, index.x);
    vec3 v1 = get_mesh_vertex(geom_id, index.y);
    vec3 v2 = get_mesh_vertex(geom_id, index.z);
    vec3 ng = normalize(cross(v1 - v0, v2 - v0));
    vec3 n0,n1,n2;
    if((flags & HAS_NORMAL) != 0){
        n0 = get_mesh_normal(geom_id, index.x);
        n1 = get_mesh_normal(geom_id, index.y);
        n2 = get_mesh_normal(geom_id, index.z);
    }else{
        n0 = n1 = n2 = ng;
    }
    vec2 tc0,tc1,tc2;
    if((flags & HAS_TC) != 0){
        tc0 = get_mesh_texcoords(geom_id, index.x);
        tc1 = get_mesh_texcoords(geom_id, index.y);
        tc2 = get_mesh_texcoords(geom_id, index.z);
    }else{
        tc0 = vec2(0,0);
        tc1 = vec2(1,0);
        tc2 = vec2(1,1);
    }
    Triangle trig;
    trig.v[0] = v0;
    trig.v[1] = v1;
    trig.v[2] = v2;
    trig.ns[0] = n0;
    trig.ns[1] = n1;
    trig.ns[2] = n2;
    trig.ng = ng;
    trig.tc[0] = tc0;
    trig.tc[1] = tc1;
    trig.tc[2] = tc2;
    return trig;
}

#endif
void init_rng(uint pixel){
    _seed_id = pixel;
    _seed = seeds[_seed_id];
}
void finalize_rng(){
    seeds[_seed_id] = _seed;
}
void init_ld(uint pixel){
    init_rng(pixel);
    _sobol_pixel = pixel;
    _cur_sobol_state = SobolSamplerStates.states[pixel];
}
void start_ld_next_sample() {
    _cur_sobol_state.index += 1;
    _cur_sobol_state.dimension = 0;
}
void finalize_ld() {
    SobolSamplerStates.states[_sobol_pixel] = _cur_sobol_state;
    finalize_rng();
}

uint cmj_hash_simple(uint i, uint p){
    i = (i ^ 61) ^ p;
    i += i << 3;
    i ^= i >> 4;
    i *= uint(0x27d4eb2d);
    return i;
}
float _sobol_imp(uint dim,uint i, uint rng){
    uint res = 0;
    int j =0;
    while (i>0){
        if((i&1)!=0){
            res ^= sobolmat[dim * SOBOL_BITS + j];
        }
        j += 1;
        i >>= 1;
    }
    float r = res / float(uint(0xffffffff));
    uint tmg_rng = cmj_hash_simple(i, rng);
    float shift =  tmg_rng / float(uint(0xffffffff));
    return r + shift - floor(r + shift);
}
float ld_float() {
    float r = _sobol_imp(_cur_sobol_state.dimension, _cur_sobol_state.index, _cur_sobol_state.rotation);
    _cur_sobol_state.dimension += 1;
    return r;
}
vec2 ld_float2() {
    return vec2(ld_float(), ld_float());
}


#endif