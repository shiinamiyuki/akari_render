#version 460
#extension GL_NV_ray_tracing : require
struct HitInfo {
    float t;
    vec2 bc;
    int prim_id;
    int inst_id;
};
layout(location = 0) rayPayloadInNV HitInfo hit_info;
hitAttributeNV vec3 attribs;
#extension GL_EXT_debug_printf : enable
void main()
{
    // const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    // hitValue = barycentrics;
    // debugPrintfEXT("hit");
    // hitValue = vec3(1,0,0);
    hit_info.bc = attribs.xy;
    hit_info.t = gl_HitTNV;
    hit_info.prim_id = gl_PrimitiveID;
    hit_info.inst_id = gl_InstanceCustomIndexNV;
}