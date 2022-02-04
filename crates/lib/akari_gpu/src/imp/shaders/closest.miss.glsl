#version 460
#extension GL_NV_ray_tracing : require
struct HitInfo {
    float t;
    vec2 bc;
    int prim_id;
    int inst_id;
};
layout(location = 0) rayPayloadInNV HitInfo hit_info;
#extension GL_EXT_debug_printf : enable
void main()
{
    // debugPrintfEXT("miss");
    hit_info.t = -1.0;
}