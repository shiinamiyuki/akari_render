#include "builtins.glsl"
struct Light {
    vec3 pos;
    vec3 color;
};

void swap(inout float x, inout float y){
    float t = x;
    x = y;
    y = t;
}
vec3 mul(mat3x3 m, vec3 v){
    return transpose(m) * v;
} 