#ifndef COMMON_GLSL
#define COMMON_GLSL
#define printf debugPrintfEXT
#extension GL_EXT_nonuniform_qualifier: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64  : enable
#define TEXTURE_FLOAT 0 
#define TEXTURE_SPECTRUM 1
#define TEXTURE_FLOAT_IMAGE 2
#define TEXTURE_SPECTRUM_IMAGE 3

struct Texture {
    int type;
    int image_tex_id;
    vec4 data;
};

struct Ray {
    vec3 o;
    vec3 d;
    float tmin;
    float tmax;
};
float origin()      { return 1.0 / 32.0; }
float float_scale() { return 1.0 / 65536.0; }
float int_scale()   { return 256.0; }

// Normal points outward for rays exiting the surface, else is flipped.
vec3 offset_ray(const vec3 p, const vec3 n)
{
  ivec3 of_i = ivec3(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

  vec3 p_i = vec3( 
      intBitsToFloat(floatBitsToInt(p.x)+((p.x < 0.0) ? -of_i.x : of_i.x)),
      intBitsToFloat(floatBitsToInt(p.y)+((p.y < 0.0) ? -of_i.y : of_i.y)),
      intBitsToFloat(floatBitsToInt(p.z)+((p.z < 0.0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin() ? p.x+ float_scale()*n.x : p_i.x,
                abs(p.y) < origin() ? p.y+ float_scale()*n.y : p_i.y,
                abs(p.z) < origin() ? p.z+ float_scale()*n.z : p_i.z);
}
Ray spawn(vec3 o, vec3 d){
    Ray ray;
    ray.o = o;
    ray.d = d;
    ray.tmin = 0.0;
    ray.tmax = 1e9;
    return ray;
}

Ray spawn_n(vec3 o, vec3 d, vec3 n){
    Ray ray;
    ray.o = offset_ray(o, n);
    ray.d = d;
    ray.tmin = 0.0;
    ray.tmax = 1e9;
    return ray;
}

Ray spawn_to_n(vec3 p1, vec3 p2, vec3 n){
    Ray ray;
    ray.o = offset_ray(p1, n);
    ray.d = normalize(p2 - p1);
    ray.tmin = 0.0;
    ray.tmax = length(p2 - p1) * 0.998;
    return ray;
}
struct Intersection {
    int prim_id;
    int geom_id;
    vec2 uv;
    vec2 texcoords;
    vec3 ng;
    vec3 ns;
};

struct BSDF {
    Texture color;
    Texture metallic;
    Texture roughness;
    Texture emission;
};

struct Transform {
    mat4 m4;
    mat4 inv_m4;
};

struct ShadowQueueItem {
    Ray ray;
    vec3 ld;
    uint sid;
};

#define DISCARDED 0 
#define TERMINATED 1
#define ALIVE 2
struct PathState {
    int state;
    int bounce;
    vec3 beta;
    vec3 l;
    uint pixel;
};
struct RayQueueItem {
    Ray ray;
    uint sid;
};
struct MaterialEvalInfo {
    vec3 wo;
    vec3 p;
    vec3 ng;
    vec3 ns;
    vec2 texcoords;
    int bsdf;
};
vec3 transform_point(Transform t, vec3 p){
    vec4 q = vec4(p, 1.0);
    q = t.m4 * q;
    return q.xyz / q.w;
}
vec3 transform_vec(Transform t, vec3 p){
    return mat3(t.m4) * p;
}
#define PI (3.1415926535)
#define FRAC_1_PI (1.0 / PI)
#define FRAC_PI_2 (PI / 2.0)
#define FRAC_PI_4 (PI / 4.0)
vec2 uniform_sample_disk_polar(vec2 u){
    float r = sqrt(u.x);
    float theta = 2.0 * PI * u.y;
    return r * vec2(cos(theta), sin(theta));
}
vec2 concentric_sample_disk(vec2 u){
    vec2 u_offset= 2.0 * u - vec2(1.0, 1.0);
    if (u_offset.x == 0.0 && u_offset.y == 0.0) {
        return vec2(0);
    }

    float theta, r;
    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = FRAC_PI_4 * (u_offset.y / u_offset.x);

    } else {
        r = u_offset.y;
        theta = FRAC_PI_2 - FRAC_PI_4 * (u_offset.x / u_offset.y);
    }
    
    return r * vec2(cos(theta), sin(theta));
}
vec3 cosine_hemisphere_samping(vec2 u){
    vec2 uv = concentric_sample_disk(u);
    float r = dot(uv, uv);
    float h = sqrt(1.0 - r);
    return vec3(uv.x, h, uv.y);
}

vec2 uniform_sample_triangle(vec2 u) {
    float su0 = sqrt(u.x);
    float b0 = 1.0 - su0;
    float b1 = u.y * su0;
    return vec2(b0, b1);
}
vec2 dir_to_spherical(vec3 v){
    float theta = acos(v.y);
    float phi = atan(v.z, v.x) + PI;
    return vec2(theta, phi);
}
vec2 spherical_to_uv(vec2 v){
    return vec2(v.x / PI , v.y / (2.0 * PI));
}
vec2 dir_to_uv(vec3 v){
    return spherical_to_uv(dir_to_spherical(v));
}
struct Frame {
    vec3 N,T, B;
};

Frame frame_from1(vec3 n){
    Frame self;
    vec3 t;
    if(abs(n.x)>abs(n.y)){
        t = normalize(vec3(-n.z, 0, n.x));
    }else{
        t = normalize(vec3(0, n.z, -n.y));
    }
    self.N = n;
    self.T = t;
    self.B = normalize(cross(self.N, self.T));
    return self;
}
vec3 to_local(const Frame f, vec3 w){
    return vec3(dot(f.T, w), dot(f.N, w), dot(f.B, w));
}

vec3 to_world(const Frame f, vec3 w){
    return f.T * w.x + f.N * w.y + f.B * w.z;
}

vec3 lerp3(vec2 uv, vec3 v0, vec3 v1, vec3 v2){
    return (1.0 - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
}
vec2 lerp3(vec2 uv, vec2 v0, vec2 v1, vec2 v2){
    return (1.0 - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
}

struct Triangle {
    vec3 v[3];
    vec3 ns[3];
    vec2 tc[3];
    vec3 ng;
};

uint _seed_id;
uint _seed;

uint rng_uint() {
    _seed = _seed * uint(1103515245) + uint(12345);
    return _seed;
}
float rng_float() {
    uint i = rng_uint();
    return float(i) / float(uint(0xffffffff));
}
vec2 rng_float2() {
    return vec2(rng_float(), rng_float());
}

vec3 safe_div(vec3 a, vec3 b){
    return a / vec3(
        b.x == 0.0 ? 1.0 : b.x,
        b.y == 0.0 ? 1.0 : b.y,
        b.z == 0.0 ? 1.0 : b.z
    );
}

layout (push_constant ) uniform PushConstants {
    uint cur_scanline;
    uint scanline_per_pass;
    uint pixels_per_scanline;
    uint total_scanlines;
#ifdef NRC_EXTRA_PC
    uint cur_sample;
    uint nrc_training;
    uint training_samples;
    uint inference_samples;
#endif
};
#endif