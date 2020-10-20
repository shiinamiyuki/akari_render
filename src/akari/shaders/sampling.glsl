#include "builtins.glsl"
#include "constants.glsl"

vec2 concentric_disk_sampling(const vec2 u) {
    vec2 uOffset = 2.0 * u - vec2(1, 1);
    if (uOffset.x == 0.0 && uOffset.y == 0.0)
        return vec2(0, 0);
    float theta;
    float r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = Pi4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = Pi2 - Pi4 * (uOffset.x / uOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}


vec3 cosine_hemisphere_sampling(const vec2 u) {
    vec2 uv = concentric_disk_sampling(u);
    float r = dot(uv, uv);
    float h = sqrt(max(float(0.0f), float(1.0 - r)));
    return vec3(uv.x, h, uv.y);
}
float cosine_hemisphere_pdf(float cosTheta) {
    return cosTheta * InvPi;
}
float uniform_sphere_pdf() { return 1.0 / (4.0 * Pi); }
vec3 uniform_sphere_sampling(const vec2 u) {
    float z = 1.0 - 2.0 * u[0];
    float r = sqrt(max(0.0,  1.0 - z * z));
    float phi = 2.0 * Pi * u[1];
    return vec3(r * cos(phi), r * sin(phi), z);
}
vec2 uniform_sample_triangle(const vec2 u) {
    float su0 = sqrt(u[0]);
    float b0 = 1.0 - su0;
    float b1 = u[1] * su0;
    return vec2(b0, b1);
}