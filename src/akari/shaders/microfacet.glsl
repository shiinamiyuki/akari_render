
#include "builtins.glsl"
#include "geometry.glsl"
#include "constants.glsl"
const int MicrofacetGGX = 0;
const int MicrofacetBeckmann = 1;
const int MicrofacetPhong = 2;

float BeckmannD(float alpha, const vec3 m) {
    if (m.y <= 0.0)
        return 0.0;
    float c = cos2_theta(m);
    float t = tan2_theta(m);
    float a2 = alpha * alpha;
    return exp(-t / a2) / (Pi * a2 * c * c);
}

float BeckmannG1(float alpha, const vec3 v, const vec3 m) {
    if (dot(v, m) * v.y <= 0.0) {
        return 0.0;
    }
    float a = 1.0 / (alpha * tan_theta(v));
    if (a < 1.6) {
        return (3.535 * a + 2.181 * a * a) / (1.0 + 2.276 * a + 2.577 * a * a);
    } else {
        return 1.0;
    }
}
float PhongG1(float alpha, const vec3 v, const vec3 m) {
    if (dot(v, m) * v.y <= 0.0) {
        return 0.0;
    }
    float a = sqrt(0.5 * alpha + 1.0) / tan_theta(v);
    if (a < 1.6) {
        return (3.535 * a + 2.181 * a * a) / (1.0 + 2.276 * a + 2.577 * a * a);
    } else {
        return 1.0;
    }
}

float PhongD(float alpha, const vec3 m) {
    if (m.y <= 0.0)
        return 0.0;
    return (alpha + 2.0) / (2.0 * Pi) * pow(m.y, alpha);
}

float GGX_D(float alpha, const vec3 m) {
    if (m.y <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    float c2 = cos2_theta(m);
    float t2 = tan2_theta(m);
    float at = (a2 + t2);
    return a2 / (Pi * c2 * c2 * at * at);
}

float GGX_G1(float alpha, const vec3 v, const vec3 m) {
    if (dot(v, m) * v.y <= 0.0) {
        return 0.0;
    }
    return 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * tan2_theta(m)));
}

struct MicrofacetModel {
    int type;
    float alpha;
};

MicrofacetModel microfacet_new(int type, float roughness){
    float alpha;
    if (type == MicrofacetPhong) {
        alpha = 2.0 / (roughness * roughness) - 2.0;
    } else {
        alpha = roughness;
    }
    return MicrofacetModel(type, alpha);
}
 float microfacet_D(const MicrofacetModel model, const vec3 m) {
    int type = model.type;
    float alpha = model.alpha;
    switch (type) {
        case MicrofacetBeckmann:{
            return BeckmannD(alpha, m);
        }
        case MicrofacetPhong:{
            return PhongD(alpha, m);
        }

        case MicrofacetGGX:{
            return GGX_D(alpha, m);
        }
    }

    return 0.0;
}
 float microfacet_G1(const MicrofacetModel model,const vec3 v, const vec3 m)  {
    int type = model.type;
    float alpha = model.alpha;
     switch (type) {
        case MicrofacetBeckmann:{
            return BeckmannG1(alpha, v, m);
        }
        case MicrofacetPhong:{
            return PhongG1(alpha, v, m);
        }
        case MicrofacetGGX:{
            return GGX_G1(alpha, v, m);
        }
    }
    return 0.0;
}
 float microfacet_G(const MicrofacetModel model, const vec3 i, const vec3 o, const vec3 m)  {
    return microfacet_G1(model,i, m) * microfacet_G1(model,o, m);
}
 vec3 microfacet_sample_wh(const MicrofacetModel model, const vec3 wo, const vec2 u)  {
    int type = model.type;
    float alpha = model.alpha;
    float phi = 2.0 * Pi * u.y;
    float cosTheta = 0.0;
    switch (type) {
        case MicrofacetBeckmann: {
            float t2 = -alpha * alpha * log(1.0 - u.x);
            cosTheta = 1.0f / sqrt(1.0 + t2);
            break;
        }
        case MicrofacetPhong: {
            cosTheta = pow(u.x, float(1.0 / (alpha + 2.0)));
            break;
        }
        case MicrofacetGGX: {
            float t2 = alpha * alpha * u.x / (1.0 - u.x);
            cosTheta = 1.0 / sqrt(1.0 + t2);
            break;
        }
    }
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    vec3 wh = vec3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
    if (!same_hemisphere(wo, wh))
        wh = -wh;
    return wh;
}
 float microfacet_evaluate_pdf(const MicrofacetModel m, const vec3 wh)  {
    return microfacet_D(m, wh) * abs_cos_theta(wh);
}
