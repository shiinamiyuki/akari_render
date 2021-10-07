struct BSDFClosure {
    Frame frame;
    vec3 color;
    float roughness;
    float metallic;
};
bool same_hemisphere(vec3 wo, vec3 wi){
    return wi.y * wo.y > 0.0;
}
float cos_theta(vec3 w){
    return w.y;
}
float abs_cos_theta(vec3 w){
    return abs(w.y);
}
float cos2_theta(vec3 w){
    return w.y * w.y;
}
float sin2_theta(vec3 w){
    return 1.0 - cos2_theta(w);
}
float sin_theta(vec3 w){
    return sqrt(max(0.0, sin2_theta(w)));
}
float tan2_theta(vec3 w){
    return sin2_theta(w) / cos2_theta(w);
}
float cos_phi(vec3 w){
    float s = sin_theta(w);
    if (s == 0.0){
        return 1.0;
    }else{
        return clamp(w.x / s, -1.0, 1.0);
    }
}
float sin_phi(vec3 w){
    float s = sin_theta(w);
    if (s == 0.0){
        return 0.0;
    }else{
        return clamp(w.z / s, -1.0, 1.0);
    }
}
float sqr(float x){
    return x*x;
}
struct GGXDistribution {
    float ax, ay;
};

float GGX_D(const GGXDistribution dist, vec3 wm){
    float tan2 = tan2_theta(wm);
    if(isinf(tan2)){
        return 0.0;
    }
    float cos4_theta = sqr(cos2_theta(wm));
    if(cos4_theta < 1e-16){
        return 0.0;
    }
    float e = tan2 * (sqr(cos_phi(wm) / dist.ax) + sqr(sin_phi(wm) / dist.ay));
    return 1 / (PI * dist.ax * dist.ay * cos4_theta * sqr(1.0 + e));
}
float GGX_Lambda(const GGXDistribution dist, vec3 wm){
    float tan2 = tan2_theta(wm);
    if(isinf(tan2)){
        return 0.0;
    }
    float alpha2 = sqr(cos_phi(wm) * dist.ax) + sqr(sin_phi(wm) * dist.ay);
    return 0.5 * (sqrt(1.0 + alpha2 * tan2) - 1.0);
}
float GGX_D1(const GGXDistribution dist, vec3 w){
    return 1.0 / (1.0 + GGX_Lambda(dist, w));
}

float GGX_G(const GGXDistribution dist, vec3 wo, vec3 wi){
     return 1.0 / (1.0 + GGX_Lambda(dist, wo) + GGX_Lambda(dist, wi));
}
vec3 GGX_sample_wm(const GGXDistribution dist, vec3 w, vec2 u){
    w = w.xzy;
    vec3 wh = normalize(vec3(dist.ax * w.x, dist.ay * w.y, w.z));
    vec3 t1 = (wh.z < 0.9999999) ? normalize(cross(vec3(0, 0, 1), wh)) : vec3(1, 0, 0);
    vec3 t2 = cross(wh, t1);
    vec2 p = uniform_sample_disk_polar(u);

    float h = sqrt(1.0 - sqr(p.x));
    p.y = mix(h, p.y, 0.5 * (1.0 + wh.z));

    float pz = sqrt(max(0.0, 1.0 - dot(p, p)));
    vec3 nh = p.x * t1 + p.y * t2 + pz * wh;

    return normalize(
            vec3(dist.ax * nh.x, dist.ay * nh.y, max(1e-6, nh.z))).xzy;

}

vec2 mul(vec2 a, vec2 b){
    return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
} 
vec2 csqr(vec2 a){
    return mul(a, a);
}
vec2 div(vec2 a, vec2 b){
    return vec2(((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y)),((a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y)));
} 
vec2 csqrt(vec2 a){
    float n = length(a);
    return vec2(sqrt((n + a.x) * 0.5), sign(a.y) * sqrt((n - a.x) * 0.5));
}

float fr_complex(float cosTheta_i, vec2 eta){
    cosTheta_i = clamp(cosTheta_i, 0.0, 1.0);
    // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    float sin2Theta_i = 1 - sqrt(cosTheta_i);
    vec2 sin2Theta_t = sin2Theta_i / csqr(eta);
    vec2 cosTheta_t = csqrt(vec2(1.0) - sin2Theta_t);

    vec2 r_parl = div(eta *  cosTheta_i - cosTheta_t, eta *  cosTheta_i + cosTheta_t);
    vec2 r_perp = div(cosTheta_i - mul(eta, cosTheta_t), cosTheta_i + mul(eta,  cosTheta_t));
    return (length(r_parl) + length(r_perp)) * 0.5;
}

vec3 fr_complex(float cosTheta_i, vec3 eta, vec3 k){
    return vec3(
        fr_complex(cosTheta_i, vec2(eta.x,k.x)),
        fr_complex(cosTheta_i, vec2(eta.y,k.y)),
        fr_complex(cosTheta_i, vec2(eta.z,k.z))
    );
}

vec3 evaluate_GGX(const GGXDistribution dist, vec3 wo, vec3 wi, vec3 eta, vec3 k){
    if(!same_hemisphere(wo,wi))
        return vec3(0);
    vec3 wm = wo + wi;
    if(all(equal(wm, vec3(0))))
        return vec3(0);
    
    vec3 f = fr_complex(abs(dot(wo, wm)), eta, k);
    return GGX_D(dist, wm) * GGX_G(dist,wo, wi) * f / (4.0 * abs_cos_theta(wo) * abs_cos_theta(wi));
}

vec3 evaluate_bsdf_local(const BSDFClosure closure, const vec3 wo, const vec3 wi){
    if(wo.y * wi.y < 0.0){
        return vec3(0);
    }
    return closure.color / PI;
}
vec3 evaluate_bsdf(const BSDFClosure closure, const vec3 wo, const vec3 wi){
    return evaluate_bsdf_local(closure, to_local(closure.frame, wo), to_local(closure.frame, wi));
}
bool sample_bsdf_local(const BSDFClosure closure, const vec3 wo, const vec2 u, inout vec3 wi, inout vec3 f, inout float pdf){
    wi = cosine_hemisphere_samping(u);
    if(wi.y * wo.y< 0.0){
        wi.y = -wi.y;
    }
    pdf = abs(wi.y) / PI;
    f = evaluate_bsdf_local(closure, wo, wi);
    return true;
}
bool sample_bsdf(const BSDFClosure closure, const vec3 wo, const vec2 u, inout vec3 wi, inout vec3 f, inout float pdf){
    bool suc =  sample_bsdf_local(closure, to_local(closure.frame, wo), u, wi, f, pdf);
    if(!suc)
        return false;
    wi = to_world(closure.frame, wi);
    return true;
}

Light sample_light(const vec2 u, inout float pdf){
    int idx = int(u.x * float(LightDistributionAliasTable.table.length()));
    idx = min(idx, LightDistributionAliasTable.table.length() - 1);
    AliasTableEntry entry = LightDistributionAliasTable.table[idx];
    if(u.y >= entry.t) {
        idx = entry.j;
    }
    pdf = LightDistributionPdf.pdf[idx];
    return Lights.lights[idx];
}

int sample_mesh_prim(const vec2 u, int area_dist_id, inout float pdf){
    int idx = int(u.x * float(MeshAreaDistributionAliasTable[area_dist_id].table.length()));
    idx = min(idx, MeshAreaDistributionAliasTable[area_dist_id].table.length() - 1);
    AliasTableEntry entry = MeshAreaDistributionAliasTable[area_dist_id].table[idx];
    if(u.y >= entry.t) {
        idx = entry.j;
    }
    pdf = MeshAreaDistributionPdf[area_dist_id].pdf[idx];
    return idx;
}

struct SurfaceSample {
    vec3 p;
    vec3 ng;
    vec2 texcoords;
    float pdf;
};

SurfaceSample sample_mesh_triangle(const vec2 u0, const vec2 u1, int instance_id,int area_dist_id){
    float pdf_prim;
    int prim_id = sample_mesh_prim(u0, area_dist_id, pdf_prim);
    vec2 bc = uniform_sample_triangle(u1);
    Triangle trig = get_triangle(instance_id, prim_id);

    vec3 p = lerp3(bc, trig.v[0],trig.v[1], trig.v[2]);
    vec2 tc = lerp3(bc, trig.tc[0],trig.tc[1], trig.tc[2]);
    vec3 ng = trig.ng;
    float area = length(cross(trig.v[1] - trig.v[0],  trig.v[2] - trig.v[0])) * 0.5;
    return SurfaceSample(p, ng,  tc, pdf_prim * 1.0 / area);

}
