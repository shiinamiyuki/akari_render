// MIT License
//
// Copyright (c) 2020 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "builtins.glsl"
#include "constants.glsl"
float cos_theta(const vec3 w) { return w.y; }

float abs_cos_theta(const vec3 w) { return abs(cos_theta(w)); }

float cos2_theta(const vec3 w) { return w.y * w.y; }

float sin2_theta(const vec3 w) { return 1.0 - cos2_theta(w); }

float sin_theta(const vec3 w) { return sqrt(max(0.0, sin2_theta(w))); }

float tan2_theta(const vec3 w) { return sin2_theta(w) / cos2_theta(w); }

float tan_theta(const vec3 w) { return sqrt(max(0.0, tan2_theta(w))); }

float cos_phi(const vec3 w) {
    float sinTheta = sin_theta(w);
    return (sinTheta == 0.0) ? 1.0 : clamp(w.x / sinTheta, -1.0, 1.0);
}
float sin_phi(const vec3 w) {
    float sinTheta = sin_theta(w);
    return (sinTheta == 0.0) ? 0.0 : clamp(w.z / sinTheta, -1.0, 1.0);
}

float cos2_phi(const vec3 w) { return cos_phi(w) * cos_phi(w); }
float sin2_phi(const vec3 w) { return sin_phi(w) * sin_phi(w); }

bool same_hemisphere(const vec3 wo, const vec3 wi) {
    return wo.y * wi.y >= 0.0;
}

bool refract(const vec3 wi, const vec3 n, float eta, inout vec3 wt) {
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;
    if (sin2ThetaT >= 1.0)
        return false;

    float cosThetaT = sqrt(1.0 - sin2ThetaT);

    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}
void swap(inout float a, inout float b){
    float t = a;
    a = b;
    b = t;
}

float fr_dielectric(float cosThetaI, float etaI, float etaT) {
    bool entering = cosThetaI > 0.0;
    if (!entering) {
        swap(etaI, etaT);
        cosThetaI = abs(cosThetaI);
    }
    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0)
        return 1.0;
    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));

    float Rpar = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rper = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return 0.5 * (Rpar * Rpar + Rper * Rper);
}



Spectrum fr_conductor(float cosThetaI, const Spectrum etaI, const Spectrum etaT,
                                            const Spectrum k) {
    float CosTheta2 = cosThetaI * cosThetaI;
    float SinTheta2 = 1.0 - CosTheta2;
    Spectrum Eta = etaT / etaI;
    Spectrum Etak = k / etaI;
    Spectrum Eta2 = Eta * Eta;
    Spectrum Etak2 = Etak * Etak;

    Spectrum t0 = Eta2 - Etak2 - SinTheta2;
    Spectrum a2plusb2 = sqrt(t0 * t0 + 4.0 * Eta2 * Etak2);
    Spectrum t1 = a2plusb2 + CosTheta2;
    Spectrum a = sqrt(0.5 * (a2plusb2 + t0));
    Spectrum t2 = 2.0 * a * cosThetaI;
    Spectrum Rs = (t1 - t2) / (t1 + t2);

    Spectrum t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
    Spectrum t4 = t2 * SinTheta2;
    Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (Rp + Rs);
}

vec3 spherical_to_xyz(float sinTheta, float cosTheta, float phi) {
    return vec3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
}

float spherical_theta(const vec3 v) { return acos(clamp(v.y, -1.0, 1.0)); }

float spherical_phi(const vec3 v) {
    float p = atan(v.z, v.x);
    return p < 0.0 ? (p + 2.0 * Pi) : p;
}