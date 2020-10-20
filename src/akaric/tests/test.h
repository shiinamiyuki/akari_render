#pragma once
#include <glm/glm.hpp>
namespace akari::shader {
    inline float cos_theta(const glm::vec3 & w)
    {
        return w.y;
    }
    inline float abs_cos_theta(const glm::vec3 & w)
    {
        return abs(cos_theta(w));
    }
    inline float cos2_theta(const glm::vec3 & w)
    {
        return w.y * w.y;
    }
    inline float sin2_theta(const glm::vec3 & w)
    {
        return 1.0 - (cos2_theta(w));
    }
    inline float sin_theta(const glm::vec3 & w)
    {
        return sqrt(fmax(0.0,sin2_theta(w)));
    }
    inline float tan2_theta(const glm::vec3 & w)
    {
        return sin2_theta(w) / (cos2_theta(w));
    }
    inline float tan_theta(const glm::vec3 & w)
    {
        return sqrt(fmax(0.0,tan2_theta(w)));
    }
    inline float cos_phi(const glm::vec3 & w)
    {
        float sinTheta = sin_theta(w);
        return sinTheta == 0.0 ? 1.0 : clamp(w.x / (sinTheta),-1.0,1.0);
    }
    inline float sin_phi(const glm::vec3 & w)
    {
        float sinTheta = sin_theta(w);
        return sinTheta == 0.0 ? 0.0 : clamp(w.z / (sinTheta),-1.0,1.0);
    }
    inline float cos2_phi(const glm::vec3 & w)
    {
        return cos_phi(w) * cos_phi(w);
    }
    inline float sin2_phi(const glm::vec3 & w)
    {
        return sin_phi(w) * sin_phi(w);
    }
    inline bool same_hemisphere(const glm::vec3 & wo, const glm::vec3 & wi)
    {
        return wo.y * wi.y >= 0.0;
    }
    inline bool refract(const glm::vec3 & wi, const glm::vec3 & n, float eta, glm::vec3 & wt)
    {
        float cosThetaI = dot(n,wi);
        float sin2ThetaI = fmax(0.0,1.0 - (cosThetaI * cosThetaI));
        float sin2ThetaT = eta * eta * sin2ThetaI;
        if(sin2ThetaT >= 1.0)
            return false;
        float cosThetaT = sqrt(1.0 - (sin2ThetaT));
        wt = eta * -wi + (eta * cosThetaI - (cosThetaT)) * n;
        return true;
    }
    inline void swap(float & a, float & b)
    {
        float t = a;
        a = b;
        b = t;
    }
    inline float fr_dielectric(float cosThetaI, float etaI, float etaT)
    {
        bool entering = cosThetaI > 0.0;
        if(!entering)
        {
            swap(etaI,etaT);
            cosThetaI = abs(cosThetaI);
        }
        float sinThetaI = sqrt(max(0.0,1.0 - (cosThetaI * cosThetaI)));
        float sinThetaT = etaI / (etaT) * sinThetaI;
        if(sinThetaT >= 1.0)
            return 1.0;
        float cosThetaT = sqrt(max(0.0,1.0 - (sinThetaT * sinThetaT)));
        float Rpar = (etaT * cosThetaI - (etaI * cosThetaT)) / (etaT * cosThetaI + etaI * cosThetaT);
        float Rper = (etaI * cosThetaI - (etaT * cosThetaT)) / (etaI * cosThetaI + etaT * cosThetaT);
        return 0.5 * (Rpar * Rpar + Rper * Rper);
    }
    inline glm::vec3 fr_conductor(float cosThetaI, const glm::vec3 & etaI, const glm::vec3 & etaT, const glm::vec3 & k)
    {
        float CosTheta2 = cosThetaI * cosThetaI;
        float SinTheta2 = 1.0 - (CosTheta2);
        glm::vec3 Eta = etaT / (etaI);
        glm::vec3 Etak = k / (etaI);
        glm::vec3 Eta2 = Eta * Eta;
        glm::vec3 Etak2 = Etak * Etak;
        glm::vec3 t0 = Eta2 - (Etak2) - (SinTheta2);
        glm::vec3 a2plusb2 = sqrt(t0 * t0 + 4.0 * Eta2 * Etak2);
        glm::vec3 t1 = a2plusb2 + CosTheta2;
        glm::vec3 a = sqrt(0.5 * (a2plusb2 + t0));
        glm::vec3 t2 = 2.0 * a * cosThetaI;
        glm::vec3 Rs = (t1 - (t2)) / (t1 + t2);
        glm::vec3 t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
        glm::vec3 t4 = t2 * SinTheta2;
        glm::vec3 Rp = Rs * (t3 - (t4)) / (t3 + t4);
        return 0.5 * (Rp + Rs);
    }
}
