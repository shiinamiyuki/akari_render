#pragma once
#include <cuda.h>
#include <akari/common/color.h>
#include <akari/common/buffer.h>
namespace akari::asl {
    template<class C>
    class Test {
        public:
        AKR_IMPORT_TYPES()
        struct Light {
            Float3 pos;
            Float3 color;
        };
        Buffer<Light> lights;
        const int const_ival = 2;
        inline AKR_XPU Float3 L()
        {
            Float3 res = Float3(0.0);
            { // for begin
                uint i = uint(0);
                while(i < lights.size())
                {
                    res += lights[i].color;
                }
                i += 1;
            } // for end
            return res;
        }
        inline AKR_XPU Float3 foo()
        {
            return Float3(0.0);
        }
        inline AKR_XPU Float3 pow4(Float3 x)
        {
            return sqr(x) * sqr(x);
        }
        inline AKR_XPU Float CosTheta(Float3 w)
        {
            return w.y;
        }
        inline AKR_XPU Float Cos2Theta(Float3 w)
        {
            return w.y * w.y;
        }
        inline AKR_XPU Float SinTheta(Float3 w)
        {
            return sqrt(Sin2Theta(w));
        }
        inline AKR_XPU Float max(Float a, Float b);
        inline AKR_XPU Float Sin2Theta(Float3 w)
        {
            return max(0.0,1.0 - (Cos2Theta(w)));
        }
        inline AKR_XPU Float Tan2Theta(Float3 w)
        {
            return Sin2Theta(w) / (Cos2Theta(w));
        }
        inline AKR_XPU Float TanTheta(Float3 w)
        {
            return sqrt(Tan2Theta(w));
        }
        inline AKR_XPU Float sqrt(Float v);
        inline AKR_XPU void setZero(Float & v)
        {
            v = 0.0;
        }
        inline AKR_XPU Float GGX_D(Float alpha, Float3 m)
        {
            if(m.y <= 0.0)
                return 0.0;
            Float a2 = alpha * alpha;
            Float c2 = Cos2Theta(m);
            Float t2 = Tan2Theta(m);
            Float at = a2 + t2;
            return a2 / (3.1415926 * c2 * c2 * at * at);
        }
        inline AKR_XPU Float3 sqr(Float3 x)
        {
            return x * x;
        }
        inline AKR_XPU void change_const(Float3 & x)
        {
            x.x = 2.0;
        }
        inline AKR_XPU Float cos_theta(const Float3 w)
        {
            return w.y;
        }
        inline AKR_XPU Float abs_cos_theta(const Float3 w)
        {
            return abs(cos_theta(w));
        }
        inline AKR_XPU Float cos2_theta(const Float3 w)
        {
            return w.y * w.y;
        }
        inline AKR_XPU Float sin2_theta(const Float3 w)
        {
            return 1.0 - (cos2_theta(w));
        }
        inline AKR_XPU Float sin_theta(const Float3 w)
        {
            return sqrt(fmax(0.0,sin2_theta(w)));
        }
        inline AKR_XPU Float tan2_theta(const Float3 w)
        {
            return sin2_theta(w) / (cos2_theta(w));
        }
        inline AKR_XPU Float tan_theta(const Float3 w)
        {
            return sqrt(fmax(0.0,tan2_theta(w)));
        }
        inline AKR_XPU Float cos_phi(const Float3 w)
        {
            Float sinTheta = sin_theta(w);
            return sinTheta == 0.0 ? 1.0 : clamp(w.x / (sinTheta),-1.0,1.0);
        }
        inline AKR_XPU Float sin_phi(const Float3 w)
        {
            Float sinTheta = sin_theta(w);
            return sinTheta == 0.0 ? 0.0 : clamp(w.z / (sinTheta),-1.0,1.0);
        }
        inline AKR_XPU Float cos2_phi(const Float3 w)
        {
            return cos_phi(w) * cos_phi(w);
        }
        inline AKR_XPU Float sin2_phi(const Float3 w)
        {
            return sin_phi(w) * sin_phi(w);
        }
        inline AKR_XPU bool same_hemisphere(const Float3 wo, const Float3 wi)
        {
            return wo.y * wi.y >= 0.0;
        }
        inline AKR_XPU Float3 reflect(const Float3 w, const Float3 n)
        {
            return -1.0 * w + 2.0 * dot(w,n) * n;
        }
        inline AKR_XPU bool refract(const Float3 wi, const Float3 n, Float eta, Float3 & wt)
        {
            Float cosThetaI = dot(n,wi);
            Float sin2ThetaI = fmax(0.0,1.0 - (cosThetaI * cosThetaI));
            Float sin2ThetaT = eta * eta * sin2ThetaI;
            if(sin2ThetaT >= 1.0)
                return false;
            Float cosThetaT = sqrt(1.0 - (sin2ThetaT));
            wt = eta * -wi + (eta * cosThetaI - (cosThetaT)) * n;
            return true;
        }
        inline AKR_XPU void swap(Float & a, Float & b)
        {
            Float t = a;
            a = b;
            b = t;
        }
        inline AKR_XPU Float fr_dielectric(Float cosThetaI, Float etaI, Float etaT)
        {
            bool entering = cosThetaI > 0.0;
            if(!entering)
            {
                swap(etaI,etaT);
                cosThetaI = abs(cosThetaI);
            }
            Float sinThetaI = sqrt(max(0.0,1.0 - (cosThetaI * cosThetaI)));
            Float sinThetaT = etaI / (etaT) * sinThetaI;
            if(sinThetaT >= 1.0)
                return 1.0;
            Float cosThetaT = sqrt(max(0.0,1.0 - (sinThetaT * sinThetaT)));
            Float Rpar = (etaT * cosThetaI - (etaI * cosThetaT)) / (etaT * cosThetaI + etaI * cosThetaT);
            Float Rper = (etaI * cosThetaI - (etaT * cosThetaT)) / (etaI * cosThetaI + etaT * cosThetaT);
            return 0.5 * (Rpar * Rpar + Rper * Rper);
        }
        inline AKR_XPU Spectrum fr_conductor(Float cosThetaI, const Spectrum etaI, const Spectrum etaT, const Spectrum k)
        {
            Float CosTheta2 = cosThetaI * cosThetaI;
            Float SinTheta2 = 1.0 - (CosTheta2);
            Spectrum Eta = __div__(etaT,etaI);
            Spectrum Etak = __div__(k,etaI);
            Spectrum Eta2 = __mul__(Eta,Eta);
            Spectrum Etak2 = __mul__(Etak,Etak);
            Spectrum t0 = __sub__(__sub__(Eta2,Etak2),SinTheta2);
            Spectrum a2plusb2 = sqrt(__add__(__mul__(t0,t0),__mul__(__mul__(4.0,Eta2),Etak2)));
            Spectrum t1 = __add__(a2plusb2,CosTheta2);
            Spectrum a = sqrt(__mul__(0.5,__add__(a2plusb2,t0)));
            Spectrum t2 = __mul__(__mul__(2.0,a),cosThetaI);
            Spectrum Rs = __div__(__sub__(t1,t2),__add__(t1,t2));
            Spectrum t3 = __add__(__mul__(CosTheta2,a2plusb2),SinTheta2 * SinTheta2);
            Spectrum t4 = __mul__(t2,SinTheta2);
            Spectrum Rp = __div__(__mul__(Rs,__sub__(t3,t4)),__add__(t3,t4));
            return __mul__(0.5,__add__(Rp,Rs));
        }
        struct BufferBinder {
            std::vector<Light> lights;
        };
        void bind(const BufferBinder& binder){
            lights.copy(binder.lights)
        }
    };
}
