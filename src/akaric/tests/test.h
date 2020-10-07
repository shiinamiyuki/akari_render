#pragma once
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
        inline __host__ __device__ Float3 L()
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
        inline __host__ __device__ Float3 foo()
        {
            return Float3(0.0);
        }
        inline __host__ __device__ Float3 pow4(Float3 x)
        {
            return sqr(x) * sqr(x);
        }
        inline __host__ __device__ Float CosTheta(Float3 w)
        {
            return w.y;
        }
        inline __host__ __device__ Float Cos2Theta(Float3 w)
        {
            return w.y * w.y;
        }
        inline __host__ __device__ Float SinTheta(Float3 w)
        {
            return sqrt(Sin2Theta(w));
        }
        inline __host__ __device__ Float max(Float a, Float b);
        inline __host__ __device__ Float Sin2Theta(Float3 w)
        {
            return max(0.0,1.0 - (Cos2Theta(w)));
        }
        inline __host__ __device__ Float Tan2Theta(Float3 w)
        {
            return Sin2Theta(w) / (Cos2Theta(w));
        }
        inline __host__ __device__ Float TanTheta(Float3 w)
        {
            return sqrt(Tan2Theta(w));
        }
        inline __host__ __device__ Float sqrt(Float v);
        inline __host__ __device__ void setZero(Float & v)
        {
            v = 0.0;
        }
        inline __host__ __device__ Float GGX_D(Float alpha, Float3 m)
        {
            if(m.y <= 0.0)
                return 0.0;
            Float a2 = alpha * alpha;
            Float c2 = Cos2Theta(m);
            Float t2 = Tan2Theta(m);
            Float at = a2 + t2;
            return a2 / (3.1415926 * c2 * c2 * at * at);
        }
        inline __host__ __device__ Float3 sqr(Float3 x)
        {
            return x * x;
        }
    };
};
