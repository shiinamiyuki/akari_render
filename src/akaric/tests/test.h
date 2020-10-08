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
        const int const_ival = 2;
        Buffer<Light> lights;
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
        struct Tuple_int_Float {
            int _0;
            Float _1;
        };
        inline AKR_XPU Tuple_int_Float test()
        {
            return Tuple_int_Float{1, 1.0};
        }
        inline AKR_XPU void test3()
        {
            Float3 l = L();
        }
        inline AKR_XPU void test2()
        {
            Tuple_int_Float __Gen_tmp0 = test();
            int x = __Gen_tmp0._0;
            Float y = __Gen_tmp0._1;
        }
        struct Tuple_int_int {
            int _0;
            int _1;
        };
        using MaterialHandle = Tuple_int_int;
        inline AKR_XPU Tuple_int_int get_handle()
        {
            return Tuple_int_int{0,0};
        }
        struct BufferBinder {
            std::vector<Light> lights;
        };
        void bind(const BufferBinder& binder){
            lights.copy(binder.lights)
        }
    };
}
