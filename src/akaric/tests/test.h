namespace akari::asl {
    template<class C>
    class Test {
        public:
        AKR_IMPORT_TYPES()
        __host__ __device__ Float3 foo()
        {
            return Float3(0.0);
        }
        __host__ __device__ Float3 pow4(Float3 x)
        {
            int i = 0;
            Float3 p = Float3(1.0);
            while(i < 4)
            {
                p = p * x;
                i = i + 1;
            }
            return p;
        }
    };
};
