// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <akari/render/geometry.hpp>
namespace akari {
    struct TraversalFlags{
        static size_t EValid = 1;
        static size_t ETerminated = 1 << 1;
        TraversalFlags(size_t f):flags(f){}
        operator size_t ()const{return flags;}
    private:
        size_t flags;
    };
    class WavefrontPathTracerImpl{
        size_t size_path_state = 0u;
        using MissShader = void(*)(void*);
        using AnyHitShader = void(*)(void*);
        using ClosestHitshader = void(*)(void*, Intersection&);
        using Continuation = void(*)(void*);
        void trace(const Ray & ray, PathState & state,)
    }; 

    template<typename PathState, class RayGenerationProgram>
    class WavefrontPathTracer{
    public:
        using MissShader = void(*)(PathState&);
        using AnyHitShader = void(*)(PathState&);
        using ClosestHitshader = void(*)(PathState &, Intersection& );
        using Continuation = void(*)(PathState&);

        void trace(const Ray& ray, PathState & state, Continuation cont){
            
        }
    };
    
    
}