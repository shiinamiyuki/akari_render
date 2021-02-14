#include <cuda.h>
#include <akari/gpu/cuda/volpath.h>
#include <akari/render_xpu.h>

namespace akari::gpu {
    using namespace akari::render;
    AKR_XPU uint32_t tid() { return threadIdx.x + blockIdx.x * blockDim.x; }
    extern "C" {
    __global__ void volpath_advance() {

        printf("tid = %d\n", tid());
        // auto camera = kg->camera;
        // auto &sampler =  kg->states[0].sampler;
        // auto sample = camera->generate_ray(sampler.next2d(), sampler.next2d(),  kg->states[0].pixel);
        // kg->states[0].ray = sample.ray;
    }
    }

} // namespace akari::gpu