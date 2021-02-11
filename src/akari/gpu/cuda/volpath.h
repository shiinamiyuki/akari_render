#pragma once

#include <akari/render_xpu.h>

namespace akari::gpu {
    using namespace akari::render;
    struct VFloat3 {
        float* x;
        float* y;
        float* z;
    };
    struct PathState {
        ivec2 pixel;
        Ray ray;
        GPUSampler sampler;
    };

    struct KernelGlobals {
        const Camera * camera = nullptr;
        PathState * states = nullptr;
    };
}