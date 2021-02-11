#include <optix_embedded_ptx.h>
namespace akari::gpu {
    extern const unsigned char *optix_ptx = ::optix_ptx;
    extern const size_t optix_ptx_size    = sizeof(::optix_ptx);
} // namespace akari::gpu