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

#include <akari/kernel/embree.inl>
#include <akari/kernel/scene.h>
#include <akari/core/logger.h>
#ifdef AKR_ENABLE_EMBREE
namespace akari {
    AKR_VARIANT void EmbreeAccelerator<C>::build(Scene<C> &scene) {
        rtcScene = rtcNewScene(device);
        for (const MeshInstance<C> &mesh : scene.meshes) {
            auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
            rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, &mesh.vertices[0], 0,
                                       sizeof(float) * 3, mesh.vertices.size() / 3);
            AKR_ASSERT_THROW(rtcGetDeviceError(device) == RTC_ERROR_NONE);
            AKR_ASSERT(mesh.indices.size() % 3 == 0);
            rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, &mesh.indices[0], 0,
                                       sizeof(int) * 3, mesh.indices.size() / 3);
            AKR_ASSERT_THROW(rtcGetDeviceError(device) == RTC_ERROR_NONE);
            rtcCommitGeometry(geometry);
            rtcAttachGeometry(rtcScene, geometry);
            rtcReleaseGeometry(geometry);
        }
        rtcCommitScene(rtcScene);
        AKR_ASSERT_THROW(rtcGetDeviceError(device) == RTC_ERROR_NONE);
    }
    AKR_VARIANT static inline RTCRay toRTCRay(const Ray<C> &_ray) {
        RTCRay ray;
        auto _o = _ray.o;
        ray.dir_x = _ray.d.x;
        ray.dir_y = _ray.d.y;
        ray.dir_z = _ray.d.z;
        ray.org_x = _o.x;
        ray.org_y = _o.y;
        ray.org_z = _o.z;
        ray.tnear = _ray.tmin;
        ray.tfar = _ray.tmax;
        ray.flags = 0;
        return ray;
    }
    AKR_VARIANT bool EmbreeAccelerator<C>::occlude(const Ray<C> &ray) const {
        auto rtcRay = toRTCRay(ray);
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        rtcOccluded1(rtcScene, &context, &rtcRay);
        return rtcRay.tfar == -std::numeric_limits<float>::infinity();
    }
    AKR_VARIANT bool EmbreeAccelerator<C>::intersect(const Ray<C> &ray, SurfaceHit<C> *intersection) const {
        RTCRayHit rayHit;
        rayHit.ray = toRTCRay(ray);
        rayHit.ray.flags = 0;
        rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        rtcIntersect1(rtcScene, &context, &rayHit);
        if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID || rayHit.hit.primID == RTC_INVALID_GEOMETRY_ID)
            return false;
        intersection->prim_id = rayHit.hit.primID;
        intersection->geom_id = rayHit.hit.geomID;
        intersection->uv = float2(rayHit.hit.u, rayHit.hit.v);
        return true;
    }
    AKR_RENDER_CLASS(EmbreeAccelerator)
} // namespace akari
#endif