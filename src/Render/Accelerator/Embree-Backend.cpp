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

#include <Akari/Core/Plugin.h>
#include <Akari/Render/Accelerator.h>
#include <Akari/Render/Mesh.h>
#include <Akari/Render/Scene.h>
#include <embree3/rtcore.h>

namespace Akari {
    class EmbreeAccelerator : public Accelerator {
        RTCScene rtcScene = nullptr;
        RTCDevice device = nullptr;

      public:
        AKR_DECL_COMP(EmbreeAccelerator, "EmbreeAccelerator")
        void Build(const Scene &scene) override {
            if (!device) {
                device = rtcNewDevice(nullptr);
            }
            if (!rtcScene) {
                rtcScene = rtcNewScene(device);
            }
            for (auto &mesh : scene.GetMeshes()) {
                auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                           mesh->GetVertexBuffer(), 0, sizeof(Vertex), mesh->GetTriangleCount() * 3);
                rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, mesh->GetIndexBuffer(),
                                           0, sizeof(ivec3), mesh->GetTriangleCount());
                rtcCommitGeometry(geometry);
                rtcAttachGeometry(rtcScene, geometry);
                rtcReleaseGeometry(geometry);
            }
            rtcCommitScene(rtcScene);
        }
        static inline RTCRay toRTCRay(const Ray &_ray) {
            RTCRay ray;
            auto _o = _ray.o;
            ray.dir_x = _ray.d.x;
            ray.dir_y = _ray.d.y;
            ray.dir_z = _ray.d.z;
            ray.org_x = _o.x;
            ray.org_y = _o.y;
            ray.org_z = _o.z;
            ray.tnear = _ray.t_min;
            ray.tfar = _ray.t_max;
            ray.flags = 0;
            return ray;
        }

        bool Intersect(const Ray &ray, Intersection *intersection) const override {
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
            intersection->primId = rayHit.hit.primID;
            intersection->meshId = rayHit.hit.geomID;
            intersection->Ng = normalize(vec3(rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z));
            intersection->uv = vec2(rayHit.hit.u, rayHit.hit.v);
            intersection->t = rayHit.ray.tfar;
            return true;
        }
        bool Occlude(const Ray &ray) const override {
            RTCRay rtcRay = toRTCRay(ray);
            RTCIntersectContext context;
            rtcInitIntersectContext(&context);
            rtcOccluded1(rtcScene, &context, &rtcRay);
            return rtcRay.tfar < 0;
        }
        Bounds3f GetBounds() const override {
            RTCBounds bounds{};
            rtcGetSceneBounds(rtcScene, &bounds);
            Bounds3f box;
            box.p_min[0] = bounds.lower_x;
            box.p_min[1] = bounds.lower_y;
            box.p_min[2] = bounds.lower_z;

            box.p_max[0] = bounds.upper_x;
            box.p_max[1] = bounds.upper_y;
            box.p_max[2] = bounds.upper_z;
            return box;
        }
        ~EmbreeAccelerator() override {
            if (rtcScene) {
                rtcReleaseScene(rtcScene);
            }
            if (device) {
                rtcReleaseDevice(device);
            }
        }
    };
    AKR_EXPORT_COMP(EmbreeAccelerator, "Accelerator")
} // namespace Akari