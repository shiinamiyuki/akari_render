// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>
#include <akari/scenegraph.h>
#include <akari/render.h>
#include <spdlog/spdlog.h>
#include <embree3/rtcore.h>
namespace akari::render {
    static inline RTCRay toRTCRay(const Ray &_ray) {
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
    using scene::Mesh;
    using scene::P;
    class EmbreeAccelImpl : public EmbreeAccel {
        RTCScene rtcScene = nullptr;
        RTCDevice device = nullptr;
        std::unordered_map<const Mesh *, RTCScene> per_mesh_scene;
#define EMBREE_CHECK(expr)                                                                                             \
    [&] {                                                                                                              \
        expr;                                                                                                          \
        auto err = rtcGetDeviceError(device);                                                                          \
        if (err != RTC_ERROR_NONE) {                                                                                   \
            spdlog::error("embree error: {}", err);                                                                    \
        }                                                                                                              \
    }()
      public:
        EmbreeAccelImpl() { device = rtcNewDevice(nullptr); }
        void build(const Scene &scene, const std::shared_ptr<scene::SceneGraph> &scene_graph) override {
            spdlog::info("building acceleration structure for {} meshes, {} instances", scene_graph->meshes.size(),
                         scene.instances.size());
            if (rtcScene) {
                rtcReleaseScene(rtcScene);
            }
            rtcScene = rtcNewScene(device);
            for (auto &mesh : scene_graph->meshes) {
                const auto m_scene = rtcNewScene(device);
                {
                    const auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                    AKR_ASSERT(mesh->vertices.data() != nullptr);
                    EMBREE_CHECK(rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                            &mesh->vertices[0], 0, sizeof(float) * 3,
                                                            mesh->vertices.size()));
                    AKR_ASSERT_THROW(rtcGetDeviceError(device) == RTC_ERROR_NONE);
                    EMBREE_CHECK(rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                                            &mesh->indices[0], 0, sizeof(int) * 3,
                                                            mesh->indices.size()));
                    AKR_ASSERT_THROW(rtcGetDeviceError(device) == RTC_ERROR_NONE);
                    EMBREE_CHECK(rtcCommitGeometry(geometry));
                    EMBREE_CHECK(rtcAttachGeometry(m_scene, geometry));
                    EMBREE_CHECK(rtcReleaseGeometry(geometry));
                    EMBREE_CHECK(rtcCommitScene(m_scene));
                }
                per_mesh_scene[mesh.get()] = m_scene;
            }
            unsigned int id = 0;
            for (auto instance : scene.instances) {
                const auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
                EMBREE_CHECK(rtcSetGeometryInstancedScene(geometry, per_mesh_scene.at(instance.mesh)));
                EMBREE_CHECK(
                    rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &instance.transform.m));
                EMBREE_CHECK(rtcCommitGeometry(geometry));
                EMBREE_CHECK(rtcAttachGeometryByID(rtcScene, geometry, id));
                EMBREE_CHECK(rtcReleaseGeometry(geometry));
                id++;
            }
            EMBREE_CHECK(rtcCommitScene(rtcScene));
        }
        bool occlude1(const Ray &ray) const override {
            auto rtcRay = toRTCRay(ray);
            RTCIntersectContext context;
            rtcInitIntersectContext(&context);
            rtcOccluded1(rtcScene, &context, &rtcRay);
            return rtcRay.tfar == -std::numeric_limits<float>::infinity();
        }
        std::optional<Intersection> intersect1(const Ray &ray) const override {
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
                return std::nullopt;
            Intersection intersection;
            intersection.prim_id = rayHit.hit.primID;
            AKR_ASSERT(rayHit.hit.geomID == 0);
            intersection.geom_id = rayHit.hit.instID[0];
            intersection.uv = vec2(rayHit.hit.u, rayHit.hit.v);
            intersection.t = rayHit.ray.tfar;
            return intersection;
        }
        Bounds3f world_bounds() const override {
            RTCBounds bounds;
            rtcGetSceneBounds(rtcScene, &bounds);
            return Bounds3f(vec3(bounds.lower_x, bounds.lower_y, bounds.lower_z),
                            vec3(bounds.upper_x, bounds.upper_y, bounds.upper_z));
        }
        ~EmbreeAccelImpl() {
            for (auto &&[_, scene] : per_mesh_scene) {
                rtcReleaseScene(scene);
            }
            rtcReleaseScene(rtcScene);
            rtcReleaseDevice(device);
        }
    };
    std::shared_ptr<EmbreeAccel> create_embree_accel() { return std::make_shared<EmbreeAccelImpl>(); }
} // namespace akari::render