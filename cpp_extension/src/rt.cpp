#include <rust-api.h>
#include <util.h>
#include <embree4/rtcore.h>
#include <mutex>

namespace akari::rt {

static RTCDevice &get_device() {
    static RTCDevice device{};
    static std::once_flag flag;
    std::call_once(flag, [&]() { device = rtcNewDevice(nullptr); });
    return device;
}
struct EmbreeApiImpl {
    RTCDevice &device;
    EmbreeApiImpl() : device(get_device()) {}
    ~EmbreeApiImpl() = default;
};
enum class EmbreeGeometryKind : uint32_t { Mesh };
struct EmbreeGeometryImpl {
    RTCScene scene{};
    EmbreeGeometryKind kind{};
    EmbreeGeometryImpl(EmbreeApiImpl &api, const EmbreeMeshBuildArgs &args)
        : kind(EmbreeGeometryKind::Mesh) {
        const auto geometry =
            rtcNewGeometry(api.device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0,
                                   RTC_FORMAT_FLOAT3, args.vertices, 0,
                                   sizeof(float) * 3, args.num_vertices);
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0,
                                   RTC_FORMAT_UINT3, args.indices, 0,
                                   sizeof(uint32_t) * 3, args.num_indices);
        rtcCommitGeometry(geometry);
        scene = rtcNewScene(api.device);
        rtcAttachGeometry(scene, geometry);
        rtcSetGeometryEnableFilterFunctionFromArguments(geometry, true);
        rtcCommitScene(scene);
    }
    ~EmbreeGeometryImpl() { rtcReleaseScene(scene); }
};
struct EmbreeTlasImpl {
    RTCScene scene{};
    EmbreeTlasImpl(EmbreeApiImpl &api, const EmbreeTlasBuildArgs &args) {
        scene = rtcNewScene(api.device);
        for (size_t i = 0; i < args.num_instances; i++) {
            const auto geometry =
                static_cast<EmbreeGeometryImpl *>(args.instances[i].geometry._0);
            const auto instance =
                rtcNewGeometry(api.device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryInstancedScene(instance, geometry->scene);
            rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                                    args.instances[i].transform);
            rtcCommitGeometry(instance);
            rtcAttachGeometryByID(scene, instance, i);
        }
        rtcCommitScene(scene);
    }
    ~EmbreeTlasImpl() { rtcReleaseScene(scene); }
    bool traverse(EmbreeRayQuery &rq) const {
        RTCRay ray{};
        ray.org_x = rq.ray.origin[0];
        ray.org_y = rq.ray.origin[1];
        ray.org_z = rq.ray.origin[2];
        ray.dir_x = rq.ray.dir[0];
        ray.dir_y = rq.ray.dir[1];
        ray.dir_z = rq.ray.dir[2];
        ray.tnear = rq.ray.t_near;
        ray.tfar = rq.ray.t_far;
        ray.time = rq.ray.time;
        RTCRayHit rh{};
        rh.ray = ray;
        struct RqContext : RTCRayQueryContext {
            EmbreeRayQuery &rq;
            explicit RqContext(EmbreeRayQuery & rq) : RTCRayQueryContext {}, rq(rq) {}
        };

        const auto mesh_filter =
            [](const struct RTCFilterFunctionNArguments *args) {
                const auto context = static_cast<RqContext *>(args->context);
                if (args->valid[0] == 0) {
                    return;
                }
                AKR_ASSERT(args->N == 1);
                const auto N = args->N;
                const auto geom_id = RTCHitN_geomID(args->hit, N, 0);
                const auto prim_id = RTCHitN_primID(args->hit, N, 0);
                const auto hit_u = RTCHitN_u(args->hit, N, 0);
                const auto hit_v = RTCHitN_v(args->hit, N, 0);
                auto tfar = RTCRayN_tfar(args->ray, N, 0);
                const auto &cb = context->rq.on_triangle_hit;
                EmbreeHit tmp_hit{tfar, hit_u, hit_v, prim_id, geom_id};
                const auto flags = cb.on_hit(cb.data, &context->rq.ray, &tmp_hit);
                if ((flags & EMBREE_HIT_ACCEPT) == 0) {
                    args->valid[0] = 0;
                }
                if ((flags & EMBREE_HIT_TERMINATE) != 0 ||
                    context->rq.terminate_on_first_hit) {
                    tfar = -std::numeric_limits<float>::infinity();
                }
            };
        const auto terminate_on_first_filter =
            [](const struct RTCFilterFunctionNArguments *args) {
                const auto context = static_cast<RqContext *>(args->context);
                if (args->valid[0] == 0) {
                    return;
                }
                AKR_ASSERT(args->N == 1);
                const auto N = args->N;
                auto tfar = RTCRayN_tfar(args->ray, N, 0);
                tfar = -std::numeric_limits<float>::infinity();
            };
        RqContext context{rq};

        RTCIntersectArguments ia{};
        rtcInitIntersectArguments(&ia);
        ia.context = &context;
        ia.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
        if (rq.on_triangle_hit.data) {
            ia.filter = mesh_filter;
        } else {
            if (rq.terminate_on_first_hit)
                ia.filter = terminate_on_first_filter;
        }
        rtcIntersect1(scene, &rh, &ia);
        rq.hit.geom_id = rh.hit.geomID;
        rq.hit.prim_id = rh.hit.primID;
        rq.hit.t = rh.ray.tfar;
        rq.hit.u = rh.hit.u;
        rq.hit.v = rh.hit.v;
        return rh.hit.geomID != RTC_INVALID_GEOMETRY_ID &&
               rh.hit.primID != RTC_INVALID_GEOMETRY_ID;
    }
};
extern "C" EmbreeApi embree_api_create() {
    EmbreeApi api{};
    api.data = new EmbreeApiImpl{};
    api.dtor = [](EmbreeApi *api) {
        delete static_cast<EmbreeApiImpl *>(api->data);
    };
    api.build_mesh = [](const EmbreeApi *api, const EmbreeMeshBuildArgs *args) {
        return EmbreeGeometry{new EmbreeGeometryImpl(
            *static_cast<EmbreeApiImpl *>(api->data), *args)};
    };
    api.destroy_mesh = [](const EmbreeApi *api, EmbreeGeometry geometry) {
        delete static_cast<EmbreeGeometryImpl *>(geometry._0);
    };
    api.build_tlas = [](const EmbreeApi *api, const EmbreeTlasBuildArgs *args) {
        return EmbreeTlas{
            new EmbreeTlasImpl(*static_cast<EmbreeApiImpl *>(api->data), *args)};
    };
    api.destroy_tlas = [](const EmbreeApi *api, EmbreeTlas tlas) {
        delete static_cast<EmbreeTlasImpl *>(tlas._0);
    };
    api.traverse = [](const EmbreeTlas *tlas_, EmbreeRayQuery *rq) {
        const auto tlas = static_cast<EmbreeTlasImpl *>(tlas_->_0);
        return tlas->traverse(*rq);
    };
    return api;
}
}// namespace akari::rt