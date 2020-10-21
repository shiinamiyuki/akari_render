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

#include "sbvh.h"
#include <akari/render/instance.h>

namespace akari::render {
    class BVHAccelerator final : public Accelerator {
        struct TriangleHandle {
            const MeshInstance *mesh = nullptr;
            int idx = -1;
            Bounds3f box;
            TriangleHandle(const MeshInstance *mesh, int idx, Bounds3f box = Bounds3f())
                : mesh(mesh), idx(idx), box(box) {}
            [[nodiscard]] Bounds3f full_bbox() const {
                auto trig = get_triangle(*mesh, idx);
                Bounds3f bbox;
                bbox = bbox.expand(trig.vertices[0]);
                bbox = bbox.expand(trig.vertices[1]);
                bbox = bbox.expand(trig.vertices[2]);
                // box.pmax += Float3(0.000001);
                return bbox;
            }
            [[nodiscard]] std::pair<TriangleHandle, TriangleHandle> split(int axis, Float split) const {
                AKR_ASSERT(split >= box.pmin[axis] && split <= box.pmax[axis]);
                auto trig = get_triangle(*mesh, idx);
                // auto bbox = full_bbox();
                int cnt[2] = {0};
                Bounds3f left, right;
                for (int i = 0; i < 3; i++) {
                    glm::dvec3 v0 = trig.vertices[i];
                    glm::dvec3 v1 = trig.vertices[(i + 1) % 3];
                    double p0 = v0[axis];
                    double p1 = v1[axis];
                    if (p0 <= split) {
                        left = left.expand(v0);
                        cnt[0]++;
                    }
                    if (p0 >= split) {
                        right = right.expand(v0);
                        cnt[1]++;
                    }
                    if ((p0 < split && p1 > split) || (p1 < split && p0 > split)) {
                        glm::dvec3 t = lerp(v0, v1, std::max(0.0, std::min(1.0, (split - p0) / (p1 - p0))));
                        // AKR_ASSERT(bbox.contains(t));
                        // AKR_ASSERT(box.contains(t));
                        left = left.expand(t);
                        right = right.expand(t);
                        cnt[0]++;
                        cnt[1]++;
                    }
                }
                left.pmax[axis] = split;
                right.pmin[axis] = split;
                // fmt::print("original: left {} right {} bound:{}\n", left, right, box);
                // AKR_ASSERT(cnt[0] > 1 && cnt[1] > 1);
                // AKR_ASSERT(!box.empty());
                // AKR_ASSERT(!left.empty() && !right.empty());
                auto _left = left.intersect(box);
                auto _right = right.intersect(box);
                // AKR_ASSERT(!_left.empty() && !_right.empty());
                return std::make_pair(TriangleHandle(mesh, idx, _left), TriangleHandle(mesh, idx, _right));
            }
        };
        struct TriangleHandleConstructor {
            using value_type = TriangleHandle;
            auto operator()(const MeshInstance *mesh, int idx) const -> TriangleHandle {
                return TriangleHandle{mesh, idx};
            }
        };

        struct TriangleIntersector {
            auto operator()(const Ray &ray, const TriangleHandle &handle, typename MeshInstance::RayHit &record) const
                -> bool {
                return handle.mesh->intersect(ray, handle.idx, &record);
            }
        };

        using MeshBVH = TBVHAccelerator<const MeshInstance *, typename MeshInstance::RayHit, TriangleIntersector,
                                        TriangleHandleConstructor>;
        using MeshBVHes = BufferView<MeshBVH>;
        struct BVHHandle {
            const MeshBVHes *scene = nullptr;
            int idx;
            Bounds3f box;
            BVHHandle(const MeshBVHes *scene, int idx, Bounds3f box = Bounds3f()) : scene(scene), idx(idx), box(box) {}
            [[nodiscard]] auto full_bbox() const { return (*scene)[idx].boundBox; }
            [[nodiscard]] std::pair<BVHHandle, BVHHandle> split(int axis, Float split) const {
                auto left = box;
                auto right = left;
                auto bbox = left;
                left.pmax[axis] = std::min<Float>(split, bbox.pmax[axis]);
                right.pmin[axis] = std::max<Float>(split, bbox.pmin[axis]);
                return std::make_pair(BVHHandle(scene, idx, left.intersect(box)),
                                      BVHHandle(scene, idx, right.intersect(box)));
            }
        };

        struct BVHHandleConstructor {
            using value_type = BVHHandle;
            auto operator()(const MeshBVHes &scene, int idx) const -> BVHHandle { return BVHHandle{&scene, idx}; }
        };

        struct BVHIntersector {
            auto operator()(const Ray &ray, const BVHHandle &handle, Intersection &record) const -> bool {
                typename MeshInstance::RayHit localHit;
                localHit.t = record.t;
                auto &bvh = (*handle.scene)[handle.idx];
                if (bvh.intersect(ray, localHit) && localHit.t < record.t) {
                    record.t = localHit.t;
                    record.uv = localHit.uv;
                    record.geom_id = handle.idx;
                    record.prim_id = localHit.prim_id;
                    return true;
                }
                return false;
            }
        };
        astd::pmr::vector<MeshBVH> meshBVHs;

        using TopLevelBVH = TBVHAccelerator<MeshBVHes, Intersection, BVHIntersector, BVHHandleConstructor, 20>;
        std::optional<TopLevelBVH> topLevelBVH;

      public:
        BVHAccelerator() : meshBVHs(Allocator<>()) {}
        void build(const Scene &scene) override {
            for (auto &instance : scene.meshes) {
                meshBVHs.emplace_back(&instance, instance.indices.size() / 3);
            }
            topLevelBVH.emplace(MeshBVHes(meshBVHs.data(), meshBVHs.size()), meshBVHs.size());
        }
        std::optional<Intersection> intersect(const Ray &ray) const override {
            Intersection isct;
            auto hit = topLevelBVH->intersect(ray, isct);
            if (hit)
                return isct;
            else
                return std::nullopt;
        }
        bool occlude(const Ray &ray) const override { return topLevelBVH->occlude(ray); }
        void reset() override {
            topLevelBVH.reset();
            meshBVHs.clear();
        }
        Bounds3f world_bounds() const override { return topLevelBVH->boundBox; }
    };
    class SBVHAcceleratorNode final : public AcceleratorNode {
      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {}
        virtual std::shared_ptr<Accelerator> create_accel(const Scene &scene) override {
            auto sbvh = std::make_shared<BVHAccelerator>();
            sbvh->build(scene);
            return sbvh;
        }
    };
    AKR_EXPORT_NODE(SBVH, SBVHAcceleratorNode)
} // namespace akari::render