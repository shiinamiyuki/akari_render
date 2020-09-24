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

#include <akari/common/math.h>
#include <akari/kernel/scene.h>
#include <akari/common/mesh.h>
#include <akari/core/logger.h>
namespace akari {
    template <typename C, class UserData, class Hit, class Intersector, class ShapeHandleConstructor>
    struct TBVHAccelerator {
        AKR_IMPORT_TYPES()
        struct BVHNode {
            Bounds3f box{};
            uint32_t first = (uint32_t)-1;
            uint32_t count = (uint32_t)-1;
            int left = -1, right = -1;

            [[nodiscard]] AKR_XPU bool is_leaf() const { return left < 0 && right < 0; }
        };

        struct Index {
            int idx;
        };
        static_assert(sizeof(Index) == sizeof(int));

        AKR_XPU auto get(const Index &handle) { return _ctor(user_data, handle.idx); }

        Bounds3f boundBox;
        UserData user_data;
        Intersector _intersector;
        ShapeHandleConstructor _ctor;
        Buffer<Index> primitives;
        Buffer<BVHNode> nodes;

        TBVHAccelerator(UserData &&user_data, size_t N, Intersector intersector = Intersector(),
                        ShapeHandleConstructor ctor = ShapeHandleConstructor())
            : user_data(std::move(user_data)), _intersector(std::move(intersector)), _ctor(std::move(ctor)) {
            for (auto i = 0; i < (int)N; i++) {
                primitives.push_back(Index{i});
            }
            info("Building BVH for {} objects", primitives.size());
            recursiveBuild(0, (int)primitives.size(), 0);
            info("BVHNodes: {}", nodes.size());
        }

        AKR_XPU static Float intersectAABB(const Bounds3f &box, const Ray3f &ray, const Vector3f &invd) {
            Vector3f t0 = (box.pmin - ray.o) * invd;
            Vector3f t1 = (box.pmax - ray.o) * invd;
            Vector3f tMin = min(t0, t1), tMax = max(t0, t1);
            // debug("t0: {} t1: {} tmin:{} tmax:{}\n", t0, t1, tMin, tMax);
            if (hmax(tMin) <= hmin(tMax)) {
                auto t = std::max(ray.tmin, hmax(tMin));
                if (t >= ray.tmax) {
                    return -1;
                }
                return t;
            }
            return -1;
        }

        int recursiveBuild(int begin, int end, int depth) {
            Bounds3f box;
            Bounds3f centroidBound;

            if (end == begin)
                return -1;
            for (auto i = begin; i < end; i++) {
                box = box.merge(get(primitives[i]).getBoundingBox());
                centroidBound = centroidBound.expand(get(primitives[i]).getBoundingBox().centroid());
            }

            if (depth == 0) {
                boundBox = box;
            }

            if (end - begin <= 4 || depth >= 32) {
                BVHNode node;

                node.box = box;
                node.first = begin;
                node.count = end - begin;
                node.left = node.right = -1;
                nodes.push_back(node);
                return (int)nodes.size() - 1;
            } else {

                int axis = depth % 3;
                auto size = centroidBound.size();
                if (size.x() > size.y()) {
                    if (size.x() > size.z()) {
                        axis = 0;
                    } else {
                        axis = 2;
                    }
                } else {
                    if (size.y() > size.z()) {
                        axis = 1;
                    } else {
                        axis = 2;
                    }
                }
                Index *mid = nullptr;
                if (size[axis] > 0) {
                    constexpr size_t nBuckets = 12;
                    struct Bucket {
                        int count = 0;
                        Bounds3f bound;

                        Bucket() = default;
                    };
                    Bucket buckets[nBuckets];
                    for (int i = begin; i < end; i++) {
                        auto offset = centroidBound.offset(get(primitives[i]).getBoundingBox().centroid())[axis];
                        int b = std::min<int>(nBuckets - 1, (int)std::floor(offset * nBuckets));
                        buckets[b].count++;
                        buckets[b].bound = buckets[b].bound.merge(get(primitives[i]).getBoundingBox());
                    }
                    Float cost[nBuckets - 1] = {0};
                    for (uint32_t i = 0; i < nBuckets - 1; i++) {
                        Bounds3f b0;
                        Bounds3f b1;
                        int count0 = 0, count1 = 0;
                        for (uint32_t j = 0; j <= i; j++) {
                            b0 = b0.merge(buckets[j].bound);
                            count0 += buckets[j].count;
                        }
                        for (uint32_t j = i + 1; j < nBuckets; j++) {
                            b1 = b1.merge(buckets[j].bound);
                            count1 += buckets[j].count;
                        }
                        float cost0 = count0 == 0 ? 0 : count0 * b0.surface_area();
                        float cost1 = count1 == 0 ? 0 : count1 * b1.surface_area();
                        cost[i] = 0.125f + (cost0 + cost1) / box.surface_area();
                        AKR_ASSERT(cost[i] >= 0);
                    }
                    int splitBuckets = 0;
                    Float minCost = cost[0];
                    for (uint32_t i = 1; i < nBuckets - 1; i++) {
                        if (cost[i] <= minCost) {
                            minCost = cost[i];
                            splitBuckets = i;
                        }
                    }
                    AKR_ASSERT(minCost > 0);
                    mid = std::partition(&primitives[begin], &primitives[end - 1] + 1, [&](Index &p) {
                        auto b = int(centroidBound.offset(get(p).getBoundingBox().centroid())[axis] * nBuckets);
                        if (b == (int)nBuckets) {
                            b = (int)nBuckets - 1;
                        }
                        return b <= splitBuckets;
                    });
                } else {
                    mid = primitives.data() + (begin + end) / 2;
                }
                auto ret = nodes.size();
                nodes.emplace_back();

                BVHNode &node = nodes.back();
                node.box = box;
                node.count = (uint32_t)-1;
                nodes.push_back(node);
                nodes[ret].left = (int)recursiveBuild(begin, int(mid - &primitives[0]), depth + 1);
                nodes[ret].right = (int)recursiveBuild(int(mid - &primitives[0]), end, depth + 1);

                return (int)ret;
            }
        }

        AKR_XPU bool intersect(const Ray3f &ray, Hit &isct) const {
            bool hit = false;
            auto invd = Vector3f(1) / ray.d;
            constexpr int maxDepth = 40;
            const BVHNode *stack[maxDepth];
            int sp = 0;
            stack[sp++] = &nodes[0];
            while (sp > 0) {

                auto p = stack[--sp];
                auto t = intersectAABB(p->box, ray, invd);

                if (t < 0 || t > isct.t) {
                    continue;
                }
                if (p->is_leaf()) {
                    for (uint32_t i = p->first; i < p->first + p->count; i++) {
                        if (_intersector(ray, _ctor(user_data, primitives[i].idx), isct)) {
                            hit = true;
                        }
                    }
                } else {
                    if (p->left >= 0)
                        stack[sp++] = &nodes[p->left];
                    if (p->right >= 0)
                        stack[sp++] = &nodes[p->right];
                }
            }
            return hit;
        }
        AKR_XPU [[nodiscard]] bool occlude(const Ray3f &ray) const {
            Hit isct;
            auto invd = Vector3f(1) / ray.d;
            constexpr int maxDepth = 64;
            const BVHNode *stack[maxDepth];
            int sp = 0;
            stack[sp++] = &nodes[0];
            while (sp > 0) {
                auto p = stack[--sp];
                auto t = intersectAABB(p->box, ray, invd);

                if (t < 0) {
                    continue;
                }
                if (p->is_leaf()) {
                    for (uint32_t i = p->first; i < p->first + p->count; i++) {
                        if (_intersector(ray, _ctor(user_data, primitives[i].idx), isct)) {
                            return true;
                        }
                    }
                } else {
                    if (p->left >= 0)
                        stack[sp++] = &nodes[p->left];
                    if (p->right >= 0)
                        stack[sp++] = &nodes[p->right];
                }
            }
            return false;
        }
    };

    template <typename C>
    class BVHAccelerator {
        AKR_IMPORT_TYPES()
        struct TriangleHandle {
            const MeshInstance<C> *mesh = nullptr;
            int idx = -1;
            [[nodiscard]] AKR_XPU Bounds3f getBoundingBox() const {
                auto trig = get_triangle<C>(*mesh, idx);
                Bounds3f box;
                box = box.expand(trig.vertices[0]);
                box = box.expand(trig.vertices[1]);
                box = box.expand(trig.vertices[2]);
                return box;
            }
        };
        struct TriangleHandleConstructor {
            AKR_XPU auto operator()(const MeshInstance<C> *mesh, int idx) const -> TriangleHandle {
                return TriangleHandle{mesh, idx};
            }
        };

        struct TriangleIntersector {
            AKR_XPU auto operator()(const Ray3f &ray, const TriangleHandle &handle,
                                    typename MeshInstance<C>::RayHit &record) const -> bool {
                return handle.mesh->intersect(ray, handle.idx, &record);
            }
        };

        using MeshBVH = TBVHAccelerator<C, const MeshInstance<C> *, typename MeshInstance<C>::RayHit,
                                        TriangleIntersector, TriangleHandleConstructor>;
        using MeshBVHes = BufferView<MeshBVH>;
        struct BVHHandle {
            const MeshBVHes *scene = nullptr;
            int idx;

            [[nodiscard]] AKR_XPU auto getBoundingBox() const { return (*scene)[idx].boundBox; }
        };

        struct BVHHandleConstructor {
            AKR_XPU auto operator()(const MeshBVHes &scene, int idx) const -> BVHHandle {
                return BVHHandle{&scene, idx};
            }
        };

        struct BVHIntersector {
            AKR_XPU auto operator()(const Ray3f &ray, const BVHHandle &handle, Intersection<C> &record) const -> bool {
                typename MeshInstance<C>::RayHit localHit;
                localHit.t = record.t;
                auto &bvh = (*handle.scene)[handle.idx];
                if (bvh.intersect(ray, localHit) && localHit.t < record.t) {
                    record.t = localHit.t;
                    record.uv = localHit.uv;
                    record.ng = localHit.ng;
                    record.geom_id = handle.idx;
                    record.prim_id = localHit.prim_id;
                    return true;
                }
                return false;
            }
        };
        Buffer<MeshBVH> meshBVHs;
        using TopLevelBVH = TBVHAccelerator<C, MeshBVHes, Intersection<C>, BVHIntersector, BVHHandleConstructor>;
        astd::optional<TopLevelBVH> topLevelBVH;

      public:
        void build(Scene<C> &scene) {
            for (auto &instance : scene.meshes) {
                meshBVHs.emplace_back(&instance, instance.indices.size() / 3);
            }
            topLevelBVH.emplace(meshBVHs, meshBVHs.size());
        }
        AKR_XPU bool intersect(const Ray<C> &ray, Intersection<C> *isct) const {
            return topLevelBVH->intersect(ray, *isct);
        }
        AKR_XPU bool occlude(const Ray<C> &ray) const { return topLevelBVH->occlude(ray); }
    };
} // namespace akari