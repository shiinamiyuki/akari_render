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

#include <Akari/Core/Logger.h>
#include <Akari/Core/Plugin.h>
#include <Akari/Render/Accelerator.h>
#include <Akari/Render/Mesh.h>
#include <Akari/Render/Scene.h>
#include <optional>

namespace Akari {
    template <class UserData, class Hit, class Intersector, class ShapeHandleConstructor> struct TBVHAccelerator {
        Intersector _intersector;
        ShapeHandleConstructor _ctor;

        struct BVHNode {
            Bounds3f box{};
            uint32_t first = -1;
            uint32_t count = -1;
            int left = -1, right = -1;

            [[nodiscard]] bool is_leaf() const { return left < 0 && right < 0; }
        };

        struct Index {
            int idx;
        };

        auto get(const Index &handle) { return _ctor(user_data, handle.idx); }

        Bounds3f boundBox;
        const UserData *user_data;
        std::vector<Index> primitives;
        std::vector<BVHNode> nodes;

        TBVHAccelerator(const UserData *user_data, size_t N, Intersector intersector = Intersector(),
                        ShapeHandleConstructor ctor = ShapeHandleConstructor())
            : user_data(user_data), _intersector(std::move(intersector)), _ctor(std::move(ctor)) {
            for (auto i = 0; i < (int)N; i++) {
                primitives.push_back(Index{i});
            }
            printf("Buiding BVH for %zd objects\n", primitives.size());
            recursiveBuild(0, (int)primitives.size(), 0);
            printf("BVHNodes: %zd\n", nodes.size());
        }

        static Float intersectAABB(const Bounds3f &box, const Ray &ray, const vec3 &invd) {
            vec3 t0 = (box.p_min - ray.o) * invd;
            vec3 t1 = (box.p_max - ray.o) * invd;
            vec3 tMin = min(t0, t1), tMax = max(t0, t1);
            if (MaxComp(tMin) <= MinComp(tMax)) {
                auto t = std::max(ray.t_min + Eps, MaxComp(tMin));
                if (t >= ray.t_max + Eps) {
                    return -1;
                }
                return t;
            }
            return -1;
        }

        int recursiveBuild(int begin, int end, int depth) {
            Bounds3f box{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
            Bounds3f centroidBound{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};

            if (end == begin)
                return -1;
            for (auto i = begin; i < end; i++) {
                box = box.UnionOf(get(primitives[i]).getBoundingBox());
                centroidBound = centroidBound.UnionOf(get(primitives[i]).getBoundingBox().Centroid());
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
                auto size = centroidBound.Size();
                if (size.x > size.y) {
                    if (size.x > size.z) {
                        axis = 0;
                    } else {
                        axis = 2;
                    }
                } else {
                    if (size.y > size.z) {
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

                        Bucket() : bound({{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}}) {}
                    };
                    Bucket buckets[nBuckets];
                    for (int i = begin; i < end; i++) {
                        auto offset = centroidBound.offset(get(primitives[i]).getBoundingBox().Centroid())[axis];
                        int b = std::min<int>(nBuckets - 1, (int)std::floor(offset * nBuckets));
                        buckets[b].count++;
                        buckets[b].bound = buckets[b].bound.UnionOf(get(primitives[i]).getBoundingBox());
                    }
                    Float cost[nBuckets - 1] = {0};
                    for (int i = 0; i < nBuckets - 1; i++) {
                        Bounds3f b0{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
                        Bounds3f b1{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
                        int count0 = 0, count1 = 0;
                        for (int j = 0; j <= i; j++) {
                            b0 = b0.UnionOf(buckets[j].bound);
                            count0 += buckets[j].count;
                        }
                        for (int j = i + 1; j < nBuckets; j++) {
                            b1 = b1.UnionOf(buckets[j].bound);
                            count1 += buckets[j].count;
                        }
                        float cost0 = count0 == 0 ? 0 : count0 * b0.SurfaceArea();
                        float cost1 = count1 == 0 ? 0 : count1 * b1.SurfaceArea();
                        cost[i] = 0.125f + (cost0 + cost1) / box.SurfaceArea();
                        assert(cost[i]>=0);
                    }
                    int splitBuckets = 0;
                    Float minCost = cost[0];
                    for (int i = 1; i < nBuckets - 1; i++) {
                        if (cost[i] <= minCost) {
                            minCost = cost[i];
                            splitBuckets = i;
                        }
                    }
                    assert(minCost > 0);
                    mid = std::partition(&primitives[begin], &primitives[end - 1] + 1, [&](Index &p) {
                        auto b = int(centroidBound.offset(get(p).getBoundingBox().Centroid())[axis] * nBuckets);
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
                node.count = -1;
                nodes.push_back(node);
                nodes[ret].left = (int)recursiveBuild(begin, int(mid - &primitives[0]), depth + 1);
                nodes[ret].right = (int)recursiveBuild(int(mid - &primitives[0]), end, depth + 1);

                return (int)ret;
            }
        }

        bool intersect(const Ray &ray, Hit &isct) const {
            bool hit = false;
            auto invd = vec3(1) / ray.d;
            constexpr int maxDepth = 64;
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
        bool occlude(const Ray &ray) const {
            Hit isct;
            auto invd = vec3(1) / ray.d;
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

    class BVHAccelerator final: public Accelerator {
        struct TriangleHandle {
            const Mesh *mesh = nullptr;
            int idx = -1;
            [[nodiscard]] Bounds3f getBoundingBox() const {
                auto v = mesh->GetVertexBuffer();
                auto i = mesh->GetIndexBuffer();
                Bounds3f box{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
                box = box.UnionOf(v[i[idx * 3 + 0]].pos);
                box = box.UnionOf(v[i[idx * 3 + 1]].pos);
                box = box.UnionOf(v[i[idx * 3 + 2]].pos);
                return box;
            }
        };
        struct TriangleHandleConstructor {
            auto operator()(const Mesh *mesh, int idx) const -> TriangleHandle { return TriangleHandle{mesh, idx}; }
        };

        struct TriangleIntersector {
            auto operator()(const Ray &ray, const TriangleHandle &handle, Mesh::RayHit &record) const -> bool {
                return handle.mesh->Intersect(ray, handle.idx, &record);
            }
        };

        using MeshBVH = TBVHAccelerator<const Mesh, Mesh::RayHit, TriangleIntersector, TriangleHandleConstructor>;
        using MeshBVHes = std::vector<MeshBVH>;
        struct BVHHandle {
            const MeshBVHes *scene = nullptr;
            int idx;

            [[nodiscard]] auto getBoundingBox() const { return (*scene)[idx].boundBox; }
        };

        struct BVHHandleConstructor {
            auto operator()(const MeshBVHes *scene, int idx) const -> BVHHandle { return BVHHandle{scene, idx}; }
        };

        struct BVHIntersector {
            auto operator()(const Ray &ray, const BVHHandle &handle, Intersection &record) const -> bool {
                Mesh::RayHit localHit;
                localHit.t = record.t;
                auto &bvh = (*handle.scene)[handle.idx];
                if (bvh.intersect(ray, localHit) && localHit.t < record.t) {
                    record.t = localHit.t;
                    record.uv = localHit.uv;
                    record.Ng = localHit.Ng;
                    record.meshId = handle.idx;
                    record.primId = localHit.face;
                    return true;
                }
                return false;
            }
        };
        std::vector<MeshBVH> meshBVHs;
        using TopLevelBVH = TBVHAccelerator<MeshBVHes, Intersection, BVHIntersector, BVHHandleConstructor>;
        std::optional<TopLevelBVH> topLevelBVH;

      public:
        AKR_DECL_COMP(BVHAccelerator, "BVHAccelerator")
        void Build(const Scene &scene) override {
            for (auto &mesh : scene.GetMeshes()) {
                meshBVHs.emplace_back(mesh.get(), mesh->GetTriangleCount());
            }
            topLevelBVH.emplace(&meshBVHs, meshBVHs.size());
        }
        bool Intersect(const Ray &ray, Intersection *intersection) const override {
            return topLevelBVH->intersect(ray, *intersection);
        }
        [[nodiscard]] bool Occlude(const Ray &ray) const override { return topLevelBVH->occlude(ray); }
    };
    AKR_EXPORT_COMP(BVHAccelerator, "Accelerator")
} // namespace Akari