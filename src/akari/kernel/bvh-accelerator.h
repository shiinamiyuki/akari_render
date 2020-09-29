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
#include <future>
#include <akari/common/math.h>
#include <akari/kernel/scene.h>
#include <akari/common/mesh.h>
#include <akari/core/logger.h>
#include <optional>
namespace akari {
    template <typename C, class UserData, class Hit, class Intersector, class ShapeHandleConstructor,
              size_t StackDepth = 64>
    struct TBVHAccelerator {
        AKR_IMPORT_TYPES()
        struct alignas(32) BVHNode {
            Bounds3f box{};
            int right = -1;
            int left = -1;
            int parent = -1;
            uint32_t first = (uint32_t)-1;
            uint32_t count = (uint32_t)-1;
            int axis = -1;

            [[nodiscard]] AKR_XPU bool is_leaf() const { return count != -1; }
            AKR_XPU int sibling(const BVHNode *nodes, int idx_this) const {
                if (idx_this == nodes[parent].left) {
                    return nodes[parent].right;
                } else {
                    return nodes[parent].left;
                }
            }
            AKR_XPU int far_child(const Ray3f &ray) const {
                if (ray.d[axis] > 0) {
                    return right;
                } else {
                    return left;
                }
            }
            AKR_XPU int near_child(const Ray3f &ray) const {
                if (ray.d[axis] > 0) {
                    return left;
                } else {
                    return right;
                }
            }
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

        astd::pmr::vector<Index> primitives;
        astd::pmr::vector<BVHNode> nodes;
        std::shared_ptr<std::mutex> m;
        TBVHAccelerator(UserData &&user_data, size_t N, Intersector intersector = Intersector(),
                        ShapeHandleConstructor ctor = ShapeHandleConstructor())
            : user_data(std::move(user_data)), _intersector(std::move(intersector)), _ctor(std::move(ctor)),
              primitives(TAllocator<Index>(default_resource())), nodes(TAllocator<BVHNode>(default_resource())) {
            m = std::make_shared<std::mutex>();
            for (auto i = 0; i < (int)N; i++) {
                primitives.push_back(Index{i});
            }
            info("Building BVH for {} objects", primitives.size());
            recursiveBuild(0, (int)primitives.size(), 0);
            info("BVHNodes: {}", nodes.size());
        }

        AKR_XPU static Float intersectAABB(const Bounds3f &box, const Ray3f &ray, const Float3 &invd) {
            Float3 t0 = (box.pmin - ray.o) * invd;
            Float3 t1 = (box.pmax - ray.o) * invd;
            Float3 tMin = min(t0, t1), tMax = max(t0, t1);
            if (hmax(tMin) <= hmin(tMax)) {
                auto t = std::max(ray.tmin, hmax(tMin));
                if (t >= ray.tmax) {
                    return -1;
                }
                return t;
            }
            return -1;
        }
        static constexpr size_t nBuckets = 16;
        struct SplitInfo {
            int axis;
            double min_cost;
            int split; // [0, nBuckets)
        };
        int create_leaf_node(const Bounds3f &box, int begin, int end) {
            BVHNode node;

            node.box = box;
            node.first = begin;
            node.count = end - begin;
            node.right = -1;
            std::lock_guard<std::mutex> _(*m);
            nodes.push_back(node);
            return (int)nodes.size() - 1;
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

            if (end - begin <= 2 || depth >= 64) {
                if (depth == 64) {
                    warning("BVH exceeds max depth; {} objects", end - begin);
                }
                return create_leaf_node(box, begin, end);
            } else {
                auto size = centroidBound.size();
                Index *mid = nullptr;
                auto try_split_with_axis = [&](int axis) -> std::optional<SplitInfo> {
                    if (size[axis] == 0) {
                        // debug("box: [{}, {}] size: {}", box.pmin, box.pmin, size);
                        return std::nullopt;
                    }
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
                        AKR_ASSERT(cost0 >= 0 && cost1 >= 0 && box.surface_area() >= 0);
                        cost[i] = 0.125f + (cost0 + cost1) / box.surface_area();
                        if (!(cost[i] >= 0)) {
                            debug("{}  {} {} {} {}", cost[i], cost0, cost1, box.surface_area(), size);
                        }
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
                    return SplitInfo{axis, minCost, splitBuckets};
                };
                std::optional<SplitInfo> best_split;
                int axis;
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
                {

                    auto candidate = try_split_with_axis(axis);
                    if (best_split.has_value() && candidate.has_value()) {
                        auto split = best_split.value();
                        auto candidate_split = candidate.value();
                        // debug("candidate: axis={}, cost={}", axis, candidate_split.min_cost);
                        if (split.min_cost > candidate_split.min_cost) {
                            candidate = best_split;
                        }
                    } else {
                        best_split = candidate;
                    }
                }
                // int axis;
                if (best_split.has_value()) {
                    axis = best_split.value().axis;
                    mid = std::partition(&primitives[begin], &primitives[end - 1] + 1, [&](Index &p) {
                        auto b = int(centroidBound.offset(get(p).getBoundingBox().centroid())[axis] * nBuckets);
                        if (b == (int)nBuckets) {
                            b = (int)nBuckets - 1;
                        }
                        return b <= best_split.value().split;
                    });
                    AKR_ASSERT((mid != &primitives[begin]) && (mid != &primitives[end - 1] + 1));
                    // debug("left:{} right:{}", mid - &primitives[begin], &primitives[end - 1] + 1 - mid);
                } else {
                    if (end - begin >= 16) {
                        warning("best split cannot be found with {} objects", end - begin);
                        if (hsum(size) == 0) {
                            warning("centroid bound is zero");
                        }
                    }
                    return create_leaf_node(box, begin, end);
                }
                size_t ret;
                {
                    std::lock_guard<std::mutex> _(*m);
                    ret = nodes.size();
                    nodes.emplace_back();
                    BVHNode &node = nodes.back();
                    node.axis = axis;
                    node.box = box;
                    node.count = (uint32_t)-1;
                }

                if (end - begin > 128 * 1024) {
                    auto left = std::async(std::launch::async, [&]() {
                        return (int)recursiveBuild(begin, int(mid - &primitives[0]), depth + 1);
                    });
                    auto right = std::async(std::launch::async, [&]() {
                        return (int)recursiveBuild(int(mid - &primitives[0]), end, depth + 1);
                    });
                    nodes[ret].left = left.get();
                    nodes[ret].right = right.get();
                } else {
                    nodes[ret].left = (int)recursiveBuild(begin, int(mid - &primitives[0]), depth + 1);
                    nodes[ret].right = (int)recursiveBuild(int(mid - &primitives[0]), end, depth + 1);
                }
                nodes[nodes[ret].left].parent = ret;
                nodes[nodes[ret].right].parent = ret;
                AKR_ASSERT(nodes[ret].right >= 0 && nodes[ret].left >= 0);
                return (int)ret;
            }
        }
        AKR_XPU bool intersect_leaf(const BVHNode &node, const Ray3f &ray, Hit &isct) const {
            bool hit = false;
            for (uint32_t i = node.first; i < node.first + node.count; i++) {
                if (_intersector(ray, _ctor(user_data, primitives[i].idx), isct)) {
                    hit = true;
                }
            }
            return hit;
        }
        AKR_XPU bool intersect_confused(const Ray3f &ray, Hit &isct) const {
            if (nodes[0].is_leaf()) {
                return intersect_leaf(nodes[0], ray, isct);
            }
            auto invd = Float3(1) / ray.d;
            bool hit = false;
            int current = nodes[0].near_child(ray);
            int last = 0;

            while (true) {
                if (current == 0)
                    return hit;
                int near = nodes[current].near_child(ray);
                int far = nodes[current].far_child(ray);
                if (last == far) {
                    last = current;
                    current = nodes[current].parent;
                    continue;
                }
                int try_child = (last == nodes[current].parent) ? near : far;
                Float t = intersectAABB(nodes[current].box, ray, invd);
                if (t > 0 && t < isct.t) {
                    if (nodes[current].is_leaf()) {
                        hit = hit | intersect_leaf(nodes[current], ray, isct);
                    }
                    last = current;
                    current = try_child;
                } else {
                    if (try_child == near) {
                        last = near;
                    } else {
                        last = current;
                        current = nodes[current].parent;
                    }
                }
            }
        }
        AKR_XPU bool intersect(const Ray3f &ray, Hit &isct) const {
            return intersect_stackfull(ray, isct);
        }
        AKR_XPU bool intersect_stackless(const Ray3f &ray, Hit &isct) const {
            if (nodes[0].is_leaf()) {
                return intersect_leaf(nodes[0], ray, isct);
            }
            enum State { FromParent, FromChild, FromSibling };
            auto invd = Float3(1) / ray.d;
            bool hit = false;
            State state = FromParent;
            int current = nodes[0].near_child(ray);
            while (true) {
                bool test_leaf = false;
                int leaf = current;
                switch (state) {
                case FromChild:
                    if (current == 0)
                        return hit;
                    if (current == nodes[nodes[current].parent].near_child(ray)) {
                        current = nodes[current].sibling(nodes.data(), current);
                        state = FromSibling;
                    } else {
                        current = nodes[current].parent;
                        state = FromChild;
                    }
                    break;

                case FromSibling: {
                    Float t = intersectAABB(nodes[current].box, ray, invd);
                    if (t < 0 || t > isct.t) {
                        current = nodes[current].parent;
                        state = FromChild;
                    } else if (nodes[current].is_leaf()) {
                        test_leaf = true;
                        current = nodes[current].parent;
                        state = FromChild;
                    } else {
                        current = nodes[current].near_child(ray);
                        state = FromParent;
                    }
                } break;
                case FromParent: {
                    Float t = intersectAABB(nodes[current].box, ray, invd);
                    if (t < 0 || t > isct.t) {
                        current = nodes[current].sibling(nodes.data(), current);
                        state = FromSibling;
                    } else if (nodes[current].is_leaf()) {
                        test_leaf = true;
                        current = nodes[current].sibling(nodes.data(), current);
                        state = FromSibling;
                    } else {
                        current = nodes[current].near_child(ray);
                        state = FromParent;
                    }
                } break;
                }
                if (test_leaf) {
                    hit = hit | intersect_leaf(nodes[leaf], ray, isct);
                }
            }
        }
        AKR_XPU bool intersect_stackless_slow(const Ray3f &ray, Hit &isct) const {
            if (nodes[0].is_leaf()) {
                return intersect_leaf(nodes[0], ray, isct);
            }
            enum State { FromParent, FromChild, FromSibling };
            auto invd = Float3(1) / ray.d;
            bool hit = false;
            State state = FromParent;
            int current = nodes[0].near_child(ray);
            while (true) {
                switch (state) {
                case FromChild:
                    if (current == 0)
                        return hit;
                    if (current == nodes[nodes[current].parent].near_child(ray)) {
                        current = nodes[current].sibling(nodes.data(), current);
                        state = FromSibling;
                    } else {
                        current = nodes[current].parent;
                        state = FromChild;
                    }
                    break;

                case FromSibling: {
                    Float t = intersectAABB(nodes[current].box, ray, invd);
                    if (t < 0 || t > isct.t) {
                        current = nodes[current].parent;
                        state = FromChild;
                    } else if (nodes[current].is_leaf()) {
                        hit = hit | intersect_leaf(nodes[current], ray, isct);
                        current = nodes[current].parent;
                        state = FromChild;
                    } else {
                        current = nodes[current].near_child(ray);
                        state = FromParent;
                    }
                } break;
                case FromParent: {
                    Float t = intersectAABB(nodes[current].box, ray, invd);
                    if (t < 0 || t > isct.t) {
                        current = nodes[current].sibling(nodes.data(), current);
                        state = FromSibling;
                    } else if (nodes[current].is_leaf()) {
                        hit = hit | intersect_leaf(nodes[current], ray, isct);
                        current = nodes[current].sibling(nodes.data(), current);
                        state = FromSibling;
                    } else {
                        current = nodes[current].near_child(ray);
                        state = FromParent;
                    }
                } break;
                }
            }
        }
        AKR_XPU bool intersect_stackfull(const Ray3f &ray, Hit &isct) const {
            bool hit = false;
            auto invd = Float3(1) / ray.d;
            constexpr size_t maxDepth = StackDepth;
            const BVHNode *stack[maxDepth];
            int sp = 0;
            const BVHNode *p = &nodes[0];
            while (p) {
                auto t = intersectAABB(p->box, ray, invd);

                if (t < 0 || t > isct.t) {
                    p = sp > 0 ? stack[--sp] : nullptr;
                    continue;
                }
                if (p->is_leaf()) {
                    hit = hit | intersect_leaf(*p, ray, isct);
                    p = sp > 0 ? stack[--sp] : nullptr;
                } else {
                    if (ray.d[p->axis] > 0) {
                        stack[sp++] = &nodes[p->right];
                        p = &nodes[p->left];
                    } else {
                        stack[sp++] = &nodes[p->left];
                        p = &nodes[p->right];
                    }
                }
            }
            return hit;
        }
        AKR_XPU [[nodiscard]] bool occlude(const Ray3f &ray) const {
            Hit isct;
            auto invd = Float3(1) / ray.d;
            constexpr size_t maxDepth = StackDepth;
            const BVHNode *stack[maxDepth];
            int sp = 0;
            const BVHNode *p = &nodes[0];
            while (p) {
                auto t = intersectAABB(p->box, ray, invd);

                if (t < 0 || t > ray.tmax) {
                    p = sp > 0 ? stack[--sp] : nullptr;
                    continue;
                }
                if (p->is_leaf()) {
                    for (uint32_t i = p->first; i < p->first + p->count; i++) {
                        if (_intersector(ray, _ctor(user_data, primitives[i].idx), isct)) {
                            return true;
                        }
                    }
                    p = sp > 0 ? stack[--sp] : nullptr;
                } else {
                    stack[sp++] = &nodes[p->far_child(ray)];
                    p = &nodes[p->near_child(ray)];
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
                box.pmax += Float3(0.000001);
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
                    record.geom_id = handle.idx;
                    record.prim_id = localHit.prim_id;
                    return true;
                }
                return false;
            }
        };
        astd::pmr::vector<MeshBVH> meshBVHs;

        using TopLevelBVH = TBVHAccelerator<C, MeshBVHes, Intersection<C>, BVHIntersector, BVHHandleConstructor, 20>;
        astd::optional<TopLevelBVH> topLevelBVH;

      public:
        BVHAccelerator() : meshBVHs(TAllocator<MeshBVH>(default_resource())) {}
        void build(Scene<C> &scene) {
            for (auto &instance : scene.meshes) {
                meshBVHs.emplace_back(&instance, instance.indices.size() / 3);
            }
            topLevelBVH.emplace(MeshBVHes(meshBVHs.data(), meshBVHs.size()), meshBVHs.size());
        }
        AKR_XPU bool intersect(const Ray<C> &ray, Intersection<C> *isct) const {
            return topLevelBVH->intersect(ray, *isct);
        }
        AKR_XPU bool occlude(const Ray<C> &ray) const { return topLevelBVH->occlude(ray); }
    };
} // namespace akari