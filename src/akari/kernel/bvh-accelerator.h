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
              size_t StackDepth = 32>
    struct TBVHAccelerator {
        AKR_IMPORT_TYPES()
        struct alignas(32) BVHNode {
            Bounds3f box{};
            int right = -1;
            int left = -1;
            uint32_t first = (uint32_t)-1;
            uint16_t count = (uint16_t)-1;
            uint16_t axis = (uint16_t)-1;

            [[nodiscard]] AKR_XPU bool is_leaf() const { return count != (uint16_t)-1; }
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

        astd::pmr::vector<Index> refs;
        astd::pmr::vector<BVHNode> nodes;
        std::shared_ptr<std::mutex> m;
        TBVHAccelerator(UserData &&user_data, size_t N, Intersector intersector = Intersector(),
                        ShapeHandleConstructor ctor = ShapeHandleConstructor())
            : user_data(std::move(user_data)), _intersector(std::move(intersector)), _ctor(std::move(ctor)),
              refs(TAllocator<Index>(default_resource())), nodes(TAllocator<BVHNode>(default_resource())) {
            m = std::make_shared<std::mutex>();
            std::vector<Index> primitives;
            for (auto i = 0; i < (int)N; i++) {
                primitives.push_back(Index{i});
            }
            info("Building BVH for {} objects", primitives.size());
            recursiveBuild(std::nullopt, std::move(primitives), 0);
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
        int create_leaf_node(const Bounds3f &box, const std::vector<Index> &primitives) {
            std::lock_guard<std::mutex> _(*m);
            BVHNode node;
            node.box = box;
            node.first = refs.size();
            node.count = primitives.size();
            node.right = -1;
            for (auto i : primitives) {
                refs.emplace_back(i);
            }
            nodes.push_back(node);
            return (int)nodes.size() - 1;
        }
        int recursiveBuild(std::optional<Bounds3f> provided_bounds, std::vector<Index> primitives, int depth) {
            Bounds3f box;
            Bounds3f centroidBound;
            if (primitives.empty())
                return -1;

            for (auto i : primitives) {
                centroidBound = centroidBound.expand(get(i).getBoundingBox().centroid());
            }
            if (!provided_bounds) {
                for (auto i : primitives) {
                    box = box.merge(get(i).getBoundingBox());
                }
            } else {
                box = provided_bounds.value();
            }
            if (depth == 0) {
                boundBox = box;
            }

            if (primitives.size() <= 4 || depth >= 64) {
                if (depth == 64) {
                    warning("BVH exceeds max depth; {} objects", primitives.size());
                }
                return create_leaf_node(box, primitives);
            } else {
                auto size = centroidBound.size();

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
                    Bucket buckets[nBuckets] = {};
                    for (auto i : primitives) {
                        auto offset = centroidBound.offset(get(i).getBoundingBox().centroid())[axis];
                        int b = std::min<int>(nBuckets - 1, (int)std::floor(offset * nBuckets));
                        buckets[b].count++;
                        buckets[b].bound = buckets[b].bound.merge(get(i).getBoundingBox());
                    }
                    double cost[nBuckets - 1] = {0};
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
                    double minCost = cost[0];
                    for (uint32_t i = 1; i < nBuckets - 1; i++) {
                        if (cost[i] <= minCost) {
                            minCost = cost[i];
                            splitBuckets = i;
                        }
                    }
                    AKR_ASSERT(minCost > 0);
                    return SplitInfo{axis, minCost, splitBuckets};
                };
                auto try_split_spatial = [&](int axis) -> std::optional<SplitInfo> {
                    struct Bucket {
                        int count = 0;
                        Bounds3f bound;
                        int enter = 0;
                        int exit = 0;
                        Bucket() = default;
                    };
                    Bucket bins[nBuckets] = {};
                    for (auto i : primitives) {
                        auto bbox = get(i).getBoundingBox();
                        int first =
                            std::clamp<int>(std::floor(box.offset(bbox.pmin)[axis] * nBuckets), 0, nBuckets - 1);
                        int last = std::clamp<int>(std::floor(box.offset(bbox.pmax)[axis] * nBuckets), 0, nBuckets - 1);

                        Bounds3f right_box = bbox;
                        for (int j = first; j < last; j++) {
                            double split = box.pmin[axis] + (j + 1) * box.size()[axis] / nBuckets;
                            auto [left, right] = get(i).split(right_box, axis, split);
                            // fmt::print("left {} right {}\n", left, right);
                            bins[j].bound = bins[j].bound.merge(left);
                            right_box = right;
                        }
                        bins[last].bound = bins[last].bound.merge(right_box);
                        bins[last].exit++;
                        bins[first].enter++;
                        // fmt::print("first {} last {}\n", first, last);
                    }

                    double cost[nBuckets - 1] = {0};
                    int splitBuckets = 0;
                    double minCost = std::numeric_limits<double>::infinity();
                    for (uint32_t i = 0; i < nBuckets - 1; i++) {
                        Bounds3f b0;
                        Bounds3f b1;
                        int count0 = 0, count1 = 0;
                        for (uint32_t j = 0; j <= i; j++) {
                            b0 = b0.merge(bins[j].bound);
                            count0 += bins[j].enter;
                        }
                        for (uint32_t j = i + 1; j < nBuckets; j++) {
                            b1 = b1.merge(bins[j].bound);
                            count1 += bins[j].exit;
                        }
                        float cost0 = count0 == 0 ? 0 : count0 * b0.surface_area();
                        float cost1 = count1 == 0 ? 0 : count1 * b1.surface_area();

                        AKR_ASSERT(cost0 >= 0 && cost1 >= 0 && box.surface_area() >= 0);
                        cost[i] = 0.125f + (cost0 + cost1) / box.surface_area();
                        // fmt::print("#{} {} {} {} {} {}\n", i, count0, count1, cost0, cost1, cost[i]);
                        AKR_ASSERT(cost[i] >= 0);
                        if (cost[i] < minCost) {
                            minCost = cost[i];
                            splitBuckets = i;
                        }
                    }
                    // fmt::print("axis={}\n", axis);
                    for (int i = 0; i < nBuckets; i++) {
                        // fmt::print("bin #{} bounds: {} enter:{} exit:{}\n", i, bins[i].bound, bins[i].enter,
                        //    bins[i].exit);
                        if (i == splitBuckets) {
                            // fmt::print("---------------------------------------------\n");
                        }
                    }
                    AKR_ASSERT(minCost > 0);
                    return SplitInfo{axis, minCost, splitBuckets};
                };
                std::optional<SplitInfo> best_sah_split;
                for (int axis = 0; axis < 3; axis++) {
                    auto candidate = try_split_with_axis(axis);
                    if (!candidate)
                        continue;
                    if (best_sah_split.has_value() && candidate.has_value()) {
                        auto split = best_sah_split.value();
                        auto candidate_split = candidate.value();
                        // debug("size: {} candidate: axis={}, cost={}", size, axis, candidate_split.min_cost);
                        if (split.min_cost > candidate_split.min_cost) {
                            best_sah_split = candidate;
                        }
                    } else {
                        best_sah_split = candidate;
                    }
                }
                std::optional<SplitInfo> best_spatial_split;
                for (int axis = 0; axis < 3; axis++) {
                    auto candidate = try_split_spatial(axis);
                    if (!candidate)
                        continue;
                    if (best_spatial_split.has_value() && candidate.has_value()) {
                        auto split = best_spatial_split.value();
                        auto candidate_split = candidate.value();
                        // debug("size: {} candidate: axis={}, cost={}", size, axis, candidate_split.min_cost);
                        if (split.min_cost > candidate_split.min_cost) {
                            best_spatial_split = candidate;
                        }
                    } else {
                        best_spatial_split = candidate;
                    }
                }
                std::vector<Index> left_partition, right_partition;
                std::optional<Bounds3f> left_bounds, right_bounds;
                int axis;
                if (best_sah_split.has_value() || best_spatial_split.has_value()) {
                    bool use_sah;
                    if (best_sah_split.has_value() && best_spatial_split.has_value()) {
                        use_sah = best_spatial_split.value().min_cost > best_sah_split.value().min_cost;
                    } else if (best_spatial_split) {
                        use_sah = false;
                    } else {
                        use_sah = true;
                    }
                    if (use_sah) {
                        auto best_split = best_sah_split.value();
                        axis = best_split.axis;
                        for (auto i : primitives) {
                            auto b = int(centroidBound.offset(get(i).getBoundingBox().centroid())[axis] * nBuckets);
                            if (b == (int)nBuckets) {
                                b = (int)nBuckets - 1;
                            }
                            if (b <= best_split.split) {
                                left_partition.emplace_back(i);
                            } else {
                                right_partition.emplace_back(i);
                            }
                        }
                    } else {
                        auto best_split = best_spatial_split.value();
                        fmt::print("{}\n", best_split.split);
                        axis = best_split.axis;
                        std::vector<Index> middle;
                        Bounds3f left_box, right_box;

                        double split_pos = box.pmin[axis] + (best_split.split + 1) * box.size()[axis] / nBuckets;
                        for (auto i : primitives) {
                            auto bbox = get(i).getBoundingBox();
                            if (bbox.pmax[axis] < split_pos) {
                                left_partition.emplace_back(i);
                            } else if (bbox.pmin[axis] > split_pos) {
                                right_partition.emplace_back(i);
                            } else {
                                middle.emplace_back(i);
                            }
                        }

                        for (auto i : middle) {
                            Bounds3f B1 = left_box;
                            Bounds3f B2 = right_box;
                            auto N1 = left_partition.size() + 1;
                            auto N2 = right_partition.size() + 1;
                            auto bbox = get(i).getBoundingBox();

                            double Csplit = B1.surface_area() * N1 + B2.surface_area() * N2;
                            double C1 = B1.merge(bbox).surface_area() * N1 + B2.surface_area() * (N2 - 1);
                            double C2 = B1.surface_area() * (N1 - 1) + B2.merge(bbox).surface_area() * N2;
                            if (Csplit < std::min(C1, C2)) {
                                left_partition.emplace_back(i);
                                right_partition.emplace_back(i);
                            } else if (C1 < std::min(C2, Csplit)) {
                                left_partition.emplace_back(i);
                                left_box = left_box.merge(bbox);
                            } else {
                                right_partition.emplace_back(i);
                                right_box = right_box.merge(bbox);
                            }
                        }
                        AKR_ASSERT(!left_partition.empty() && !right_partition.empty());
                        // AKR_ASSERT(left_partition.size() < primitives.size());
                        // AKR_ASSERT(right_partition.size() < primitives.size());
                        left_bounds = left_box;
                        right_bounds = right_box;
                    }
                } else {
                    if (primitives.size() >= 16) {
                        warning("best split cannot be found with {} objects; size: {}", primitives.size(), size);
                        if (hsum(size) == 0) {
                            warning("centroid bound is zero");
                        }
                    }
                    return create_leaf_node(box, primitives);
                }
                size_t ret;
                {
                    std::lock_guard<std::mutex> _(*m);
                    ret = nodes.size();
                    nodes.emplace_back();
                    BVHNode &node = nodes.back();
                    node.axis = axis;
                    node.box = box;
                    node.count = (uint16_t)-1;
                }
                AKR_ASSERT(!left_partition.empty() && !right_partition.empty());
                if (primitives.size() > 64 * 1024) {
                    auto left = std::async(std::launch::async, [&]() -> int {
                        return (int)recursiveBuild(left_bounds, std::move(left_partition), depth + 1);
                    });
                    auto right = std::async(std::launch::async, [&]() -> int {
                        return (int)recursiveBuild(right_bounds, std::move(right_partition), depth + 1);
                    });
                    nodes[ret].left = left.get();
                    nodes[ret].right = right.get();
                } else {
                    nodes[ret].left = (int)recursiveBuild(left_bounds, std::move(left_partition), depth + 1);
                    nodes[ret].right = (int)recursiveBuild(right_bounds, std::move(right_partition), depth + 1);
                }
                AKR_ASSERT(nodes[ret].right >= 0 && nodes[ret].left >= 0);
                return (int)ret;
            }
        }
        AKR_XPU bool intersect_leaf(const BVHNode &node, const Ray3f &ray, Hit &isct) const {
            bool hit = false;
            for (uint32_t i = node.first; i < node.first + node.count; i++) {
                if (_intersector(ray, _ctor(user_data, refs[i].idx), isct)) {
                    hit = true;
                }
            }
            return hit;
        }
        AKR_XPU bool intersect(const Ray3f &ray, Hit &isct) const {
            bool hit = false;
            auto invd = Float3(1) / ray.d;
            constexpr size_t maxDepth = StackDepth;
            const BVHNode *stack[maxDepth];
            int sp = 0;
            const BVHNode *p = &nodes[0];
            while (p) {
                BVHNode node = *p;
                auto t = intersectAABB(node.box, ray, invd);

                if (t < 0 || t > isct.t) {
                    p = sp > 0 ? stack[--sp] : nullptr;
                    continue;
                }
                if (node.is_leaf()) {
                    hit = hit | intersect_leaf(node, ray, isct);
                    p = sp > 0 ? stack[--sp] : nullptr;
                } else {
                    if (ray.d[node.axis] > 0) {
                        stack[sp++] = &nodes[node.right];
                        p = &nodes[node.left];
                    } else {
                        stack[sp++] = &nodes[node.left];
                        p = &nodes[node.right];
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
                BVHNode node = *p;
                auto t = intersectAABB(node.box, ray, invd);

                if (t < 0 || t > ray.tmax) {
                    p = sp > 0 ? stack[--sp] : nullptr;
                    continue;
                }
                if (node.is_leaf()) {
                    for (uint32_t i = node.first; i < node.first + node.count; i++) {
                        if (_intersector(ray, _ctor(user_data, refs[i].idx), isct)) {
                            return true;
                        }
                    }
                    p = sp > 0 ? stack[--sp] : nullptr;
                } else {
                    stack[sp++] = &nodes[node.far_child(ray)];
                    p = &nodes[node.near_child(ray)];
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
                // box.pmax += Float3(0.000001);
                return box;
            }
            [[nodiscard]] std::pair<Bounds3f, Bounds3f> split(const Bounds3f &bound, int axis, Float split) const {
                auto trig = get_triangle<C>(*mesh, idx);
                const uint32_t offsets[] = {2, 0, 1};
                Bounds3f left, right;
                for (int i = 0; i < 3; i++) {
                    float3 v0 = trig.vertices[i];
                    float3 v1 = trig.vertices[offsets[i]];
                    double p0 = v0[axis];
                    double p1 = v1[axis];
                    if (p0 < split) {
                        left = left.expand(v0);
                    } else {
                        right = right.expand(v0);
                    }
                    if ((p0 <= split && p1 >= split) || (p1 <= split && p0 >= split)) {
                        float3 t = lerp(p0, p1, std::max(0.0, std::min(1.0, (split - p0) / (p1 - p0))));
                        left = left.expand(t);
                        right = right.expand(t);
                    }
                }
                // fmt::print("original: left {} right {} bound:{}\n", left, right, bound);
                // AKR_ASSERT(!left.empty() && !right.empty());
                return std::make_pair(left.intersect(bound), right.intersect(bound));
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
            [[nodiscard]] std::pair<Bounds3f, Bounds3f> split(const Bounds3f &bound, int axis, Float split) const {
                auto left = (*scene)[idx].boundBox;
                auto right = left;
                auto bbox = left;
                left.pmax[axis] = std::min<Float>(split, bbox.pmax[axis]);
                right.pmin[axis] = std::max<Float>(split, bbox.pmin[axis]);
                return std::make_pair(left.intersect(bound), right.intersect(bound));
            }
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