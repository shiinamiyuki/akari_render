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
#include <akari/core/math.h>
#include <akari/core/mesh.h>
#include <akari/core/logger.h>
#include <akari/render/scene.h>
#include <optional>
namespace akari::render {
    template <class UserData, class Hit, class Intersector, class ShapeHandleConstructor, size_t StackDepth = 64>
    struct TBVHAccelerator {
        using Ref = typename ShapeHandleConstructor::value_type;
        struct alignas(32) BVHNode {
            Bounds3f box{};
            int right = -1;
            int left = -1;
            uint32_t first = (uint32_t)-1;
            uint16_t count = (uint16_t)-1;
            uint16_t axis = (uint16_t)-1;

            [[nodiscard]] bool is_leaf() const { return count != (uint16_t)-1; }
            int far_child(const Ray &ray) const {
                if (ray.d[axis] > 0) {
                    return right;
                } else {
                    return left;
                }
            }
            int near_child(const Ray &ray) const {
                if (ray.d[axis] > 0) {
                    return left;
                } else {
                    return right;
                }
            }
        };

        Ref get(const int idx) { return _ctor(user_data, idx); }

        bool enable_sbvh = true;
        Bounds3f boundBox;
        UserData user_data;
        Intersector _intersector;
        ShapeHandleConstructor _ctor;

        astd::pmr::vector<int> indicies;
        astd::pmr::vector<BVHNode> nodes;
        std::shared_ptr<std::mutex> m;
        std::atomic<uint32_t> n_splits;
        TBVHAccelerator(TBVHAccelerator &&rhs) : n_splits(rhs.n_splits.load()) {}
        TBVHAccelerator(UserData &&user_data, size_t N, Intersector intersector = Intersector(),
                        ShapeHandleConstructor ctor = ShapeHandleConstructor())
            : user_data(std::move(user_data)), _intersector(std::move(intersector)), _ctor(std::move(ctor)),
              n_splits(0) {
            m = std::make_shared<std::mutex>();
            std::vector<Ref> refs;
            for (auto i = 0; i < (int)N; i++) {
                auto ref = get(i);
                ref.box = ref.full_bbox();
                refs.push_back(ref);
            }
            info("Building BVH for {} objects", refs.size());
            recursiveBuild(std::move(refs), 0);
            info("BVHNodes: {} #splits:{}", nodes.size(), n_splits.load());
        }

        static Float intersectAABB(const Bounds3f &box, const Ray &ray, const Vec3 &invd) {
            Vec3 t0 = (box.pmin - ray.o) * invd;
            Vec3 t1 = (box.pmax - ray.o) * invd;
            Vec3 tMin = min(t0, t1), tMax = max(t0, t1);
            auto m0 = hmax(tMin);
            auto m1 = hmin(tMax);
            if (m0 <= m1) {
                auto t = std::max(ray.tmin, m0);
                if (t >= ray.tmax) {
                    return -1;
                }
                return t;
            }
            return -1;
        }
        static constexpr size_t nBuckets = 32;
        struct SplitInfo {
            int axis;
            double min_cost;
            Bounds3f overlapped;
            int split; // [0, nBuckets)
            double split_pos;
        };
        int create_leaf_node(const Bounds3f &box, const std::vector<Ref> &refs) {
            std::lock_guard<std::mutex> _(*m);
            BVHNode node;
            node.box = box;
            node.first = indicies.size();
            node.count = refs.size();
            node.right = -1;
            for (auto i : refs) {
                indicies.emplace_back(i.idx);
            }
            nodes.push_back(node);
            return (int)nodes.size() - 1;
        }
        int recursiveBuild(std::vector<Ref> refs, int depth) {
            Bounds3f box;
            Bounds3f centroidBound;
            if (refs.empty())
                return -1;

            for (auto i : refs) {
                centroidBound = centroidBound.expand(i.box.centroid());
                box = box.merge(i.box);
            }

            if (depth == 0) {
                boundBox = box;
            }

            if (glm::all(glm::lessThan(box.extents(), Vec3(0.0))) || refs.size() <= 2 || depth >= 62) {
                if (depth == 62) {
                    warning("BVH exceeds max depth; {} objects", refs.size());
                }
                if (refs.size() >= 8) {
                    warning("leaf node with {} objects", refs.size());
                }
                return create_leaf_node(box, refs);
            } else {
                auto size = centroidBound.size();

                auto try_split_with_axis = [&](int axis) -> std::optional<SplitInfo> {
                    if (size[axis] == 0 || box.surface_area() == 0.0) {
                        // debug("box: [{}, {}] size: {}", box.pmin, box.pmin, size);
                        return std::nullopt;
                    }
                    struct Bucket {
                        int count = 0;
                        Bounds3f bound;

                        Bucket() = default;
                    };
                    Bucket buckets[nBuckets] = {};
                    double bucket_size = size[axis] / nBuckets;
                    for (auto i : refs) {
                        auto offset = centroidBound.offset(i.box.centroid())[axis];
                        int b = std::clamp<int>((int)std::floor(offset * nBuckets), 0, nBuckets - 1);
                        buckets[b].count++;
                        buckets[b].bound = buckets[b].bound.merge(i.box);
                    }
                    double cost[nBuckets - 1] = {0};
                    Bounds3f overlapped[nBuckets - 1];
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
                        float cost0 = count0 == 0 ? 0 : count0 * b0.surface_area() / box.surface_area();
                        float cost1 = count1 == 0 ? 0 : count1 * b1.surface_area() / box.surface_area();
                        AKR_ASSERT(cost0 >= 0 && cost1 >= 0 && box.surface_area() >= 0);
                        cost[i] = 0.125f + (cost0 + cost1);
                        overlapped[i] = (b0.intersect(b1));
                        if (!(cost[i] >= 0)) {
                            debug("#objects {}  {} {} {} {}", refs.size(), cost[i], cost0, cost1, box.surface_area(),
                                  size);
                        }
                        // AKR_ASSERT(cost[i] >= 0);
                    }
                    int splitBuckets = 0;
                    double minCost = cost[0];
                    bool cannot_split = false;
                    for (int i = 0; i < nBuckets; i++) {
                        cannot_split = cannot_split || buckets[i].count == refs.size();
                    }
                    for (uint32_t i = 1; i < nBuckets - 1; i++) {

                        if (cost[i] <= minCost) {
                            minCost = cost[i];
                            splitBuckets = i;
                        }
                    }
                    if (cannot_split) {
                        // std::lock_guard<std::mutex> _(*m);
                        // for(int i =0; i <nBuckets;i++){
                        //     debug("#{} {} {}\n",i, buckets[i].count, buckets[i].bound);
                        // }
                        // for(int i =0; i <nBuckets-1;i++){
                        //     debug("cost[{}] = {}\n",i, cost[i]);
                        // }
                        return std::nullopt;
                    }
                    if (minCost > 0)
                        return SplitInfo{axis, minCost, overlapped[splitBuckets], splitBuckets,
                                         box.pmin[axis] + (splitBuckets + 1) * bucket_size};
                    else
                        return std::nullopt;
                };
                auto try_split_spatial = [&](int axis) -> std::optional<SplitInfo> {
                    if (size[axis] <= 0.0 || box.surface_area() == 0.0)
                        return std::nullopt;
                    struct Bucket {
                        int count = 0;
                        Bounds3f bound;
                        int enter = 0;
                        int exit = 0;
                        Bucket() = default;
                    };
                    Bucket bins[nBuckets] = {};
                    double bucket_size = size[axis] / nBuckets;
                    for (auto i : refs) {
                        auto bbox = i.box;
                        int first = std::clamp<int>((double(bbox.pmin[axis]) - double(box.pmin[axis])) / bucket_size, 0,
                                                    nBuckets - 1);
                        int last = std::clamp<int>((double(bbox.pmax[axis]) - double(box.pmin[axis])) / bucket_size,
                                                   first, nBuckets - 1);

                        Ref right_ref = i;
                        for (int j = first; j < last; j++) {
                            double split = double(box.pmin[axis]) + (j + 1) * bucket_size;
                            auto [left, right] = right_ref.split(axis, split);
                            // AKR_ASSERT(!left.box.empty() && !right.box.empty());
                            // fmt::print("left {} right {}\n", left, right);
                            bins[j].bound = bins[j].bound.merge(left.box);
                            right_ref = right;
                        }
                        bins[last].bound = bins[last].bound.merge(right_ref.box);
                        bins[last].exit++;
                        bins[first].enter++;
                        // fmt::print("first {} last {}\n", first, last);
                    }

                    double cost[nBuckets - 1] = {0};
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
                        float cost0 = count0 == 0 ? 0 : count0 * b0.surface_area() / box.surface_area();
                        float cost1 = count1 == 0 ? 0 : count1 * b1.surface_area() / box.surface_area();

                        AKR_ASSERT(cost0 >= 0 && cost1 >= 0 && box.surface_area() >= 0);
                        cost[i] = 0.125f + (cost0 + cost1);
                        // fmt::print("#{} {} {} {} {} {}\n", i, count0, count1, cost0, cost1, cost[i]);

                        AKR_ASSERT(cost[i] >= 0);
                    }

                    int splitBuckets = 0;
                    double minCost = cost[0];
                    bool all_same = true;
                    for (uint32_t i = 1; i < nBuckets - 1; i++) {
                        if (cost[i] != cost[i - 1])
                            all_same = false;
                        if (cost[i] <= minCost) {
                            minCost = cost[i];
                            splitBuckets = i;
                        }
                    }
                    if (all_same) {
                        splitBuckets = nBuckets / 2;
                    }
                    // if (refs.size() <= 40) {
                    //     std::lock_guard<std::mutex> _(*m);
                    //     fmt::print("total {}\n", refs.size());
                    //     for (uint32_t i = 0; i < nBuckets; i++)
                    //         fmt::print("#{} {} {}\n", i, bins[i].enter, bins[i].exit);
                    //     fmt::print("split {}\n", splitBuckets);
                    // }
                    AKR_ASSERT(minCost > 0);
                    return SplitInfo{axis, minCost, Bounds3f(), splitBuckets,
                                     box.pmin[axis] + (splitBuckets + 1) * bucket_size};
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
                bool try_spatial = enable_sbvh && depth <= 40;
                if (best_sah_split) {
                    auto lambda = best_sah_split.value().overlapped;
                    double alpha = 1e-5;
                    if (lambda.surface_area() / boundBox.surface_area() <= alpha) {
                        try_spatial = false;
                    }
                }
                std::optional<SplitInfo> best_spatial_split;
                for (int axis = 0; try_spatial && axis < 3; axis++) {
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
                std::vector<Ref> left_partition, right_partition;
                int axis;
                if (best_sah_split.has_value() || best_spatial_split.has_value()) {
                    bool use_sah;
                    if (best_sah_split.has_value() && best_spatial_split.has_value()) {
                        // debug("size: {}, cost 1: {} cost 2:{}", refs.size(), best_sah_split.value().min_cost,
                        //       best_spatial_split.value().min_cost);
                        use_sah = best_spatial_split.value().min_cost > best_sah_split.value().min_cost;
                    } else if (best_spatial_split) {
                        use_sah = false;
                    } else {
                        use_sah = true;
                    }

                    if (use_sah) {
                        auto best_split = best_sah_split.value();
                        axis = best_split.axis;
                        for (auto i : refs) {

                            // AKR_ASSERT(!i.box.empty());
                            auto b = int(centroidBound.offset(i.box.centroid())[axis] * nBuckets);
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
                        // fmt::print("{} {}\n", best_split.split, best_split.min_cost);
                        axis = best_split.axis;
                        std::vector<Ref> middle;
                        Bounds3f left_box, right_box;

                        double split_pos = best_split.split_pos;
                        for (auto i : refs) {
                            auto bbox = i.box;
                            if (i.box.empty())
                                continue;
                            // AKR_ASSERT(!i.box.empty());
                            if (bbox.pmax[axis] <= split_pos) {
                                left_partition.emplace_back(i);
                                left_box = left_box.merge(bbox);
                            } else if (bbox.pmin[axis] >= split_pos) {
                                right_partition.emplace_back(i);
                                right_box = right_box.merge(bbox);
                            } else {
                                middle.emplace_back(i);
                            }
                        }

                        for (auto i : middle) {
                            auto [left, right] = i.split(axis, split_pos);
                            AKR_ASSERT(!left.box.empty() || !right.box.empty());
                            if (left.box.empty()) {
                                right_partition.emplace_back(i);
                                right_box = right_box.merge(i.box);
                            } else if (right.box.empty()) {
                                left_partition.emplace_back(i);
                                left_box = left_box.merge(i.box);
                            }

                            // if(left.empt)
                            Bounds3f B1 = left_box;
                            Bounds3f B2 = right_box;
                            auto N1 = left_partition.size() + 1;
                            auto N2 = right_partition.size() + 1;
                            auto bbox = i.box;

                            double Csplit = B1.surface_area() * N1 + B2.surface_area() * N2;
                            double C1 = B1.merge(bbox).surface_area() * N1 + B2.surface_area() * (N2 - 1);
                            double C2 = B1.surface_area() * (N1 - 1) + B2.merge(bbox).surface_area() * N2;
                            // debug("{} {} {}\n", Csplit, C1, C2);
                            if (Csplit < std::min(C1, C2)) {
                                left_partition.emplace_back(left);
                                right_partition.emplace_back(right);
                                n_splits++;
                            } else if (C1 < std::min(C2, Csplit)) {
                                left_partition.emplace_back(i);
                                left_box = left_box.merge(bbox);
                            } else {
                                right_partition.emplace_back(i);
                                right_box = right_box.merge(bbox);
                            }
                        }
                        // AKR_ASSERT(!left_box.empty() && !right_box.empty());
                        AKR_ASSERT(!left_partition.empty() && !right_partition.empty());
                        // AKR_ASSERT(left_partition.size() < refs.size());
                        // AKR_ASSERT(right_partition.size() < refs.size());
                    }
                } else {
                    if (refs.size() >= 8) {
                        warning("best split cannot be found with {} objects; size: {}", refs.size(), size);
                        if (hsum(size) == 0) {
                            warning("centroid bound is zero");
                        }
                    }
                    return create_leaf_node(box, refs);
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
                if (refs.size() > 128 * 1024) {
                    auto left = std::async(std::launch::async, [&]() -> int {
                        return (int)recursiveBuild(std::move(left_partition), depth + 1);
                    });
                    auto right = std::async(std::launch::async, [&]() -> int {
                        return (int)recursiveBuild(std::move(right_partition), depth + 1);
                    });
                    nodes[ret].left = left.get();
                    nodes[ret].right = right.get();
                } else {
                    nodes[ret].left = (int)recursiveBuild(std::move(left_partition), depth + 1);
                    nodes[ret].right = (int)recursiveBuild(std::move(right_partition), depth + 1);
                }
                AKR_ASSERT(nodes[ret].right >= 0 && nodes[ret].left >= 0);
                return (int)ret;
            }
        }
        bool intersect_leaf(const BVHNode &node, const Ray &ray, Hit &isct) const {
            bool hit = false;
            uint32_t first = node.first;
            uint32_t count = node.count;
            for (uint32_t i = first; i < first + count; i++) {
                if (_intersector(ray, _ctor(user_data, indicies[i]), isct)) {
                    hit = true;
                }
            }
            return hit;
        }

        bool intersect(const Ray &ray, Hit &isct) const {
            bool hit = false;
            auto invd = Vec3(1) / ray.d;
            constexpr size_t maxDepth = StackDepth;
            const BVHNode *stack[maxDepth];
            int sp = 0;
            const BVHNode *p = &nodes[0];
            while (p) {
                AKR_ASSERT(sp < maxDepth);
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
        [[nodiscard]] bool occlude(const Ray &ray) const {
            Hit isct;
            auto invd = Vec3(1) / ray.d;
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
                        if (_intersector(ray, _ctor(user_data, indicies[i]), isct)) {
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

} // namespace akari::render