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
#include <akari/core/buffer.h>
#include <akari/kernel/meshview.h>
namespace akari {
    template <class Float, class UserData, class ShapeHandleConstructor>
    struct BVHBuilder {
        AKR_IMPORT_CORE_TYPES()
        struct BVHNode {
            Bounds3f box{};
            uint32_t first = (uint32_t)-1;
            uint32_t count = (uint32_t)-1;
            int left = -1, right = -1;

            [[nodiscard]] bool is_leaf() const { return left < 0 && right < 0; }
        };

        struct Index {
            int idx;
        };
        static_assert(sizeof(Index) == sizeof(int));

        auto get(const Index &handle) { return _ctor(user_data, handle.idx); }

        Bounds3f boundBox;
        const UserData *user_data;
        ShapeHandleConstructor _ctor;
        Buffer<Index> primitives;
        Buffer<BVHNode> nodes;

        BVHBuilder(const UserData *user_data, size_t N, ShapeHandleConstructor ctor = ShapeHandleConstructor())
            : user_data(user_data), _ctor(std::move(ctor)) {
            for (auto i = 0; i < (int)N; i++) {
                primitives.push_back(Index{i});
            }
            info("Building BVH for {} objects\n", primitives.size());
            recursiveBuild(0, (int)primitives.size(), 0);
            info("BVHNodes: {}\n", nodes.size());
        }
        int recursiveBuild(int begin, int end, int depth) {
            auto MaxFloat = Constants<Float>::Inf();
            auto MinFLoat = -MaxFloat;
            Bounds3f box{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
            Bounds3f centroidBound{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};

            if (end == begin)
                return -1;
            for (auto i = begin; i < end; i++) {
                box = box.merge(get(primitives[i]).getBoundingBox());
                centroidBound = centroidBound.merge(get(primitives[i]).getBoundingBox().centroid());
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
                        auto offset = centroidBound.offset(get(primitives[i]).getBoundingBox().centroid())[axis];
                        int b = std::min<int>(nBuckets - 1, (int)std::floor(offset * nBuckets));
                        buckets[b].count++;
                        buckets[b].bound = buckets[b].bound.merge(get(primitives[i]).getBoundingBox());
                    }
                    Float cost[nBuckets - 1] = {0};
                    for (uint32_t i = 0; i < nBuckets - 1; i++) {
                        Bounds3f b0{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
                        Bounds3f b1{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
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
                        assert(cost[i] >= 0);
                    }
                    int splitBuckets = 0;
                    Float minCost = cost[0];
                    for (uint32_t i = 1; i < nBuckets - 1; i++) {
                        if (cost[i] <= minCost) {
                            minCost = cost[i];
                            splitBuckets = i;
                        }
                    }
                    assert(minCost > 0);
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
    };
    template <typename Float>
    struct TwoLevelMeshBVH {
        struct TriangleHandle {
            AKR_IMPORT_CORE_TYPES()
            const Mesh *mesh = nullptr;
            int idx = -1;
            [[nodiscard]] Bounds3f getBoundingBox() const {
                auto MaxFloat = Constants<Float>::Inf();
                auto MinFLoat = -MaxFloat;
                auto trig = get_triangle(*mesh, idx);
                Bounds3f box{{MaxFloat, MaxFloat, MaxFloat}, {MinFloat, MinFloat, MinFloat}};
                box = box.expand(triangle.vertices[0]);
                box = box.expand(triangle.vertices[1]);
                box = box.expand(triangle.vertices[2]);
                return box;
            }
        };
        struct TriangleHandleConstructor {
            auto operator()(const Mesh *mesh, int idx) const -> TriangleHandle { return TriangleHandle{mesh, idx}; }
        };
        using MeshBVHBuillder = BVHBuillder<Float, const Mesh, TriangleHandleConstructor>;
    
        struct BVHHandle {
            BufferView<const MeshBVHes>scene = nullptr;
            int idx;

            [[nodiscard]] auto getBoundingBox() const { return (*scene)[idx].boundBox; }
        };

        struct BVHHandleConstructor {
            auto operator()(const MeshBVHes *scene, int idx) const -> BVHHandle { return BVHHandle{scene, idx}; }
        };
    };

} // namespace akari