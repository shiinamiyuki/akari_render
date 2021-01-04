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

#pragma once
#include <akari/util.h>
#include <akari/pmr.h>
#include <algorithm>
#include <execution>
namespace akari::render {

    /*
    Point {
        vec3 p()const;
    }
    */
    template <typename Point>
    class KDTree {
        struct KDTreeNode {
            int axis = -1;
            double split = 0.0;
            const KDTreeNode *left = nullptr;
            const KDTreeNode *right = nullptr;
            const Point *point = nullptr;
        };
        const Point *points = nullptr;
        size_t num_points = 0;
        astd::pmr::monotonic_buffer_resource rsrc =
            astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource());
        Allocator<> alloc;
        const KDTreeNode *root = nullptr;
        const KDTreeNode *recursive_build(int depth, std::vector<int> &indices, std::pair<size_t, size_t> range) {
            if (range.second == range.first) {
                return nullptr;
            }
            if (range.second == range.first + 1) {
                auto node = alloc.new_object<KDTreeNode>();
                node->point = &points[range.first];
                return node;
            }
            int axis = depth % 3;
            std::sort(std::execution::par, indices.begin() + range.first, indices.begin() + range.second,
                      [=](int a, int b) { return points[a].p()[axis] < points[b].p()[axis]; });
            size_t mid = (range.first + range.second) / 2;
            auto node = alloc.new_object<KDTreeNode>();
            node->point = &points[mid];
            node->left = recursive_build(depth + 1, indices, std::make_pair(range.first, mid));
            node->right = recursive_build(depth + 1, indices, std::make_pair(mid + 1, range.second));
            return node;
        }
        void build() {

            std::vector<int> v(num_points);
            for (size_t i = 0; i < num_points; i++) {
                v[i] = i;
            }
            root = recursive_build(0, v, std::make_pair(size_t(0), v.size()));
        }

      public:
        template <class AllocatorT, class Container>
        void knn(AllocatorT alloc, const Point &query, size_t k, double radius, Container &container) const {
            auto cmp = [&](const Point &a, const Point &b) {
                auto d1 = a.p() - query.p();
                auto d2 = b.p() - query.p();
                return dot(d1, d1) < dot(d2, d2);
            };
            using Compare = decltype(cmp);
            auto cmp_node = [&](const KDTreeNode *a, const query *b) {
                auto d1 = a->point->p() - query.p();
                auto d2 = b->point->p() - query.p();
                return dot(d1, d1) < dot(d2, d2);
            };
            using CompareNode = decltype(cmp_node);
            container.resize(0);
            std::vector<const KDTreeNode *, AllocatorT> queue(alloc);
            queue.push_back(root);
            while (!queue.empty()) {
                auto *node = queue.front();
                std::pop_heap(queue.begin(), queue.end());
                auto p = node->point->p();
                if (l2norm(p - query) < radius * radius) {
                    if (container.size() < k) {
                        container.push_back(p);
                        std::push_heap(container.begin(), container.end());
                    } else {
                        std::pop_heap(container.begin(), container.end());
                        container.push_back(p);
                        std::push_heap(container.begin(), container.end());
                    }
                }
                auto max_dist = radius * radius;
                if (!container.empty()) {
                    max_dist = std::min<double>(max_dist, l2norm(container.front().p() - query));
                }
                if (node->left) {
                    auto d = (node->left->point->p()[axis] - query[axis]);
                    d *= d;
                    if (d < max_dist) {
                        queue.push_back(node->left);
                        std::push_heap(queue.begin(), queue.end());
                    }
                }
                if (node->right) {
                    auto d = (node->right->point->p()[axis] - query[axis]);
                    d *= d;
                    if (d < max_dist) {
                        queue.push_back(node->right);
                        std::push_heap(queue.begin(), queue.end());
                    }
                }
            }
        }
        KDTree(const Point *points, size_t num_points) : points(points), num_points(num_points), alloc(&rsrc) {
            build();
        }
    };
} // namespace akari::render