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
#include <akari/common/bufferview.h>
#include <akari/core/bvhbuilder.h>

namespace akari {
    template <class Float, class UserData, class Hit, class Intersector, class ShapeHandleConstructor>
    struct BVHIntersector {
        AKR_IMPORT_CORE_TYPES()
        using Builder = BVHBuilder<Float, UserData, ShapeHandleConstructor>;
        using BVHNode = typename Builder::BVHNode;
        using Index = typename Builder::Index;
        static_assert(sizeof(Index) == sizeof(int));

        auto get(const Index &handle) { return _ctor(user_data, handle.idx); }

        Bounds3f boundBox;
        const UserData *user_data;
        Intersector _intersector;
        ShapeHandleConstructor _ctor;
        BufferView<const Index> primitives;
        BufferView<const BVHNode> nodes;

        AKR_XPU ShapeHandleConstructor(const Builder &builder, Intersector intersector = Intersector(),
                                       ShapeHandleConstructor ctor = ShapeHandleConstructor())
            : user_data(builder.user_data), _intersector(std::move(intersector)), _ctor(std::move(ctor)),
              primitives(builder.primitives), nodes(builder.nodes) {}

        AKR_XPU static Float intersectAABB(const Bounds3f &box, const Ray3f &ray, const Vector3f &invd) {
            Vector3f t0 = (box.p_min - ray.o) * invd;
            Vector3f t1 = (box.p_max - ray.o) * invd;
            Vector3f tMin = min(t0, t1), tMax = max(t0, t1);
            if (hmax(tMin) <= hmin(tMax)) {
                auto t = std::max(ray.t_min, hmax(tMin));
                if (t >= ray.t_max) {
                    return -1;
                }
                return t;
            }
            return -1;
        }

        AKR_XPU bool intersect(const Ray3f &ray, Hit &isct) const {
            bool hit = false;
            auto invd = Vector3f(1) / ray.d;
            constexpr int maxDepth = 64;
            const BVHNode *stack[maxDepth];
            int sp = 0;
            stack[sp++] = &nodes[0];
            while (sp > 0) {

                auto p = stack[--sp];
                //                PrintBox(p->box, "p->box");
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
        [[nodiscard]] AKR_XPU bool occlude(const Ray3f &ray) const {
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
} // namespace akari