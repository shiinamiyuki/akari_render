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
#include <stack>
#include <spdlog/spdlog.h>
#include <akari/render.h>
#include <akari/thread.h>
namespace akari::render {
    struct AdamOptimizer {
        int t = 0;
        double m = 0;
        double v = 0;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double learning_rate = 0.01;
        double regularization = 0.01;
        double theta = 0.0;
        void step(double grad) {
            // AKR_ASSERT(!std::isnan(grad));
            if (std::isnan(grad))
                return;
            if (std::isinf(grad)) {
                grad = grad > 0.0 ? 100.0 : -100.0;
            }
            t++;
            double l = learning_rate * std::sqrt(1 - std::pow(beta1, t)) / std::pow(beta2, t);
            m = beta1 * m + (1.0 - beta1) * grad;
            v = beta2 * v + (1.0 - beta2) * grad * grad;
            theta = theta - l * m / (std::sqrt(v) + eps);
            theta = std::clamp<double>(theta, -20.0, 20.0);
            AKR_ASSERT(!std::isnan(m));
            AKR_ASSERT(!std::isnan(v));
            AKR_ASSERT(!std::isnan(theta));
        }
    };
    struct SDTreeDepositRecord {
        bool is_delta = false;
        Vec3 p;
        Vec3 wi;
        Vec3 n;
        Float bsdf = 0.0;
        Float bsdf_pdf = 0.0;
        Float radiance = 0.0;
        Float sample_pdf = 0.0;
    };

    class QTreeNode {
      public:
        std::array<AtomicFloat, 4> _sum;
        std::array<int, 4> _children = {-1, -1, -1, -1};

        QTreeNode() { setSum(0); }

        void copyFrom(const QTreeNode &node) {
            _children = (node._children);
            for (int i = 0; i < 4; i++) {
                _sum[i].set(float(node._sum[i]));
            }
        }

        QTreeNode(const QTreeNode &node) { copyFrom(node); }

        QTreeNode &operator=(const QTreeNode &node) {
            if (&node == this) {
                return *this;
            }
            copyFrom(node);
            return *this;
        }

        void setSum(Float val) {
            for (int i = 0; i < 4; i++) {
                _sum[i].set(val);
            }
        }

        QTreeNode *child(int i, std::vector<QTreeNode> &nodes) const {
            return _children[i] > 0 ? &nodes[_children[i]] : nullptr;
        }

        static int childIndex(const vec2 &p) {
            int x, y;
            if (p.x < 0.5f) {
                x = 0;
            } else {
                x = 1;
            }
            if (p.y < 0.5f) {
                y = 0;
            } else {
                y = 1;
            }
            return x + 2 * y;
        }

        [[nodiscard]] bool isLeaf(int i) const { return _children[i] <= 0; }

        [[nodiscard]] static vec2 offset(size_t i) { return vec2(i & 1u, i >> 1u) * 0.5f; }

        [[nodiscard]] Float sum() const {
            Float v = 0;
            for (auto &i : _sum) {
                v += (float)i;
            }
            return v;
        }

        float eval(const vec2 &p, std::vector<QTreeNode> &nodes) const {
            auto idx = childIndex(p);
            if (child(idx, nodes)) {
                return 4.0f * child(idx, nodes)->eval((p - offset(idx)) * 2.0f, nodes);
            } else {
                AKR_CHECK(_sum[idx].value() >= 0.0f);
                return 4.0f * float(_sum[idx]);
            }
        }

        float pdf(const vec2 &p, std::vector<QTreeNode> &nodes) const {
            auto idx = childIndex(p);
            //            if (!(_sum[idx].value() > 0)) {
            //                return 0.0f;
            //            }
            auto s = sum();
            AKR_CHECK(s >= 0);
            AKR_CHECK(!std::isnan(s));
            auto factor = s <= 0.0f ? 0.25f : _sum[idx].value() / s;
            if (factor < 0 || std::isnan(factor)) {
                spdlog::info("{} {} {}\n", factor, _sum[idx].value(), s);
            }
            AKR_ASSERT(!std::isnan(factor));
            AKR_ASSERT(factor >= 0);
            if (child(idx, nodes)) {
                return 4.0f * factor * child(idx, nodes)->pdf((p - offset(idx)) * 2.0f, nodes);
            } else {
                return 4.0f * factor;
            }
        }

        vec2 sample(vec2 u, vec2 u2, std::vector<QTreeNode> &nodes) const {
            std::array<float, 4> m = {float(_sum[0]), float(_sum[1]), float(_sum[2]), float(_sum[3])};
            auto left = m[0] + m[2];
            auto right = m[1] + m[3];
            auto total = left + right;
            // log::log("total: {}\n", total);
            //            AKR_CHECK(total > 0);
            if (total == 0) {
                total = 1;
                m[0] = m[1] = m[2] = m[3] = 0.25;
                left = m[0] + m[2];
                right = m[1] + m[3];
            }

            int x, y;
            if (u[0] < left / total) {
                x = 0;
                u[0] /= left / total;
            } else {
                x = 1;
                u[0] = (u[0] - left / total) / (right / total);
            }
            auto up = m[x];
            auto down = m[2 + x];
            total = up + down;
            if (u[1] < up / total) {
                y = 0;
                u[1] /= up / total;
            } else {
                y = 1;
                u[1] = (u[1] - up / total) / (down / total);
            }
            int child = x + 2 * y;
            vec2 sampled;
            if (this->child(child, nodes)) {
                sampled = this->child(child, nodes)->sample(u, u2, nodes);
            } else {
                sampled = u2;
            }
            return vec2(x, y) * 0.5f + sampled * 0.5f;
        }

        void deposit(const vec2 &p, Float e, std::vector<QTreeNode> &nodes) {
            int idx = childIndex(p);
            _sum[idx].add(e);
            auto c = child(idx, nodes);
            AKR_CHECK(e >= 0);
            AKR_CHECK(_sum[idx].value() >= 0);
            if (c) {
                c->deposit((p - offset(idx)) * 2.0f, e, nodes);
            }
        }
    };

    static vec3 canonicalToDir(const vec2 &p) {
        const Float cosTheta = 2 * p.y - 1;
        const Float phi = 2 * Pi * p.x;

        const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        sinPhi = sin(phi);
        cosPhi = cos(phi);

        return vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
    }

    static vec2 dirToCanonical(const vec3 &d) {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return vec2(0, 0);
        }

        const Float cosTheta = std::min(std::max(d.y, -1.0f), 1.0f);
        Float phi = std::atan2(d.z, d.x);
        while (phi < 0)
            phi += 2.0 * Pi;

        return vec2(phi / (2 * Pi), (cosTheta + 1) / 2);
    }

    class DTree {
      public:
        AtomicFloat sum;
        AtomicFloat weight;
        std::vector<QTreeNode> nodes;

        DTree() : sum(0), weight(0) {
            nodes.emplace_back();
            _build();
        }

        DTree(const DTree &other) {
            sum.set(other.sum.value());
            weight.set(other.weight.value());
            nodes = (other.nodes);
        }

        DTree &operator=(const DTree &other) {
            if (&other == this) {
                return *this;
            }
            sum.set(other.sum.value());
            weight.set(other.weight.value());
            nodes = (other.nodes);
            return *this;
        }

        vec2 sample(const vec2 &u, const vec2 &u2) { return nodes[0].sample(u, u2, nodes); }

        Float pdf(const vec2 &p) { return nodes[0].pdf(p, nodes); }

        Float eval(const vec2 &u) {
            return nodes[0].eval(u, nodes); /// weight.value();
        }

        void _build() {
            auto updateSum = [this](QTreeNode &node, auto &&update) -> double {
                for (int i = 0; i < 4; i++) {
                    if (!node.isLeaf(i)) {
                        auto c = node.child(i, nodes);
                        update(*c, update);
                        node._sum[i].set(c->sum());
                    }
                }
                //    log::log("sum: {}\n",node.sum());
                return node.sum();
            };
            sum.set(updateSum(nodes.front(), updateSum));
        }

        void reset() {
            for (auto &i : nodes) {
                i.setSum(0);
            }
            sum.set(0);
        }

        void refine(const DTree &prev, Float threshold) {
            struct StackNode {
                size_t node = (size_t)-1, otherNode = (size_t)-1;
                const DTree *tree = nullptr;
                int depth;
            };
            std::stack<StackNode> stack;
            nodes.clear();
            nodes.emplace_back();
            stack.push({0, 0, &prev, 1});
            sum.set(0.0f);
            auto total = prev.sum.value();
            //            log::log("{} {}\n", total, threshold);
            while (!stack.empty()) {
                auto node = stack.top();
                stack.pop();

                // log::log("other index: {}, sum: {}\n",node.otherNode, otherNode.sum());
                for (int i = 0; i < 4; ++i) {
                    AKR_CHECK(node.otherNode >= 0);
                    auto &otherNode = node.tree->nodes.at(node.otherNode);
                    auto fraction = total == 0.0f ? std::pow(0.25, node.depth) : otherNode._sum[i].value() / total;
                    // log::log("{} {}\n", otherNode._sum[i].value(), fraction);
                    if (fraction >= threshold && node.depth < 10) {
                        if (otherNode.isLeaf(i)) {
                            stack.push({nodes.size(), nodes.size(), this, node.depth + 1});
                        } else {
                            AKR_CHECK(otherNode._children[i] > 0);
                            AKR_CHECK((size_t)otherNode._children[i] != node.otherNode);
                            AKR_CHECK(node.tree == &prev);
                            stack.push({nodes.size(), (size_t)otherNode._children[i], &prev, node.depth + 1});
                        }
                        nodes[node.node]._children[i] = nodes.size();
                        auto val = otherNode._sum[i].value() / 4.0f;
                        nodes.emplace_back();
                        nodes.back().setSum(val);
                    }
                }
            }
            //           log::log("QTreeNodes: {}\n", nodes.size());
            weight.add(1);
            reset();
        }

        void deposit(const vec2 &p, Float e) {
            if (e <= 0)
                return;
            sum.add(e);
            nodes[0].deposit(p, e, nodes);
        }
    };

    class DTreeWrapper {
      public:
        bool valid = true;
        DTree building, sampling;
        AdamOptimizer opt;
        SpinLock lock;
        // std::mutex lock;
        DTreeWrapper() = default;
        DTreeWrapper(const DTreeWrapper &rhs) : building(rhs.building), sampling(rhs.sampling), opt(rhs.opt) {}
        DTreeWrapper &operator=(const DTreeWrapper &rhs) {
            building = rhs.building;
            sampling = rhs.sampling;
            opt = rhs.opt;
            return *this;
        }
        double selection_prob() const { return 1.0 / (1.0 + std::exp(-opt.theta)); }
        //        DTreeWrapper(){
        //            sampling.nodes[0].setSum(0.25);
        //        }

        vec3 sample(const vec2 &u, const vec2 &u2) { return canonicalToDir(sampling.sample(u, u2)); }

        Float pdf(const vec3 &w) { return sampling.pdf(dirToCanonical(w)) * Inv4Pi; }

        Float eval(const vec3 &w) { return sampling.eval(dirToCanonical(w)); }

        void deposit(const SDTreeDepositRecord &record) {
            AKR_CHECK(!building.nodes.empty());
            auto p = dirToCanonical(record.wi);
            building.deposit(p, record.radiance);
            return;
            // https://tom94.net/data/courses/vorba19guiding/vorba19guiding.pdf
            const auto product_estimate = record.radiance * record.bsdf * abs(dot(record.n, record.wi));
            const auto learned_pdf = record.is_delta ? 0.0 : pdf(record.wi);

            {
                std::lock_guard<SpinLock> guard(lock);
                const auto a = selection_prob();
                const auto combined_pdf = a * record.bsdf_pdf + (1.0 - a) * learned_pdf;
                const auto grad_a =
                    product_estimate * (record.bsdf_pdf - learned_pdf) / (record.sample_pdf * combined_pdf);
                const auto grad_theta = grad_a * a * (1.0 - a);
                const auto reg_grad = opt.regularization * opt.theta;
                opt.step(grad_theta + reg_grad);
                // if(product_estimate > 0.0)
                //  spdlog::info("grad = {}, theta = {}", grad_theta, opt.theta);
            }
        }

        void refine() {
            // spdlog::info("a = {}", selection_prob());
            AKR_CHECK(building.sum.value() >= 0.0f);
            //            building._build();
            sampling = building;
            sampling._build();
            AKR_CHECK(sampling.sum.value() >= 0.0f);
            building.refine(sampling, 0.01);
        }
    };

    class STreeNode {
      public:
        DTreeWrapper dTree;
        std::atomic<int> nSample = 0;
        std::array<int, 2> _children = {-1, -1};
        int axis = 0;
        bool _isLeaf = true;

        STreeNode() = default;

        STreeNode(const STreeNode &other)
            : dTree(other.dTree), nSample((int)other.nSample), _children(other._children), axis(other.axis),
              _isLeaf(other._isLeaf) {}

        [[nodiscard]] bool isLeaf() const { return _isLeaf; }

        vec3 sample(vec3 p, const vec2 &u, const vec2 &u2, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                return dTree.sample(u, u2);
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    return nodes[_children[0]].sample(p, u, u2, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    return nodes[_children[1]].sample(p, u, u2, nodes);
                }
            }
        }

        Float pdf(vec3 p, const vec3 &w, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                return dTree.pdf(w);
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    return nodes.at(_children[0]).pdf(p, w, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    return nodes.at(_children[1]).pdf(p, w, nodes);
                }
            }
        }

        std::pair<DTreeWrapper *, int> getDTree(vec3 p, std::vector<STreeNode> &nodes) {
            //            AKR_CHECK(0.0f - 1e-6f <= p[axis] && p[axis] <= 1.0f + 1e-6f);
            //            log::log("{} {}\n",axis,p[axis]);
            if (isLeaf()) {
                return {&dTree, 0};
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    auto [tree, depth] = nodes.at(_children[0]).getDTree(p, nodes);
                    return {tree, depth + 1};
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    auto [tree, depth] = nodes.at(_children[1]).getDTree(p, nodes);
                    return {tree, depth + 1};
                }
            }
        }

        Float eval(vec3 p, const vec3 &w, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                return dTree.eval(w);
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    return nodes.at(_children[0]).eval(p, w, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    return nodes.at(_children[1]).eval(p, w, nodes);
                }
            }
        }

        void deposit(SDTreeDepositRecord record, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                nSample++;
                dTree.deposit(record);

            } else {
                if (record.p[axis] < 0.5f) {
                    record.p[axis] *= 2.0f;
                    nodes.at(_children[0]).deposit(record, nodes);
                } else {
                    record.p[axis] = (record.p[axis] - 0.5f) * 2.0f;
                    nodes.at(_children[1]).deposit(record, nodes);
                }
            }
        }
    };

    class STree {
      public:
        std::vector<STreeNode> nodes;

        explicit STree(const Bounds3f &box) : nodes(1) {
            auto sz = hmax(box.size()) * 0.5f;
            auto centroid = box.centroid();
            this->box = Bounds3f{centroid - vec3(sz), centroid + vec3(sz)};
        }

        Bounds3f box;

        vec3 sample(const vec3 &p, const vec2 &u, const vec2 &u2) {
            return nodes.at(0).sample(box.offset(p), u, u2, nodes);
        }

        Float pdf(const vec3 &p, const vec3 &w) { return nodes.at(0).pdf(box.offset(p), w, nodes); }

        auto dTree(const vec3 &p) { return nodes[0].getDTree(box.offset(p), nodes); }

        Float eval(const vec3 &p, const vec3 &w) { return nodes.at(0).eval(box.offset(p), w, nodes); }

        void deposit(SDTreeDepositRecord record) {
            auto &irradiance = record.radiance;
            AKR_ASSERT(irradiance >= 0 && !std::isnan(irradiance) && !std::isinf(irradiance));
            if (irradiance >= 0 && !std::isnan(irradiance)) {
                record.p = box.offset(record.p);
                nodes.at(0).deposit(record, nodes);
            }
        }

        void refine(int idx, size_t maxSample, int depth) {
            if (nodes[idx].isLeaf() && (size_t)nodes[idx].nSample > maxSample) {
                for (int i = 0; i < 2; i++) {
                    nodes[idx]._children[i] = nodes.size();
                    nodes.emplace_back();
                    auto &node = nodes.back();
                    node.nSample = nodes[idx].nSample / 2;
                    node.axis = (nodes[idx].axis + 1) % 3;
                    node.dTree = nodes[idx].dTree;
                    AKR_CHECK(node.isLeaf());
                }
                nodes[idx]._isLeaf = false;
                nodes[idx].dTree = DTreeWrapper();
                nodes[idx].dTree.valid = false;
            }
            if (!nodes[idx].isLeaf()) {
                for (int i = 0; i < 2; i++) {
                    refine(nodes[idx]._children[i], maxSample, depth + 1);
                }
                AKR_CHECK(nodes[idx]._children[0] > 0 && nodes[idx]._children[1] > 0);
            }
        }

        void refine(size_t maxSample) {
            AKR_CHECK(maxSample > 0);
            for (auto &i : nodes) {
                if (i.isLeaf()) {
                    i.dTree.refine();
                }
            }
            refine(0, maxSample, 0);
            for (auto &i : nodes) {
                i.nSample = 0;
            }
        }
    };

    struct PPGConfig {
        Sampler sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp = 16;
    };
    std::shared_ptr<STree> render_ppg(std::vector<std::pair<Array2D<Spectrum>, Spectrum>> &all_samples,
                                      PPGConfig config, const Scene &scene);
    Image render_ppg(PPGConfig config, const Scene &scene);

    std::shared_ptr<STree> bake_sdtree(PPGConfig config, const Scene &scene);

} // namespace akari::render