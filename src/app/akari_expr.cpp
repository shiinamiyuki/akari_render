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

#include <akari/core/logger.h>
#include <akari/core/math.h>
#include <variant>
#include <magic_enum.hpp>
#include <unordered_set>
#include <iostream>
namespace akari::compute {
    namespace ir {
        /*
        akari::compute IR:

        call ::= term term*
        term ::= constant | function | primitive | call

        */
        class NodeVisitor;
        class CallNode;
        class ConstantNode;
        class FunctionNode;
        class PrimitiveNode;
        class VarNode;
        class NodeVisitor {
          public:
            virtual void visit(CallNode &) = 0;
            virtual void visit(ConstantNode &) = 0;
            virtual void visit(FunctionNode &) = 0;
            virtual void visit(PrimitiveNode &) = 0;
            virtual void visit(VarNode &) = 0;
        };
        class Node : public std::enable_shared_from_this<Node> {
          public:
            template <typename T> bool isa() const { return typeid(*this) == typeid(T); }
            template <typename T> std::shared_ptr<const T> cast() const {
                return std::dynamic_pointer_cast<const T>(shared_from_this());
            }
            template <typename T> std::shared_ptr<T> cast() { return std::dynamic_pointer_cast<T>(shared_from_this()); }
            virtual void accept(NodeVisitor &) = 0;
            virtual std::string type_name() const = 0;
            virtual std::string dump(size_t level = 0) const = 0;
            static std::string indent(size_t level) { return std::string(level, ' '); }
            virtual std::string str_this() const = 0;
            virtual ~Node() = default;
        };
#define AKR_DECL_NODE(Type)                                                                                            \
    std::string type_name() const { return #Type; }                                                                    \
    void accept(NodeVisitor &vis) { vis.visit(*this); }
        class Expr : public Node {};
        class VarNode : public Expr {
            size_t id;
            std::string _name;

          public:
            VarNode(size_t id, std::string name) : id(id), _name(std::move(name)) {}
            AKR_DECL_NODE(VarNode)
            std::string dump(size_t level) const {
                return indent(level) + fmt::format("{} %{} [{}] '{}'", (void *)this, id, type_name(), _name);
            }
            std::string str_this() const override { return fmt::format("[{}] '{}'", type_name(), _name); }
        };

        enum class Primitive {
            EAdd,
            ESub,
            EMul,
            EDiv,
            EBroadcast,
            EVAdd, // a: vec, b: vec, width: int
            EVSub,
            EVMul,
            EVDiv,
            ESQMatmul,
            ESQMatVmul, // a: mat, b: vec, width: int
            EDot,
            ECross,
        };

        class PrimitiveNode : public Expr {
            Primitive _prim;

          public:
            PrimitiveNode(Primitive p) : _prim(p) {}
            Primitive primitive() const { return _prim; }
            AKR_DECL_NODE(PrimitiveNode)
            std::string dump(size_t level) const {
                return indent(level) + fmt::format("{} Primitive::{}", (void *)this, magic_enum::enum_name(_prim));
            }
            std::string str_this() const override {
                return fmt::format("[{}] '{}'", type_name(), magic_enum::enum_name(_prim));
            }
        }; // namespace akari::compute

        // Anonymous Function Node
        class FunctionNode : public Expr {
            std::vector<std::shared_ptr<VarNode>> _parameters;
            std::shared_ptr<Expr> _body;

          public:
            FunctionNode(std::vector<std::shared_ptr<VarNode>> parameters, std::shared_ptr<Expr> body)
                : _parameters(std::move(parameters)), _body(std::move(body)) {}
            const std::vector<std::shared_ptr<VarNode>> &parameters() const { return _parameters; }
            auto body() const { return _body; }
            AKR_DECL_NODE(FunctionNode)
            std::string dump(size_t level) const {
                auto s = indent(level) + fmt::format("{} [{}] (", (void *)this, type_name());
                for (size_t i = 0; i < _parameters.size(); i++) {
                    s.append(fmt::format("[{}] ", _parameters[i]->dump(0)));
                }
                s.append(")\n");
                s.append(_body->dump(level + 1));
                return s;
            }
            std::string str_this() const override { return fmt::format("[{}] '{}'", type_name(), (void *)this); }
        };

        class CallNode : public Expr {
            std::shared_ptr<Node> _op;
            std::vector<std::shared_ptr<Expr>> _args;

          public:
            const std::shared_ptr<Node> &op() const { return _op; }
            const std::vector<std::shared_ptr<Expr>> &args() const { return _args; }
            AKR_DECL_NODE(CallNode)
            void set_op(std::shared_ptr<Node> op) { _op = std::move(op); }
            void add_arg(std::shared_ptr<Expr> a) { _args.emplace_back(std::move(a)); }
            std::string dump(size_t level) const {

                auto to_string = [](const std::shared_ptr<Node> &node) {
                    std::string s;
                    if (node->isa<FunctionNode>() || node->isa<CallNode>()) {
                        s = fmt::format("[{} {}]", (void *)node.get(), node->type_name());
                    } else {
                        s = fmt::format("[{}]", node->dump(0));
                    }
                    return s;
                };
                auto s = indent(level) + fmt::format("{} [{}] {} (", (void *)this, type_name(), to_string(_op));
                for (size_t i = 0; i < _args.size(); i++) {
                    s.append(to_string(_args[i])).append(" ");
                }
                s.append(")\n");
                return s;
            }
            std::string str_this() const override { return fmt::format("[{}] '{}'", type_name(), (void *)this); }
        };

        template <typename T> struct is_value : std::false_type {};
        template <> struct is_value<int> : std::true_type {};
        template <> struct is_value<float> : std::true_type {};
        template <> struct is_value<vec2> : std::true_type {};
        template <> struct is_value<vec3> : std::true_type {};
        template <typename T> std::shared_ptr<Expr> to_expr(T &&v) {
            if constexpr (is_value<T>::value) {
                return std::make_shared<ConstantNode>(v);
            } else {
                return v;
            }
        }
        template <typename... Ts> std::shared_ptr<CallNode> call(Primitive op, Ts &&... _args) {
            std::vector<std::shared_ptr<Expr>> args = {to_expr(std::forward<Ts>(_args))...};
            auto cnode = std::make_shared<CallNode>();
            cnode->set_op(std::make_shared<PrimitiveNode>(op));
            for (auto &a : args) {
                cnode->add_arg(a);
            }
            return cnode;
        }
        class ConstantNode : public Expr {
          public:
            AKR_DECL_NODE(ConstantNode)
            using Value = std::variant<int, float, vec2, vec3, mat3, mat4>;
            template <typename T, typename = std::enable_if_t<is_value<T>::value>>
            ConstantNode(const T &v) : _value(v) {}
            const Value &value() const { return _value; }
            std::string v_str() const {
                return std::visit(
                    [](auto &&arg) -> std::string {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                            return fmt::format("{}", arg);
                        }
                        return "unknown";
                    },
                    _value);
            }
            std::string dump(size_t level) const { return indent(level) + v_str(); };
            std::string str_this() const override { return fmt::format("[{}] '{}'", type_name(), v_str()); }

          private:
            Value _value;
        };

        class DebugDumpGraph {
            std::unordered_set<std::shared_ptr<Node>> visited;
            std::vector<std::string> subgraphs;

          public:
            void dump_subgraph(const std::shared_ptr<FunctionNode> &func) {
                auto s = fmt::format("subgraph {} {{ label=\"{}\";\n", id(func), func->str_this());
                for (auto &p : func->parameters()) {
                    declare(p);
                    s.append(fmt::format("{} -> {};\n", dump(p), id(func)));
                }
                s.append(dump(func->body()));
                s.append("}");
                subgraphs.emplace_back(s);
            }
            std::string id(const std::shared_ptr<Node> &expr) { return fmt::format("{}", (void *)expr.get()); }
            std::string declare(const std::shared_ptr<Node> &expr) {
                if (visited.count(expr) > 0) {
                    return "";
                }
                auto label = expr->str_this();
                std::string out = fmt::format("{} [label=\"{}\"]; \n", id(expr), label);
                if (expr->isa<CallNode>()) {
                    auto cnode = expr->cast<CallNode>();
                    declare(cnode->op());
                    for (auto &a : cnode->args()) {
                        declare(a);
                    }

                } else if (expr->isa<FunctionNode>()) {
                    auto func = expr->cast<FunctionNode>();
                    dump_subgraph(func);
                    return out + id(expr);
                }
                return out;
            }
            std::string dump(const std::shared_ptr<Node> &expr) {
                if (visited.count(expr) > 0) {
                    return fmt::format("{}", id(expr));
                }
                visited.insert(expr);
                std::string out = "";
                if (expr->isa<CallNode>()) {
                    auto cnode = expr->cast<CallNode>();
                    declare(cnode->op());
                    for (auto &a : cnode->args()) {
                        declare(a);
                    }

                } else if (expr->isa<FunctionNode>()) {
                    auto func = expr->cast<FunctionNode>();
                    dump_subgraph(func);
                    return out + id(expr);
                }
            }
        }
    } // namespace ir
      /*
      akari::compute C++ DSL
      Var x;
  
      Function add([](Var a, Var b){
        return a + b;
      });
      */
    namespace lang {
        static size_t new_var_id() {
            static size_t id = 0;
            return id++;
        }

        // template<typename T=void>
        namespace detail {
            struct named {};

            struct any {};
        } // namespace detail

        struct Function;
        struct Type {
            template <typename T> struct Basic {};
        };
        struct Var {
            template <typename T, typename = std::enable_if_t<ir::is_value<T>::value>>
            Var(const T &v, const std::string &name = "") : name(name) {
                expr = std::make_shared<ir::ConstantNode>(v);
            }
            inline Var(const Function &f);
            Var(detail::named, const std::string &name = "") {
                expr = std::make_shared<ir::VarNode>(new_var_id(), name);
            }
            Var(std::shared_ptr<ir::Expr> expr) : expr(std::move(expr)) {}
            const std::shared_ptr<ir::Expr> &get_expr() const { return expr; }
            friend Var operator+(const Var &lhs, const Var &rhs) {
                return Var(ir::call(ir::Primitive::EAdd, lhs.expr, rhs.expr));
            }
            friend Var operator-(const Var &lhs, const Var &rhs) {
                return Var(ir::call(ir::Primitive::ESub, lhs.expr, rhs.expr));
            }
            friend Var operator*(const Var &lhs, const Var &rhs) {
                return Var(ir::call(ir::Primitive::EMul, lhs.expr, rhs.expr));
            }
            friend Var operator/(const Var &lhs, const Var &rhs) {
                return Var(ir::call(ir::Primitive::EDiv, lhs.expr, rhs.expr));
            }
            Var operator+=(const Var &rhs) {
                *this = *this + rhs;
                return *this;
            }
            Var operator-=(const Var &rhs) {
                *this = *this - rhs;
                return *this;
            }
            Var operator*=(const Var &rhs) {
                *this = *this * rhs;
                return *this;
            }
            Var operator/=(const Var &rhs) {
                *this = *this / rhs;
                return *this;
            }

          private:
            std::string name;
            std::shared_ptr<ir::Expr> expr;
        };

        struct Function {
            friend struct Var;
            template <typename T, T... ints, typename F>
            Var invoke(std::index_sequence<ints...>, const F &f, const std::vector<Var> &parameters) {
                return f(parameters.at(ints)...);
            }
            template <typename... Ts>
            // typename = std::enable_if_t<std::conjunction_v<std::is_same_v<std::decay_t<Ts>, Var>>>>
            Function(std::function<Var(Ts...)> f) {
                std::vector<Var> parameters;
                for (size_t i = 0; i < sizeof...(Ts); i++) {
                    parameters.emplace_back(detail::named{}, fmt::format("param{}", i));
                }

                Var body = invoke(std::index_sequence_for<Ts...>{}, f, parameters);
                std::vector<std::shared_ptr<ir::VarNode>> true_parameters;
                std::transform(parameters.begin(), parameters.end(), std::back_inserter(true_parameters),
                               [](const Var &var) {
                                   AKR_ASSERT(var.get_expr() && var.get_expr()->isa<ir::VarNode>());
                                   return var.get_expr()->cast<ir::VarNode>();
                               });
                func_node = std::make_shared<ir::FunctionNode>(true_parameters, body.get_expr());
            }
            template <typename F> Function(F &&f) : Function(std::function(f)) {}

          private:
            std::shared_ptr<ir::FunctionNode> func_node;
        };
        inline Var::Var(const Function &f) : expr(f.func_node) {}
    } // namespace lang
} // namespace akari::compute

#if 0

void test(){
  struct Point : Struct {
    Var<float> x("x"), y("y"), z("z");
  };
  Function same_hemisphere = [](const Var<vec3>& u, const Var<vec3> &v){
    return select(dot(u, v) > 0, true, false);
  };
  Var x(2.0f);
  for(int i =0;i<10;i++){
    x *= 2.0f;
  }
  if_(x > 10).then([=]{
    x += 20;
  }).else_([&]{
    x += 2;
  });
  while_(x > 10).do_([=]{
    
  });
}

#endif
int main() {
    using namespace akari::compute;
    using namespace lang;
    // Var x(1);
    // Var y = x + 1;
    // y += 2;
    Function add([](Var x, Var y) -> Var { return x + y; });
    Var f = add;
    std::cout << f.get_expr()->dump() << std::endl;
}
