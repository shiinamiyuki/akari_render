#include <type_traits>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <unordered_set>
#include <iostream>
#include <stack>
namespace akari::compute {
    struct Base : std::enable_shared_from_this<Base> {
        template <typename T> bool isa() const { return typeid(*this) == typeid(T); }
        template <typename T> std::shared_ptr<T> cast() { return std::dynamic_pointer_cast<T>(shared_from_this()); }
        template <typename T> std::shared_ptr<const T> cast() const {
            return std::dynamic_pointer_cast<const T>(shared_from_this());
        }
        virtual ~Base() = default;
    };
    struct Type : Base {};
    using TypePtr = std::shared_ptr<Type>;
    struct Node : Base {};
    struct Expr : Node {
        TypePtr type;
        Expr(TypePtr type) : type(type) {}
    };
    using NodePtr = std::shared_ptr<Node>;
    using ExprPtr = std::shared_ptr<Expr>;
    struct VarStore : Node {
        NodePtr dst;
        ExprPtr src;
        VarStore(NodePtr dst, ExprPtr src) : dst(dst), src(src) {}
    };
    struct ConstantNode : Expr {
        float val;
        ConstantNode(TypePtr type, float v) : Expr(type), val(v) {}
    };

    struct BinOpNode : Expr {
        enum Op { add, sub, mul, div };
        Op op;
        NodePtr lhs = nullptr;
        NodePtr rhs = nullptr;
        BinOpNode(TypePtr type, Op op, NodePtr lhs, NodePtr rhs) : Expr(type), op(op), lhs(lhs), rhs(rhs) {}
    };
    struct Block;
    using BlockPtr = std::shared_ptr<Block>;

    struct Kernel {
        std::vector<Block *> blocks;
        BlockPtr entry;
    };

    struct Float32Type : Type {};

    struct Block : Base {
        BlockPtr next;
        Block(Kernel *kernel) { kernel->blocks.emplace_back(this); }
    };
    struct BasicBlock : Block {
        std::vector<NodePtr> body;

        using Block::Block;
    };
    using BB = std::shared_ptr<BasicBlock>;
    struct IfBlock : Block {
        NodePtr cond;
        BlockPtr if_true;
        BlockPtr if_false;
        using Block::Block;
    };
    struct KernelNode : Node {
        BlockPtr body;
    };

    struct Context {
        Node *root;
    };

    struct NodeInserter {
        BB bb;
        BB cur_insertion_block() { return bb; }
        NodePtr insert(NodePtr p) const {
            // std::cout << "inserted " << typeid(*p).name() << std::endl;
            bb->body.emplace_back(p);
            return p;
        }
        void reset(BB bb) { this->bb = bb; }
    };
    NodeInserter _inserter;
    NodeInserter &get_inserter() { return _inserter; }
    Kernel *cur_kernel = nullptr;

    struct BuildIf {
        std::shared_ptr<IfBlock> if_block, parent_if;
        bool completed = false;
        struct push_t {};
        static constexpr uint32_t has_if = 1u;
        static constexpr uint32_t has_else = 1u << 1;
        uint32_t state = 0;
        std::stack<std::shared_ptr<IfBlock>> st;
        std::stack<uint32_t> states;
        template <typename C> BuildIf(C &&c) {
            auto bb = get_inserter().cur_insertion_block();
            if_block = std::make_shared<IfBlock>(cur_kernel);
            if_block->cond = c().node;
            if_block->if_false = std::make_shared<BasicBlock>(cur_kernel);
            bb->next = if_block;
            parent_if = if_block;
        }

        template <class F> BuildIf &operator<<(F &&f) {
            if (completed) {
                throw std::runtime_error("if is completed");
            }

            if (state & has_if) {
                throw std::runtime_error("multiple then block!");
            }
            auto bb = std::make_shared<BasicBlock>(cur_kernel);
            get_inserter().reset(bb);
            f();
            if_block->if_true = bb;
            state |= has_if;
            get_inserter().reset(if_block->if_false->cast<BasicBlock>());

            return *this;
        }
        template <class F> BuildIf &operator>>(F &&f) {
            if (completed) {
                throw std::runtime_error("if is completed");
            }
            if (!(state & has_if)) {
                throw std::runtime_error("no if block!");
            }
            if (state & has_else) {
                throw std::runtime_error("multiple else block!");
            }
            if constexpr (std::is_same_v<std::decay_t<F>, NodePtr>) {
                // else if
                get_inserter().reset(if_block->if_false->cast<BasicBlock>());
                auto elseif = std::make_shared<IfBlock>(cur_kernel);
                elseif->cond = f;
                elseif->if_false = std::make_shared<BasicBlock>(cur_kernel);
                if_block->if_false = elseif;
                st.push(if_block);
                if_block = elseif;
                states.push(state);
                state = 0;
            } else {

                f();
                // if_block->if_false = bb;

                if (!st.empty()) {

                    if_block = st.top();
                    state = states.top();
                    st.pop();
                    states.pop();
                } else {
                    completed = true;
                }
            }
            return *this;
        }
        ~BuildIf() {

            auto bb = std::make_shared<BasicBlock>(cur_kernel);
            get_inserter().reset(bb);
            parent_if->next = bb;
        }
    };
#define If(cond)   BuildIf([&]() { return cond; }) << [&]
#define Elif(cond) >> ([&]() -> Var<float> { return cond; })().node << [&]
#define Else       >> [&]

    template <typename Float, typename Derived> struct VarBaseFloat {
        static TypePtr get_type() {
            static auto type = std::make_shared<Float32Type>();
            return type;
        }
        NodePtr node;
        VarBaseFloat(NodePtr node) : node(std::move(node)) {}
        VarBaseFloat(Float v = 0) { from(v); }
        void from(Float v) {
            node = std::make_shared<ConstantNode>(get_type(), v);
            get_inserter().insert(node);
        }
        Derived operator+(const VarBaseFloat &rhs) const {
            return Derived(
                get_inserter().insert(std::make_shared<BinOpNode>(get_type(), BinOpNode::add, node, rhs.node)));
        }
        Derived operator-(const VarBaseFloat &rhs) const {
            return Derived(
                get_inserter().insert(std::make_shared<BinOpNode>(get_type(), BinOpNode::sub, node, rhs.node)));
        }
        Derived operator*(const VarBaseFloat &rhs) const {
            return Derived(
                get_inserter().insert(std::make_shared<BinOpNode>(get_type(), BinOpNode::mul, node, rhs.node)));
        }
        Derived operator/(const VarBaseFloat &rhs) const {
            return Derived(
                get_inserter().insert(std::make_shared<BinOpNode>(get_type(), BinOpNode::div, node, rhs.node)));
        }
        Derived &operator=(Float v) {
            from(v);
            return static_cast<Derived &>(*this);
        }
        Derived &operator=(const VarBaseFloat &rhs) {
            if (&rhs == this) {
                return static_cast<Derived &>(*this);
            }
            if (!node)
                from(Float(0.0));
            get_inserter().insert(std::make_shared<VarStore>(node, rhs.node->template cast<Expr>()));
            return static_cast<Derived &>(*this);
        }
    };

    template <typename T> struct Var {};

    template <> struct Var<float> : VarBaseFloat<float, Var<float>> {
        using VarBaseFloat<float, Var<float>>::VarBaseFloat;
    };

    template <typename F> Kernel compile_kernel(F &&f) {
        Kernel kernel;
        auto &inserter = get_inserter();
        auto bb = std::make_shared<BasicBlock>(&kernel);
        kernel.entry = bb;
        cur_kernel = &kernel;
        inserter.reset(bb);
        f();
        return kernel;
    }
    class CodeGen {
        int var_cnt = 0;
        int new_var() { return var_cnt++; }
        const Kernel &kernel;
        std::unordered_map<NodePtr, int> node2var;
        std::unordered_set<NodePtr> visited_vars;
        std::unordered_set<BlockPtr> visited;
        std::string gen_type(TypePtr type) {
            if (type->isa<Float32Type>()) {
                return "float";
            }
            return "void";
        }
        std::string var(NodePtr p) {
            std::ostringstream os;
            // if (node2var.find(p) == node2var.end()) {
            // }
            os << "v" << node2var.at(p);
            return os.str();
        }
        std::string gen_node(const NodePtr &node) {
            // std::cout << typeid(*node).name() << std::endl;
            std::ostringstream os;
            // if (visited_vars.find(node) != visited_vars.end()) {

            //     return "";
            // }
            // visited_vars.insert(node);
            if (node2var.find(node) == node2var.end()) {
                node2var[node] = new_var();
            }
            if (node->isa<VarStore>()) {
                auto st = node->cast<VarStore>();
                os << var(st->dst) << " = " << var(st->src) << "; // var store\n";
                // visited_vars.erase(st->dst);
                return os.str();
            }

            auto e = node->cast<Expr>();
            if (node->isa<ConstantNode>()) {
                auto cst = node->cast<ConstantNode>();
                os << gen_type(e->type) << " " << var(node) << " = " << cst->val << ";\n";
            } else if (node->isa<BinOpNode>()) {
                std::string op;
                auto binop = node->cast<BinOpNode>();
                if (binop->op == BinOpNode::add) {
                    op = "+";
                } else if (binop->op == BinOpNode::sub) {
                    op = "-";
                } else if (binop->op == BinOpNode::mul) {
                    op = "*";
                } else if (binop->op == BinOpNode::div) {
                    op = "/";
                } else {
                    std::abort();
                }
                os << gen_type(e->type) << " " << var(e) << "=" << var(binop->lhs) << op << var(binop->rhs) << ";\n";
            }
            return os.str();
        }
        std::string gen_bb(BB bb) {
            std::ostringstream os;

            for (auto &node : bb->body) {
                os << gen_node(node);
            }

            return os.str();
        }
        std::string gen_block(BlockPtr b) {
            if (b->isa<IfBlock>()) {
                auto if_block = b->cast<IfBlock>();
                std::ostringstream os;
                os << "if((bool)" << var(if_block->cond) << "){\n";
                os << codegen(if_block->if_true);   
                os << "\n}else{\n" << codegen(if_block->if_false) << "}\n";
                return os.str();
            }
            return gen_bb(b->cast<BasicBlock>());
        }
        std::string codegen(BlockPtr b) {
            std::ostringstream os;

            for (; b != nullptr; b = b->next) {
                std::cout << typeid(*b).name() << std::endl;
                os << gen_block(b);
            }
            return os.str();
        }

      public:
        CodeGen(const Kernel &kernel) : kernel(kernel) {}
        std::string codegen() {
            std::ostringstream os;
            os << "__kernel__ void main(){";
            os << codegen(kernel.entry) << "}\n";
            return os.str();
        }
    };
} // namespace akari::compute
int main() {
    using namespace akari::compute;
    Kernel kernel = compile_kernel([]() {
        Var<float> a = 1.0f;
        a = a + a;
        If(a) { a = a + a; }
        Elif(2.0f) { a = a * a; }
        Else { a = a - a; };
    });
    CodeGen gen(kernel);
    std::cout << gen.codegen() << std::endl;
}