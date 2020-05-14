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
#include <akari/plugins/complex_texture.h>
#include <asmjit/asmjit.h>
#include <memory>
#if 0
namespace akari::Expr {
    /*
     *  AkariExpr:
     *  ASSIGN_EXPR := 'let' VAR '=' EXPR ';'
     *  FUNCTION_CALL := BUILTIN_FUNCTION '(' [ EXPR ',' ] EXPR ')'
     *  BUILTIN_FUNCTION := one of [texture, perlin,   ]
     *  SELECT_EXPR := 'select' '(' EXPR, EXPR EXPR ')'
     * */
    enum ASTNodeKind {
        EBinaryExpr,
        EVar,
        EBuiltin,
        EFunctionCall,
        ESelect,
    };
    class ASTNode {
      public:
        virtual ASTNodeKind GetKind() const = 0;
    };
    class Expr : public ASTNode {

    };
    class BinaryExpr : public Expr {
      public:
        BinaryExpr(std::shared_ptr<Expr>)
        [[nodiscard]] ASTNodeKind GetKind() const override { return EBinaryExpr; }
    };
    class Var : public Expr {
      public:
        ASTNodeKind GetKind() const override { return EVar; }
        std::shared_ptr<BinaryExpr> operator + (const std::shared_ptr<Expr> & rhs ) {

        }
    };
} // namespace akari::Expr
#endif
int main() {
    using namespace akari;
    const auto source = R"(
["+", 1, 2]
)";
    auto texture = CreateComplexTexture(json::parse(source));
    texture->commit();
    auto res = texture->evaluate(ShadingPoint());
    info("{} {} {}\n",res[0],res[1],res[2]);
}
