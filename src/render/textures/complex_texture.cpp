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
#include <akari/core/plugin.h>
#include <akari/core/spectrum.h>
#include <akari/plugins/complex_texture.h>
#include <akari/render/texture.h>
#include <asmjit/asmjit.h>

namespace akari {
    namespace detail {
        enum Opcode : uint8_t {
            OP_NONE,
            OP_ADD, // ADD A, B, C ; R[A] = R[B] + R[C]
            OP_SUB,
            OP_MUL,
            OP_DIV,
            OP_MOVE, // MOV A B; R[A] = R[B]
            OP_DOT,  // DOT A,B,C; A = B * C
            OP_CROSS,
            OP_CMP_LT,
            OP_CMP_LE,
            OP_CMP_GT,
            OP_CMP_GE,
            OP_CMP_EQ,
            OP_CMP_NE,
            OP_BZ, // BZ A, OFFSET;
            OP_JMP,
            OP_TEXTURE, // TEXTURE A, B, C; R[A] = texture2D(R[B] as Texture, R[C] as TexCoord)
            OP_FLOAT,   // LOAD NEXT 4 BYTES TO A
        };
        using Register = vec4;
        struct Instruction {
            union {
                struct {
                    Opcode op;
                    uint8_t A;
                    union {
                        struct {
                            uint8_t B, C;
                        };
                        uint16_t offset;
                    };
                };
                float asFloat;
            };
            Instruction(float f) : asFloat(f) {}
            Instruction() = default;
            Instruction(Opcode op, uint8_t A, uint8_t B, uint8_t C) : op(op), A(A), B(B), C(C) {}
            Instruction(Opcode op, uint8_t A) : op(op), A(A), offset(0) {}
        };
        static_assert(sizeof(Instruction) == sizeof(uint8_t) * 4);
        struct ExecutionEngine {
            alignas(64) std::array<Register, 64> registers{};
            size_t programSize = 0;
            const uint8_t *program = nullptr;
            uint32_t pc = 0;
            void Run(const ShadingPoint &sp) {
                while (pc < programSize) {
                    const Instruction &instruction =
                        *reinterpret_cast<const Instruction *>(&program[pc * sizeof(Instruction)]);
                    auto op = instruction.op;
                    auto A = instruction.A;
                    auto B = instruction.B;
                    auto C = instruction.C;
                    auto offset = instruction.offset;
                    auto &R = registers;
                    bool incPc = true;
                    switch (op) {
                    case OP_FLOAT: {
                        pc++;
                        R[A] =
                            vec4((*reinterpret_cast<const Instruction *>(&program[pc * sizeof(Instruction)])).asFloat);
                        break;
                    }
                    case OP_ADD:
                        R[A] = R[B] + R[C];
                        break;
                    case OP_NONE:
                        break;
                    case OP_SUB:
                        R[A] = R[B] - R[C];
                        break;
                    case OP_MUL:
                        R[A] = R[B] * R[C];
                        break;
                    case OP_DIV:
                        R[A] = R[B] / R[C];
                        break;
                    case OP_MOVE:
                        R[A] = R[B];
                        break;
                    case OP_DOT:
                        R[A] = vec4(dot(vec3(R[B]), vec3(R[C])));
                        break;
                    case OP_CROSS:
                        R[A] = vec4(cross(vec3(R[B]), vec3(R[C])), R[A].w);
                        break;
                    case OP_CMP_LT:
                        R[A].x = R[B].x < R[C].x;
                        break;
                    case OP_CMP_LE:
                        R[A].x = R[B].x <= R[C].x;
                        break;
                    case OP_CMP_GT:
                        R[A].x = R[B].x > R[C].x;
                        break;
                    case OP_CMP_GE:
                        R[A].x = R[B].x >= R[C].x;
                        break;
                    case OP_CMP_EQ:
                        R[A].x = R[B].x == R[C].x;
                        break;
                    case OP_CMP_NE:
                        R[A].x = R[B].x != R[C].x;
                        break;
                    case OP_BZ:
                        if (!R[A].x) {
                            incPc = false;
                            pc += instruction.offset;
                        }
                        break;
                    case OP_JMP:
                        incPc = false;
                        pc = instruction.offset;
                        break;
                    case OP_TEXTURE: {
                        //
                    } break;
                    }
                    if (incPc) {
                        pc++;
                    }
                }
            }
        };
    } // namespace detail
    class ComplexTexture final : public Texture {
        json program;
        std::vector<detail::Instruction> compiledProgram;

      public:
        ComplexTexture() = default;
        ComplexTexture(const json &prog) : program(prog) {}
        AKR_SER(program)
        AKR_DECL_COMP()
        void commit() override {
            using namespace detail;
            uint8_t id = 0;
            auto compile = [=, &id](const json &expr, auto &&F) -> uint8_t {
                auto ret = id++;
                if (expr.is_number()) {
                    compiledProgram.emplace_back(OP_FLOAT, ret);
                    compiledProgram.emplace_back(expr.get<float>());
                    return ret;
                }
                auto op = expr.at(0);
                if (op == "+") {
                    compiledProgram.emplace_back(OP_ADD, ret, F(expr.at(1), F), F(expr.at(2), F));
                } else if (op == "-") {
                    compiledProgram.emplace_back(OP_SUB, ret, F(expr.at(1), F), F(expr.at(2), F));
                } else if (op == "*") {
                    compiledProgram.emplace_back(OP_MUL, ret, F(expr.at(1), F), F(expr.at(2), F));
                } else if (op == "/") {
                    compiledProgram.emplace_back(OP_DIV, ret, F(expr.at(1), F), F(expr.at(2), F));
                }
                return ret;
            };
            compile(program, compile);
        }
        Spectrum evaluate(const ShadingPoint &sp) const override {
            using namespace detail;
            ExecutionEngine engine;
            engine.programSize = compiledProgram.size();
            engine.program = (const uint8_t *)compiledProgram.data();
            engine.Run(sp);
            return vec3(engine.registers[0]);
        }
    };
    AKR_EXPORT_PLUGIN(ComplexTexture, p) {}
    std::shared_ptr<Texture> CreateComplexTexture(const json &program) {
        return std::make_shared<ComplexTexture>(program);
    }
} // namespace akari