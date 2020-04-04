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
#include <Akari/Core/Plugin.h>
#include <Akari/Core/Spectrum.h>
#include <Akari/Plugins/ComplexTexture.h>
#include <Akari/Render/Texture.h>

namespace Akari {
    namespace detail {
        enum Opcode : uint8_t {
            OP_NONE,
            OP_ADD, // ADD A, B, C ; R[C] = R[A] + R[B]
            OP_SUB,
            OP_MUL,
            OP_DIV,
            OP_MOVE, // MOV A B; R[B] = R[A]
            OP_DOT,  // DOT A,B,C
            OP_CROSS,
            OP_CMP_LT,
            OP_CMP_LE,
            OP_CMP_GT,
            OP_CMP_GE,
            OP_CMP_EQ,
            OP_CMP_NE,
            OP_BZ, // BZ A, OFFSET;
            OP_JMP,
            OP_TEXTURE, // TEXTURE A, B, C; R[C] = texture2D(R[A] as Texture, R[B] as TexCoord)
        };
        using Register = vec4;
        struct Instruction {
            Opcode op;
            uint8_t A;
            union {
                struct {
                    uint8_t B, C;
                };
                uint16_t offset;
            };
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
                    case OP_ADD:
                        R[C] = R[A] + R[B];
                        break;
                    case OP_NONE:
                        break;
                    case OP_SUB:
                        R[C] = R[A] - R[B];
                        break;
                    case OP_MUL:
                        R[C] = R[A] * R[B];
                        break;
                    case OP_DIV:
                        R[C] = R[A] / R[B];
                        break;
                    case OP_MOVE:
                        R[B] = R[A];
                        break;
                    case OP_DOT:
                        R[C] = vec4(dot(vec3(R[A]), vec3(R[B])));
                        break;
                    case OP_CROSS:
                        R[C] = vec4(cross(vec3(R[A]), vec3(R[B])), R[C].w);
                        break;
                    case OP_CMP_LT:
                        R[C].x = R[A].x < R[B].x;
                        break;
                    case OP_CMP_LE:
                        R[C].x = R[A].x <= R[B].x;
                        break;
                    case OP_CMP_GT:
                        R[C].x = R[A].x > R[B].x;
                        break;
                    case OP_CMP_GE:
                        R[C].x = R[A].x >= R[B].x;
                        break;
                    case OP_CMP_EQ:
                        R[C].x = R[A].x == R[B].x;
                        break;
                    case OP_CMP_NE:
                        R[C].x = R[A].x != R[B].x;
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

      public:
        AKR_SER(program)
        AKR_DECL_COMP(ComplexTexture, "ComplexTexture")
        void Commit() override {}
        Spectrum Evaluate(const ShadingPoint &sp) const override { return Spectrum(0); }
    };
    AKR_EXPORT_COMP(ComplexTexture, "Texture")
} // namespace Akari