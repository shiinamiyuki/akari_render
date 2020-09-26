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
#include <akari/core/nodes/material.h>
#include <akari/kernel/material.h>
#include <akari/core/logger.h>
namespace akari {
    AKR_VARIANT class DiffuseMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        Color3f color;
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = arena->alloc<Texture<C>>(ConstantTexture<C>(color));
            return arena->alloc<Material<C>>(DiffuseMaterial<C>(tex));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = load_array<Color3f>(value);
            }
        }
    };
    AKR_VARIANT class GlossyMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        Color3f color;
        Float roughness = Float(0.1);
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = arena->alloc<Texture<C>>(ConstantTexture<C>(color));
            auto tex_rough = arena->alloc<Texture<C>>(ConstantTexture<C>(roughness));
            return arena->alloc<Material<C>>(GlossyMaterial<C>(tex, tex_rough));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = load_array<Color3f>(value);
            }
            if (field == "roughness") {
                roughness = value.get<double>().value();
            }
        }
    };
    AKR_VARIANT class EmissiveMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        bool double_sided = false;
        Color3f color;
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = arena->alloc<Texture<C>>(ConstantTexture<C>(color));
            return arena->alloc<Material<C>>(EmissiveMaterial<C>(tex));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = load_array<Color3f>(value);
            }
        }
    };

    AKR_VARIANT void RegisterMaterialNode<C>::register_nodes() {
        AKR_IMPORT_TYPES()
        register_node<C, DiffuseMaterialNode<C>>("DiffuseMaterial");
        register_node<C, EmissiveMaterialNode<C>>("EmissiveMaterial");
        register_node<C, GlossyMaterialNode<C>>("GlossyMaterial");
    }

    AKR_VARIANT void RegisterMaterialNode<C>::register_python_nodes(py::module &m) {
#ifdef AKR_ENABLE_PYTHON
        py::class_<MaterialNode<C>, SceneGraphNode<C>, std::shared_ptr<MaterialNode<C>>>(m, "Material");
        py::class_<DiffuseMaterialNode<C>, MaterialNode<C>, std::shared_ptr<DiffuseMaterialNode<C>>>(m,
                                                                                                     "DiffuseMaterial")
            .def(py::init<>())
            .def_readwrite("color", &DiffuseMaterialNode<C>::color)
            .def("commit", &DiffuseMaterialNode<C>::commit);
        py::class_<EmissiveMaterialNode<C>, MaterialNode<C>, std::shared_ptr<EmissiveMaterialNode<C>>>(
            m, "EmissiveMaterial")
            .def(py::init<>())
            .def_readwrite("color", &EmissiveMaterialNode<C>::color)
            .def_readwrite("double_sided", &EmissiveMaterialNode<C>::double_sided)
            .def("commit", &EmissiveMaterialNode<C>::commit);
#endif
    }

    AKR_RENDER_STRUCT(RegisterMaterialNode)
} // namespace akari